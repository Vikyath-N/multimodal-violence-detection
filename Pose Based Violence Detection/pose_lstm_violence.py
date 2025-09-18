#!/usr/bin/env python
"""
pose_lstm_violence.py
=====================
End-to-end pose-based aggression / non-aggression classifier for body-cam clips.

Folder layout
-------------
dataset/
├── videos/             # *.mp4
├── ann/                # *.json   (schema shown in the prompt)
└── cache/              # auto-generated .npz pose files

Usage
-----
$ python pose_lstm_violence.py --root ./dataset --epochs 20 --clip_len 60
"""
import json, os, argparse, pickle, math
from pathlib import Path
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

# --------------------------- 1.  configuration --------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--root",      type=str, default="dataset")
parser.add_argument("--clip_len",  type=int, default=60,   help="frames / window")
parser.add_argument("--stride",    type=int, default=30,   help="window shift")
parser.add_argument("--epochs",    type=int, default=20)
parser.add_argument("--lr",        type=float, default=1e-4)
parser.add_argument("--batch",     type=int, default=32)
parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

ROOT = Path(args.root)
VIDEO_DIR  = ROOT / "videos"
ANN_DIR    = ROOT / "ann"
CACHE_DIR  = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# --------------------------- 2.  annotation utils ------------------------ #
def load_ann(path):
    """Return (duration_sec, fps, aggressive_ranges:list[(s,e)])"""
    meta = json.loads(Path(path).read_text())
    fps  = meta["fps"]
    dur  = meta["duration_sec"]
    aggr = [(seg["start"], seg["end"])
            for seg in meta["segments"]
            if seg["label"] == "aggression"]
    return dur, fps, aggr

def make_frame_flags(dur, fps, ranges):
    """Binary array length Nframes with 1 inside any aggressive range."""
    n = int(math.ceil(dur * fps))
    flags = np.zeros(n, dtype=np.uint8)
    for (s, e) in ranges:
        s_idx, e_idx = int(s * fps), int(e * fps)
        flags[s_idx:e_idx] = 1
    return flags

# --------------------------- 3.  pose extraction ------------------------- #
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

def extract_pose(video_path, fps_target=None):
    """Return ndarray: T × 33 × 4 (x,y,z,vis). x,y in [0,1]."""
    cap = cv2.VideoCapture(str(video_path))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    raw = []
    with mp_pose as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            if fps_target:                       # optional temporal down-sample
                cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                native_fps = cap.get(cv2.CAP_PROP_FPS)
                if cur % int(native_fps / fps_target): continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lmks = res.pose_landmarks.landmark
                raw.append([(p.x, p.y, p.z, p.visibility) for p in lmks])
            else:                                # all zeros if not detected
                raw.append(np.zeros((33,4)))
    cap.release()
    arr = np.asarray(raw, dtype=np.float32)
    # centre coordinates around hip midpoint (slide 10)
    if len(arr):
        hips = (arr[:, 23, :2] + arr[:, 24, :2]) / 2.0      # left/right hip
        arr[:, :, :2] -= hips[:, None, :]
    return arr

def cache_pose_and_flags(ann_path):
    """Compute poses + flags once per video, store in .npz."""
    dur, fps, ranges = load_ann(ann_path)
    json_name = Path(ann_path).stem
    cache_file = CACHE_DIR / f"{json_name}.npz"
    if cache_file.exists():
        return cache_file
    video_file = json.loads(Path(ann_path).read_text())["video_file"]
    poses = extract_pose(VIDEO_DIR / video_file)
    flags = make_frame_flags(dur, fps, ranges)
    np.savez_compressed(cache_file, poses=poses, flags=flags, fps=fps)
    return cache_file

# --------------------------- 4.  dataset --------------------------------- #
class PoseWindowDataset(Dataset):
    def __init__(self, cache_files, clip_len=60, stride=30):
        self.samples = []                          # list[(np.ndarray,label)]
        for cf in cache_files:
            dat = np.load(cf)
            poses, flags = dat["poses"], dat["flags"]
            T, F = poses.shape[0], poses.shape[1]*poses.shape[2]
            poses = poses.reshape(T, F)            # flatten keypoints
            for i in range(0, T-clip_len+1, stride):
                clip = poses[i:i+clip_len]
                tgt  = flags[i:i+clip_len]
                y    = 1 if tgt.mean() > .3 else 0
                self.samples.append((clip.astype(np.float32), y))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

# --------------------------- 5.  model ----------------------------------- #
class ViolenceLSTM(nn.Module):
    def __init__(self, feat_dim=132, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers,
                            batch_first=True, bidirectional=True, dropout=.3)
        self.head = nn.Sequential(nn.Linear(hidden*2, 64),
                                  nn.ReLU(), nn.Dropout(.3),
                                  nn.Linear(64, 1))
    def forward(self, x):              # x: B × T × F
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.head(h).squeeze(1)

# --------------------------- 6.  train / eval ---------------------------- #
def main():
    # 6-a  build cache (pose extraction is the slow step)
    cache_files = [cache_pose_and_flags(p) for p in ANN_DIR.glob("*.json")]

    # 6-b  dataset & splits
    full_ds = PoseWindowDataset(cache_files, args.clip_len, args.stride)
    len_val = int(0.15 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-len_val, len_val])
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4)

    # 6-c  model
    feat_dim = 33*4
    model = ViolenceLSTM(feat_dim).to(args.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.]).to(args.device))

    for epoch in range(1, args.epochs+1):
        model.train()
        tot = 0
        for xb, yb in tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()
            tot += loss.item()*len(xb)
        print(f"  train loss: {tot/len(train_ds):.4f}")

        # validation
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                logits = model(xb.to(args.device)).cpu()
                preds += torch.sigmoid(logits).tolist()
                gts   += yb.tolist()
        auc = roc_auc_score(gts, preds)
        print(f"  val AUC: {auc:.3f}")

    torch.save(model.state_dict(), ROOT / "pose_lstm.pt")
    print("Model saved ➜ pose_lstm.pt")

    # 6-d  frame-wise inference logits for late fusion
    model.eval()
    with torch.no_grad():
        for cf in cache_files:
            dat = np.load(cf)
            poses = torch.from_numpy(dat["poses"].reshape(dat["poses"].shape[0], -1))
            windows, logits = [], np.zeros(len(poses))
            T = len(poses); CL = args.clip_len
            for i in range(0, T-CL+1, args.stride):
                win = poses[i:i+CL].unsqueeze(0).to(args.device)
                p = torch.sigmoid(model(win)).cpu().item()
                logits[i+CL//2] = p                # centre timestamp
            np.save(cf.with_suffix(".pose_logit.npy"), logits.astype(np.float16))
    print("Per-frame logits exported for late fusion.")

#     json_obj = video_numpy_to_episode_json(
#     vid_name = "my_clip.mp4",
#     frames   = frames,
#     fps      = fps,
#     model    = model,
#     clip_len = args.clip_len,
#     stride   = args.stride,
#     threshold=0.60,
#     device   = args.device,
# )

    # with open("my_clip_episode.json", "w") as f:
    #     json.dump(json_obj, f, indent=2)
    # print("Saved my_clip_episode.json")

# ------------------------------------------------------------------ #
# 7.  single-video JSON export                                       #
# ------------------------------------------------------------------ #
def video_numpy_to_episode_json(
        vid_name: str,
        frames: np.ndarray,                 # T × H × W × 3 (uint8 / float)
        fps: float,
        model: nn.Module,
        clip_len: int = 60,
        stride: int = 30,
        threshold: float = 0.60,
        device: str = "cpu" # switch to CUDA or refrcator for CARC system 
    ):
    """
    Extract pose → ViolenceLSTM → return one-episode JSON like:

    {
      "duration_sec": ...,
      "aggression_episode": {
        "start": ...,
        "end": ...,
        "threshold": 0.60,
        "detections": [ {t, pose[], embedding[], violence_score}, … ]
      }
    }
    """
    # 1. pose for each frame ------------------------------------------------
    T = len(frames)
    poses = np.zeros((T, 33, 4), dtype=np.float32)
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        for i, f in enumerate(frames):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                poses[i] = [(p.x, p.y, p.z, p.visibility) 
                            for p in res.pose_landmarks.landmark]

    # center xy around hip-midpoint as in training
    if T:
        hips = (poses[:, 23, :2] + poses[:, 24, :2]) / 2.0
        poses[:, :, :2] -= hips[:, None, :]

    # 2. per-frame violence score via sliding windows ----------------------
    model.eval()
    poses_flat = poses.reshape(T, -1)              # T × 132
    scores = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, T - clip_len + 1, stride):
            clip = torch.from_numpy(
                poses_flat[i : i + clip_len]
            ).unsqueeze(0).to(device)              # 1 × clip_len × 132
            prob = torch.sigmoid(model(clip)).item()
            centre = i + clip_len // 2
            scores[centre] = prob

    # 3. find first aggression episode ≥ threshold -------------------------
    above = scores >= threshold
    episodes = []
    start = None
    for idx, flag in enumerate(above):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            episodes.append((start, idx - 1))
            start = None
    if start is not None:
        episodes.append((start, T - 1))
    if not episodes:
        return {"duration_sec": T / fps, "aggression_episode": None}

    s_idx, e_idx = episodes[0]
    dets = []
    for t_idx in range(s_idx, e_idx + 1):
        dets.append({
            "t": round(t_idx / fps, 2),
            "pose": [
                {
                    "name": kpt_name,
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                    "z": round(float(z), 3),
                    "vis": round(float(v), 2)
                }
                for (kpt_name, (x, y, z, v)) in zip(
                    mp.solutions.pose.POSE_LANDMARKS,
                    poses[t_idx]
                )
            ],
            "embedding": [
                round(float(v), 3) for v in poses_flat[t_idx].tolist()
            ],
            "violence_score": round(float(scores[t_idx]), 3)
        })

    episode_json = {
        "duration_sec": round(T / fps, 2),
        "aggression_episode": {
            "start": round(s_idx / fps, 2),
            "end":   round(e_idx / fps, 2),
            "threshold": threshold,
            "detections": dets,
        }
    }
    return episode_json


if __name__ == "__main__":
    main()
