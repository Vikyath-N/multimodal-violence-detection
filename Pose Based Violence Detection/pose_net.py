#!/usr/bin/env python3
"""
Phase-1 annotation: sample frames, get pose (MediaPipe) + violence labels (GPT-4o).

Run:
    python annotation.py --video_dir videos --out_dir labels
"""

import argparse, os, json, base64, re, sys
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
import openai
import mediapipe as mp
from dotenv import load_dotenv                     # pip install python-dotenv

# --------------------------- CONFIG & SECRETS ---------------------------- #
load_dotenv()                                      # reads .env at repo root
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAMPLE_FPS     = 1          # frames per second to sample
CHUNK_SIZE     = 20         # max frames per GPT call
MODEL          = "o4-mini"
# ------------------------------------------------------------------------- #
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env or environment")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)

# ---------- helpers ------------------------------------------------------ #
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$",
                      re.IGNORECASE | re.MULTILINE)

def clean_json_block(raw: str) -> str:
    return FENCE_RE.sub("", raw).strip()

def encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buf).decode()

def extract_pose(frame):
    """Return 33 × 4 array: (x,y,z,vis); coords normalised to [0,1]."""
    h, w = frame.shape[:2]
    res  = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return np.zeros((33, 4), dtype=np.float32)
    lmks = res.pose_landmarks.landmark
    out  = np.zeros((33, 4), dtype=np.float32)
    for i, p in enumerate(lmks):
        out[i] = [p.x, p.y, p.z, p.visibility]
    return out

def prompt_for_frames(video_name: str, frames_chunk):
    system_msg = (
        "You are an expert security analyst. "
        "For each IMAGE you receive, along with its POSE JSON, "
        "label it 'aggression' or 'nonaggression'. "
        "Return a JSON array keeping the same order: "
        "[{\"t\": <sec>, \"label\": \"aggression|nonaggression\"}, …]"
    )
    user_content = [{
        "type": "text",
        "text": (f"{len(frames_chunk)} frames from '{video_name}'. "
                 "JSON only, no extra text.")
    }]
    for ts, b64, pose_j in frames_chunk:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        user_content.append({
            "type": "text",
            "text": f"pose = {pose_j}"
        })

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]
# ------------------------------------------------------------------------- #

def annotate_video(path: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(path))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = frame_count / fps
    sample_int  = int(round(fps / SAMPLE_FPS))

    sampled, poses = [], []
    pbar = tqdm(total=frame_count, desc=f"Sampling {path.name}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_int == 0:
            ts = idx / fps
            pose_arr  = extract_pose(frame)
            pose_json = json.dumps(pose_arr.tolist())  # send lightweight JSON
            sampled.append((ts, encode_frame(frame), pose_json))
            poses.append(pose_arr)
        idx += 1
        pbar.update(1)
    cap.release(); pbar.close()

    # --------------- GPT-4o annotation ---------------------------------- #
    labels = []
    for i in tqdm(range(0, len(sampled), CHUNK_SIZE), desc="Annotating"):
        chunk = sampled[i:i+CHUNK_SIZE]
        resp  = client.chat.completions.create(
                    model=MODEL,
                    messages=prompt_for_frames(path.name, chunk))
        raw   = resp.choices[0].message.content or ""
        try:
            labels.extend(json.loads(clean_json_block(raw)))
        except Exception:
            print("\n[WARN] parse error – chunk skipped\n", raw, file=sys.stderr)

    if not labels:
        print(f"[WARN] No labels for {path.name}")
        return

    # ----------- merge consecutive labels into segments ----------------- #
    segments, cur = [], None
    for item in labels:
        ts, lab = float(item["t"]), item["label"]
        if cur is None:
            cur = {"start": ts, "end": ts, "label": lab}
        elif lab == cur["label"] and abs(ts - cur["end"]) <= 1/SAMPLE_FPS + 1e-3:
            cur["end"] = ts
        else:
            segments.append(cur); cur = {"start": ts, "end": ts, "label": lab}
    if cur: segments.append(cur)
    for s in segments: s["end"] += 1/SAMPLE_FPS   # include last frame

    # --------------------- write artefacts ------------------------------ #
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{path.stem}_labels.json").write_text(
        json.dumps({
            "video_file": path.name,
            "fps": fps,
            "duration_sec": duration,
            "segments": segments}, indent=2))

    np.savez_compressed(out_dir / f"{path.stem}_pose.npz",
                        poses=np.stack(poses), fps=fps)
    print(f"✔ {path.stem}: labels + pose saved")


# ------------------------------ CLI ------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", type=Path, default="videos")
    ap.add_argument("--out_dir",   type=Path, default="labels")
    ap.add_argument("--resume",    action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vids = [p for p in args.video_dir.iterdir() if p.suffix.lower() in {
            ".mp4", ".mov", ".avi", ".mkv"}]

    for v in sorted(vids):
        lab_file = args.out_dir / f"{v.stem}_labels.json"
        if args.resume and lab_file.exists():
            print(f"⏩  {v.stem} done – skip")
            continue
        annotate_video(v, args.out_dir)

if __name__ == "__main__":
    main()
