#!/usr/bin/env python3
"""
annotation.py  –  batch-annotate videos for aggression / non-aggression
using the GPT-4o vision model.

Run with defaults:
    python annotation.py                 # looks in ./videos, writes to ./labels
or override:
    python annotation.py --video_dir ./clips --out_dir ./my_labels
"""

import argparse, os, json, base64, re, sys
from pathlib import Path
import cv2
from tqdm import tqdm
import openai

# --------------------------- CONFIG & SECRETS ---------------------------- #
OPENAI_API_KEY = "ENV_FILE_OPEN_API_KEY" # located in .env file
SAMPLE_FPS     = 1                # frames per second to sample
CHUNK_SIZE     = 20               # max frames sent per GPT call
MODEL          = "o4-mini"        # keep as-is or change to an official name
# ------------------------------------------------------------------------- #

_api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "OpenAI API key missing. Set OPENAI_API_KEY in code or env."
    )

client = openai.OpenAI(api_key=_api_key)

# ---------- helpers ------------------------------------------------------ #
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$",
                      re.IGNORECASE | re.MULTILINE)

def clean_json_block(raw: str) -> str:
    """Strip ``` and ```json fences + leading / trailing whitespace."""
    return FENCE_RE.sub("", raw).strip()

def encode_frame(frame):
    """JPEG-encode a frame → base64 str."""
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return base64.b64encode(buf).decode("utf-8")

def prompt_for_frames(video_name: str, fps: float, frames_chunk):
    """Craft the multimodal prompt for GPT-4o."""
    system_msg = (
        "You are an expert security analyst. "
        "For each image you receive, decide if it shows a violent/aggressive action "
        "or a non-aggressive, neutral situation. "
        "Return a JSON array with one element per image, preserving order, "
        "where each element is {\"t\": <float sec>, \"label\": \"aggression\"|\"nonaggression\"}."
    )
    user_content = [{
        "type": "text",
        "text": (
            f"Here are {len(frames_chunk)} frames sampled from '{video_name}'. "
            "Respond ONLY with the JSON array described above."
        )
    }]

    for ts, b64 in frames_chunk:
        user_content.append({
            "type": "image_url",
            "image_url": { "url": f"data:image/jpeg;base64,{b64}" }
        })

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]
# ------------------------------------------------------------------------- #

def annotate_video(path: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps
    sample_int   = int(round(fps / SAMPLE_FPS))

    sampled = []
    pbar = tqdm(total=frame_count, desc=f"Sampling {path.name}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_int == 0:
            sampled.append((idx / fps, encode_frame(frame)))
        idx += 1
        pbar.update(1)
    cap.release(); pbar.close()

    # ---------------- query GPT-4o --------------------------------------- #
    labels = []
    for i in tqdm(range(0, len(sampled), CHUNK_SIZE), desc="Annotating"):
        chunk     = sampled[i:i+CHUNK_SIZE]
        messages  = prompt_for_frames(path.name, fps, chunk)
        response  = client.chat.completions.create(model=MODEL, messages=messages)
        raw_reply = response.choices[0].message.content or ""
        try:
            chunk_labels = json.loads(clean_json_block(raw_reply))
            labels.extend(chunk_labels)
        except Exception:
            print("\n[WARN] Could not parse model reply; skipping chunk:\n",
                  raw_reply, "\n", file=sys.stderr)
            continue
    if not labels:
        print(f"[WARN] No labels for {path.name}; skipping output.")
        return
    # --------------- merge contiguous frames ---------------------------- #
    segments, current = [], None
    for item in labels:
        ts, label = float(item["t"]), item["label"]
        if current is None:
            current = {"start": ts, "end": ts, "label": label}
        elif label == current["label"] and abs(ts - current["end"]) <= 1/SAMPLE_FPS + 1e-3:
            current["end"] = ts
        else:
            segments.append(current); current = {"start": ts, "end": ts, "label": label}
    if current:
        segments.append(current)
    for seg in segments:                         # pad end so last frame included
        seg["end"] += 1 / SAMPLE_FPS

    out_data = {
        "video_file"  : path.name,
        "fps"         : fps,
        "duration_sec": duration_sec,
        "segments"    : segments,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Wrote {out_path}")


# ------------------------------ CLI ENTRY ------------------------------- #
def main() -> None:
    """Walk the video directory, annotate each unseen clip, write JSON."""
    parser = argparse.ArgumentParser(
        description="Annotate videos for aggression / non-aggression with GPT-4o"
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("videos"),
        help="Input video folder (default: ./videos)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("labels"),
        help="Output JSON folder (default: ./labels)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos that already have a *_labels.json file",
    )
    args = parser.parse_args()

    # make sure the output directory exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    supported_ext = {".mp4", ".mov", ".avi", ".mkv"}
    videos = sorted(
        [p for p in args.video_dir.iterdir() if p.suffix.lower() in supported_ext]
    )

    for vp in videos:
        out_file = args.out_dir / f"{vp.stem}_labels.json"
        if args.resume and out_file.exists():
            print(f"✔  {out_file.name} exists – skipping")
            continue

        annotate_video(vp, args.out_dir)


if __name__ == "__main__":
    main()

