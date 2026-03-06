"""
Step 1 - extract_frames.py
Extracts frames from local video files or YouTube URLs.

Usage:
  python extract_frames.py --source data/videos/gameplay.mp4
  python extract_frames.py --url "https://www.youtube.com/watch?v=XXXX"
  python extract_frames.py --source data/videos/
  python extract_frames.py --source data/videos/ --max-frames 5000
"""

import os, cv2, argparse, subprocess, sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def resize_frame(frame, w=config.FRAME_W, h=config.FRAME_H):
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def frames_from_video(video_path, out_dir, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step    = max(1, round(src_fps / config.EXTRACT_FPS))
    os.makedirs(out_dir, exist_ok=True)
    saved = src_idx = 0
    pbar = tqdm(total=min(total // step, max_frames or 10**9),
                desc=f"  {Path(video_path).name}", unit="frame", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret: break
        if src_idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{saved:07d}.jpg"),
                        resize_frame(frame), [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
            pbar.update(1)
            if max_frames and saved >= max_frames: break
        src_idx += 1
    pbar.close()
    cap.release()
    return saved


def download_youtube(url, out_path):
    print(f"  Downloading: {url}")
    try:
        subprocess.run(["yt-dlp", "-f",
            "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]",
            "--merge-output-format", "mp4", "-o", out_path, "--no-playlist", url],
            capture_output=True, text=True, check=True)
        base = out_path.replace(".mp4", "")
        for ext in [".mp4", ".mkv", ".webm"]:
            if os.path.exists(base + ext): return base + ext
        return out_path if os.path.exists(out_path) else None
    except FileNotFoundError:
        print("  [!] yt-dlp not found. Install: pip install yt-dlp")
    except subprocess.CalledProcessError as e:
        print(f"  [!] yt-dlp error: {e.stderr.strip()}")
    return None


def process_video(video_path, max_frames=None):
    stem    = Path(video_path).stem
    out_dir = os.path.join(config.FRAMES_DIR, stem)
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        n = len(os.listdir(out_dir))
        print(f"  [{stem}] Already extracted ({n} frames). Skipping.")
        return n
    n = frames_from_video(video_path, out_dir, max_frames)
    print(f"  [{stem}] Saved {n} frames.")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     type=str, default=None)
    parser.add_argument("--url",        type=str, default=None)
    parser.add_argument("--urls-file",  type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    total = 0

    if args.url:
        idx  = len(os.listdir(config.VIDEOS_DIR))
        dest = os.path.join(config.VIDEOS_DIR, f"yt_{idx:03d}.mp4")
        path = download_youtube(args.url, dest)
        if path: total += process_video(path, args.max_frames)

    if args.urls_file:
        with open(args.urls_file) as f:
            urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        for i, url in enumerate(urls):
            dest = os.path.join(config.VIDEOS_DIR, f"yt_{i:03d}.mp4")
            path = download_youtube(url, dest)
            if path: total += process_video(path, args.max_frames)

    if args.source:
        src = Path(args.source)
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        videos = [str(src)] if src.is_file() else \
                 sorted([str(p) for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS]) \
                 if src.is_dir() else []
        print(f"\nFound {len(videos)} video(s)")
        for v in videos:
            total += process_video(v, args.max_frames)

    if total > 0:
        print(f"\nDone. Total frames: {total}")

if __name__ == "__main__":
    main()
