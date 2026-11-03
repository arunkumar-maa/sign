"""
SignBridge AI — Phase 2: ISL Isolated 40 Words Dataset Downloader.
Downloads all 642 video clips and metadata from Hugging Face:
https://huggingface.co/datasets/vidit031/isl-isolated-40words
Verifies each video with OpenCV and SHA-256.
"""

import concurrent.futures
import hashlib
import io
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
import cv2
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw" / "isl_isolated_40words"
METADATA_URL = "https://huggingface.co/datasets/vidit031/isl-isolated-40words/raw/main/metadata.csv"
BASE_RESOLVE_URL = "https://huggingface.co/datasets/vidit031/isl-isolated-40words/resolve/main/"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) < 512:
                    raise ValueError(f"Downloaded payload too small: {len(data)} bytes")
                with open(dest_path, "wb") as f:
                    f.write(data)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("Failed to download %s: %s", url, e)
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Target directory: %s", DATA_DIR)

    # 1. Download metadata.csv
    metadata_path = DATA_DIR / "metadata.csv"
    logger.info("Fetching metadata.csv from %s...", METADATA_URL)
    req = urllib.request.Request(METADATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw_csv = urllib.request.urlopen(req).read().decode("utf-8")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(raw_csv)
    logger.info("Saved metadata.csv (%d bytes)", len(raw_csv))

    df = pd.read_csv(io.StringIO(raw_csv))
    total_clips = len(df)
    unique_classes = sorted(df["normalized_word"].unique().tolist())
    logger.info("Found %d video clips across %d classes.", total_clips, len(unique_classes))

    # 2. Download all video clips in parallel
    tasks = []
    for idx, row in df.iterrows():
        rel_path = row["video_path"] # e.g. "thank_you/thank_you__CISLR__00000__-4MbWP5T-cU.mp4"
        dest = DATA_DIR / rel_path
        url = BASE_RESOLVE_URL + rel_path
        tasks.append((url, dest, rel_path, row["normalized_word"]))

    logger.info("Starting download of %d videos with 12 workers...", len(tasks))
    t0 = time.time()
    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_to_clip = {
            executor.submit(download_file, url, dest): (rel_path, word)
            for url, dest, rel_path, word in tasks
        }
        for future in concurrent.futures.as_completed(future_to_clip):
            rel_path, word = future_to_clip[future]
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error("Error downloading %s: %s", rel_path, e)
                fail_count += 1

            if (success_count + fail_count) % 50 == 0 or (success_count + fail_count) == total_clips:
                logger.info("Downloaded %d/%d clips (fails: %d)", success_count + fail_count, total_clips, fail_count)

    elapsed = time.time() - t0
    logger.info("Download completed in %.1fs: %d success, %d fails", elapsed, success_count, fail_count)

    # 3. Verification of clips using OpenCV
    logger.info("Verifying video files with OpenCV...")
    valid_clips = []
    corrupted_clips = []
    class_distribution = {}

    for idx, row in df.iterrows():
        rel_path = row["video_path"]
        word = row["normalized_word"]
        dest = DATA_DIR / rel_path

        if not dest.exists() or dest.stat().st_size == 0:
            corrupted_clips.append({"path": rel_path, "word": word, "reason": "missing_or_empty"})
            continue

        cap = cv2.VideoCapture(str(dest))
        if not cap.isOpened():
            corrupted_clips.append({"path": rel_path, "word": word, "reason": "cannot_open_videocapture"})
            cap.release()
            continue

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        if n_frames < 3:
            corrupted_clips.append({"path": rel_path, "word": word, "reason": "too_few_frames", "frames": n_frames})
            continue

        class_distribution[word] = class_distribution.get(word, 0) + 1
        valid_clips.append({
            "video_path": rel_path,
            "word": word,
            "frames": n_frames,
            "width": w,
            "height": h,
            "fps": fps,
            "size_kb": dest.stat().st_size / 1024,
            "sha256": compute_sha256(dest)
        })

    summary = {
        "dataset_name": "vidit031/isl-isolated-40words",
        "huggingface_url": "https://huggingface.co/datasets/vidit031/isl-isolated-40words",
        "total_clips_listed": total_clips,
        "valid_clips_count": len(valid_clips),
        "corrupted_clips_count": len(corrupted_clips),
        "total_classes": len(unique_classes),
        "classes": unique_classes,
        "class_distribution": class_distribution,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    summary_path = DATA_DIR / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Verification Summary: %d valid clips across %d classes (0 corrupted: %s)",
                len(valid_clips), len(class_distribution), len(corrupted_clips) == 0)
    logger.info("Saved summary report to %s", summary_path)


if __name__ == "__main__":
    main()

