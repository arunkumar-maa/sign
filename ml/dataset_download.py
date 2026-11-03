"""
SignBridge AI — Phase 5: ISL Dataset Downloader.
Downloads and verifies the isolated-word dataset from Hugging Face:
https://huggingface.co/datasets/vidit031/isl-isolated-40words
Contains 642 video clips across 40 Indian Sign Language vocabulary classes.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "isl_isolated_40words"
REPO_ID = "vidit031/isl-isolated-40words"
METADATA_URL = f"https://huggingface.co/datasets/{REPO_ID}/raw/main/metadata.csv"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_clip(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
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


def download_and_verify_dataset() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = DATA_DIR / "metadata.csv"

    # 1. Fetch metadata.csv
    if not metadata_path.exists():
        logger.info("Fetching metadata.csv from %s...", METADATA_URL)
        req = urllib.request.Request(METADATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw_csv = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(raw_csv)
    else:
        logger.info("Local metadata.csv found at %s", metadata_path)

    df = pd.read_csv(metadata_path)
    total_clips = len(df)
    logger.info("Metadata records %d clips across %d classes.", total_clips, df["normalized_word"].nunique())

    # 2. Verify / Download clips
    downloaded = 0
    verified = 0

    for idx, row in df.iterrows():
        class_name = str(row["normalized_word"]).strip()
        filename = str(row["file_name"]).strip()
        dest_file = DATA_DIR / class_name / filename

        if not dest_file.exists():
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/videos/{class_name}/{filename}"
            if download_clip(url, dest_file):
                downloaded += 1

        if dest_file.exists() and dest_file.stat().st_size > 1024:
            verified += 1

    summary = {
        "dataset_repo": REPO_ID,
        "total_records": total_clips,
        "newly_downloaded": downloaded,
        "verified_clips": verified,
        "classes_count": df["normalized_word"].nunique(),
        "classes": sorted(df["normalized_word"].unique().tolist()),
    }
    logger.info("Dataset summary: %d verified clips across %d classes.", verified, summary["classes_count"])
    return summary


def main():
    download_and_verify_dataset()


if __name__ == "__main__":
    main()

