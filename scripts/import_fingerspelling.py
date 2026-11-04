"""
SignBridge AI — Import Fingerspelling Dataset (ISL Fingerspelling CC BY-NC 4.0).
Integrates:
Hugging Face kirandevraj/ISL-Fingerspelling — CC BY-NC 4.0
- 1,308 continuous video segments, 14,685 character annotations, 3 genuine signers
- Downloads curated video segments covering all letters A-Z (26 classes)
- Extracts 282-dim MediaPipe hand landmark sequences (30 frames)
- Registers genuine samples in data/isl_signs/landmarks.jsonl
- Pure import pipeline: all model training logic belongs in scripts/train_alphabet.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import urllib.request

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isl_data_collection.config import (
    DATA_ROOT,
    FEATURE_DIM,
    FEATURE_DIR,
    LANDMARK_MANIFEST,
    SEQUENCE_LENGTH,
)
from isl_data_collection.hand_landmarks import ensure_hand_task_model, result_to_features
from isl_data_collection.storage import append_landmark_record, upsert_landmark_record
from backend.vision.feature_schema import resample_sequence, validate_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("import_fingerspelling")

HF_FS_BASE = "https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling"
ANN_URL = f"{HF_FS_BASE}/raw/main/letter_annotations.csv"
SPLIT_URL = f"{HF_FS_BASE}/raw/main/signer_split.csv"
VIDEO_BASE_URL = f"{HF_FS_BASE}/resolve/main/videos"

RAW_FS_DIR = PROJECT_ROOT / "data" / "raw" / "isl_fingerspelling"
VIDEOS_DIR = RAW_FS_DIR / "videos"
ALPHABET_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def download_file(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """Download a file with user-agent header if not already present."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        return False


def stage_annotations_and_select_videos() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Download annotations and select curated video UIDs covering all letters A-Z."""
    RAW_FS_DIR.mkdir(parents=True, exist_ok=True)
    ann_path = RAW_FS_DIR / "letter_annotations.csv"
    split_path = RAW_FS_DIR / "signer_split.csv"

    if not ann_path.exists():
        logger.info("Downloading letter_annotations.csv...")
        download_file(ANN_URL, ann_path)
    if not split_path.exists():
        logger.info("Downloading signer_split.csv...")
        download_file(SPLIT_URL, split_path)

    ann_df = pd.read_csv(ann_path)
    split_df = pd.read_csv(split_path)

    merged = pd.merge(ann_df, split_df, on="uid")
    merged["char"] = merged["letter"].astype(str).str.upper().str.strip()
    merged = merged[merged["char"].isin(ALPHABET_CLASSES)].copy()

    # Select top 3 videos per letter for each split (train and test) to achieve robust class coverage
    uids_needed = set()
    for split_name in ["train", "test"]:
        sub = merged[merged["split"] == split_name]
        for ch in ALPHABET_CLASSES:
            uids_with_ch = sub[sub["char"] == ch]["uid"].unique()
            uids_needed.update(uids_with_ch[:3])

    selected_uids = sorted(list(uids_needed))
    logger.info("Selected %d curated video segments covering all 26 letters across Train & Test splits.",
                len(selected_uids))
    return ann_df, split_df, selected_uids


def download_curated_videos(selected_uids: List[str], max_workers: int = 4) -> Dict[str, Path]:
    """Download the selected video clips concurrently."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_map: Dict[str, Path] = {}

    def fetch_uid(uid: str) -> Tuple[str, Optional[Path]]:
        dest = VIDEOS_DIR / f"{uid}.mp4"
        url = f"{VIDEO_BASE_URL}/{uid}.mp4"
        ok = download_file(url, dest, timeout=40)
        return uid, dest if ok and dest.exists() and dest.stat().st_size > 1000 else None

    logger.info("Checking / downloading %d segment videos from Hugging Face...", len(selected_uids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_uid, uid): uid for uid in selected_uids}
        for future in as_completed(futures):
            uid, path = future.result()
            if path:
                downloaded_map[uid] = path

    logger.info("Successfully staged %d/%d segment videos locally.", len(downloaded_map), len(selected_uids))
    return downloaded_map


def extract_character_features(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    detector: Any,
) -> Optional[np.ndarray]:
    """
    Extract frames for [start_frame, end_frame] and run MediaPipe Tasks HandLandmarker.
    Returns resampled (30, 282) sequence or None.
    """
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    vectors: List[np.ndarray] = []
    current_frame = 0
    hands_detected = 0

    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame

        while current_frame <= end_frame:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            rgb_small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (320, 240))
            rgb_small = np.ascontiguousarray(rgb_small)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            result = detector.detect(image)
            feat, cnt = result_to_features(result)
            vectors.append(feat)
            if cnt > 0:
                hands_detected += 1
            current_frame += 1
    except Exception as exc:
        logger.warning("Error reading frame interval [%d, %d]: %s", start_frame, end_frame, exc)
        return None
    finally:
        cap.release()

    if len(vectors) < 2 or hands_detected == 0:
        return None

    arr = np.stack(vectors).astype(np.float32)
    seq_30 = resample_sequence(arr, target_length=SEQUENCE_LENGTH)
    if not validate_sequence(seq_30):
        return None
    return seq_30


def build_fingerspelling_dataset(
    ann_df: pd.DataFrame,
    split_df: pd.DataFrame,
    downloaded_videos: Dict[str, Path],
) -> List[Dict[str, Any]]:
    """Extract 30-frame landmark sequences for all character instances and register records."""
    import mediapipe as mp
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ensure_hand_task_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    merged = pd.merge(ann_df, split_df, on="uid")
    merged["char"] = merged["letter"].astype(str).str.upper().str.strip()
    merged = merged[merged["char"].isin(ALPHABET_CLASSES)].copy()
    merged = merged[merged["uid"].isin(downloaded_videos.keys())].copy()

    records: List[Dict[str, Any]] = []

    logger.info("Extracting character instances from %d annotations...", len(merged))
    for idx, row in merged.iterrows():
        uid = str(row["uid"])
        ch = str(row["char"])
        signer = str(row["signer"])
        split = str(row["split"])
        start_f = int(row["start_frame"])
        end_f = int(row["end_frame"])
        vpath = downloaded_videos.get(uid)
        if not vpath or not vpath.exists():
            continue

        sample_id = hashlib.md5(f"fs_{uid}_{start_f}_{end_f}_{ch}".encode()).hexdigest()[:12]
        npz_path = FEATURE_DIR / f"alphabet_{sample_id}.npz"

        seq = None
        if npz_path.exists():
            try:
                seq = np.load(npz_path)["features"]
                if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                    seq = None
            except Exception:
                seq = None

        if seq is None:
            seq = extract_character_features(vpath, start_f, end_f, detector)
            if seq is not None:
                np.savez_compressed(npz_path, features=seq)

        if seq is not None:
            rec = {
                "sample_id": sample_id,
                "category": "alphabet",
                "label": ch,
                "signer_id": signer,
                "split": split,
                "video_path": str(vpath.relative_to(PROJECT_ROOT)),
                "features_path": str(npz_path.relative_to(PROJECT_ROOT)),
                "source": "kirandevraj/ISL-Fingerspelling",
                "license": "CC BY-NC 4.0",
                "start_frame": start_f,
                "end_frame": end_f,
            }
            records.append(rec)
            upsert_landmark_record(rec)

    detector.close()
    logger.info("Successfully registered %d character instances across all letters.", len(records))
    return records


def main():
    logger.info("=== STEP: IMPORT ISL FINGERSPELLING DATASET (A–Z) ===")
    ann_df, split_df, selected_uids = stage_annotations_and_select_videos()
    downloaded_videos = download_curated_videos(selected_uids, max_workers=4)
    records = build_fingerspelling_dataset(ann_df, split_df, downloaded_videos)
    logger.info("Fingerspelling import complete: %d records processed. Ready for training via scripts/train_alphabet.py.",
                len(records))


if __name__ == "__main__":
    main()
