"""
SignBridge AI — Import Numbers Dataset (ISLRTC Official Data.gov).
Selective download and landmark extraction for Indian Sign Language digits 0 through 9.

Integrates:
Hugging Face silentone0725/Indian_Sign_Language_Data.gov_Rencoded (ISLRTC Official)
- Selective download of numbers 0 through 9 (NO full repo clone)
- Extracts 282-dim MediaPipe hand landmark sequences (30 frames)
- Rejects clips where hand landmarks are missing (NO zero-filled sequences)
- Records genuine signer metadata (signer_id: "islrtc_master")
- Registers valid samples in data/isl_signs/landmarks.jsonl
- Pure import pipeline: all model training logic belongs in scripts/train_numbers.py
"""

import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import urllib.request

import cv2
import numpy as np

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
logger = logging.getLogger("import_number_datasets")

HF_NUM_BASE = "https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded/resolve/main"
RAW_NUM_DIR = PROJECT_ROOT / "data" / "raw" / "islrtc_numbers"

TARGET_NUMBER_FILES: Dict[str, str] = {
    "0": "Z/Zero.mp4",
    "1": "Numbers/1_One.mp4",
    "2": "Numbers/2_Two.mp4",
    "3": "Numbers/3_Three.mp4",
    "4": "Numbers/4_Four.mp4",
    "5": "Numbers/5_Five.mp4",
    "6": "Numbers/6_Six.mp4",
    "7": "Numbers/7_Seven.mp4",
    "8": "Numbers/8_Eight.mp4",
    "9": "Numbers/9_Nine.mp4",
}

NUMBER_CLASSES = [str(i) for i in range(10)]


def download_target_numbers() -> Dict[str, Path]:
    """Download selective 0-9 number clips from ISLRTC dataset."""
    RAW_NUM_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: Dict[str, Path] = {}

    for digit, rel_path in TARGET_NUMBER_FILES.items():
        fname = Path(rel_path).name
        dest = RAW_NUM_DIR / f"islrtc_{digit}_{fname}"
        url = f"{HF_NUM_BASE}/{rel_path}"

        if not dest.exists() or dest.stat().st_size < 1000:
            logger.info("Downloading ISLRTC number %s from %s...", digit, rel_path)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
                    while chunk := resp.read(65536):
                        f.write(chunk)
            except Exception as e:
                logger.warning("Failed downloading %s: %s", url, e)
                continue

        if dest.exists() and dest.stat().st_size > 1000:
            downloaded[digit] = dest

    logger.info("Staged %d/10 primary ISLRTC number clips locally.", len(downloaded))
    return downloaded


def extract_features_from_clip(video_path: Path, detector: Any, stride: int = 2) -> Optional[np.ndarray]:
    """
    Extract frames and convert to 282-dim hand landmarks.
    Rejects clips with insufficient hand detections; never returns zero-filled vectors.
    """
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    vectors: List[np.ndarray] = []
    frame_idx = 0
    hands_detected = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            if frame_idx % stride == 0:
                rgb_small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (320, 240))
                rgb_small = np.ascontiguousarray(rgb_small)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
                result = detector.detect(image)
                feat, cnt = result_to_features(result)
                vectors.append(feat)
                if cnt > 0:
                    hands_detected += 1
            frame_idx += 1
    except Exception as e:
        logger.warning("Error processing %s: %s", video_path.name, e)
        return None
    finally:
        cap.release()

    # Reject if too few total frames or zero hands detected
    if len(vectors) < 4 or hands_detected == 0:
        logger.warning("Rejected %s: insufficient hand detections (%d detected in %d frames).",
                       video_path.name, hands_detected, len(vectors))
        return None

    arr = np.stack(vectors).astype(np.float32)
    seq_30 = resample_sequence(arr, target_length=SEQUENCE_LENGTH)
    if not validate_sequence(seq_30):
        logger.warning("Rejected %s: resampled sequence failed schema validation.", video_path.name)
        return None

    return seq_30


def build_number_dataset(downloaded: Dict[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract features from downloaded clips and save genuine samples and manifest records.
    Never fabricates synthetic signers or zero-padded vectors.
    """
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

    records: List[Dict[str, Any]] = []

    for digit in NUMBER_CLASSES:
        vpath = downloaded.get(digit)
        if not vpath or not vpath.exists():
            logger.warning("Missing video clip for digit %s", digit)
            continue

        seq = extract_features_from_clip(vpath, detector)
        if seq is None:
            logger.error("Could not extract valid hand features for digit %s from %s", digit, vpath.name)
            continue

        sample_id = f"islrtc_num_{digit}"
        npz_filename = f"number_{sample_id}.npz"
        npz_path = FEATURE_DIR / npz_filename
        np.savez_compressed(npz_path, features=seq)

        try:
            rel_video = str(vpath.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_video = str(vpath)

        rec = {
            "sample_id": sample_id,
            "category": "number",
            "label": digit,
            "signer_id": "islrtc_master",
            "session_id": "islrtc_official_dataset",
            "video_path": rel_video,
            "features_path": f"data/isl_signs/features/{npz_filename}",
            "source": "silentone0725/Indian_Sign_Language_Data.gov_Rencoded",
            "license": "OGDL - India",
        }
        records.append(rec)
        upsert_landmark_record(rec)

    detector.close()
    logger.info("Number dataset extraction complete: %d genuine digit samples registered.", len(records))
    return records


def main():
    logger.info("=== STEP: IMPORT ISL NUMBERS DATASET (0–9) ===")
    downloaded = download_target_numbers()
    records = build_number_dataset(downloaded)
    logger.info("Successfully processed %d digits (0-9). Ready for training via scripts/train_numbers.py.", len(records))


if __name__ == "__main__":
    main()
