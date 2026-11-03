"""
SignBridge AI — Phase 5: Sequence Builder & Signer-Partitioned Splitting.
Processes all video clips in data/raw/isl_isolated_40words, normalizes sequences to 30 frames,
partitions strictly by signer ID to avoid data leakage, and outputs numpy arrays for training.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from backend.vision.hand_landmarks import HandLandmarker
from ml.extract_landmarks import extract_landmarks_from_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "isl_isolated_40words"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "isl40"


def get_signer_id(filename: str) -> str:
    """Extract signer / session ID from video filename."""
    m_sess = re.search(r"session(\d+)", filename, re.IGNORECASE)
    if m_sess:
        return f"sess_{m_sess.group(1)}"
    m_cislr = re.search(r"CISLR__(\d+)", filename, re.IGNORECASE)
    if m_cislr:
        return f"cislr_{m_cislr.group(1)}"
    m_s = re.search(r"-s(\d+)-", filename, re.IGNORECASE)
    if m_s:
        return f"signer_{m_s.group(1)}"
    return f"signer_{abs(hash(filename)) % 15}"


def build_and_save_dataset():
    """Build and save dataset if not already present."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if pre-existing processed dataset exists and is complete
    x_train_path = OUT_DIR / "X_train.npy"
    if x_train_path.exists():
        logger.info("Processed dataset already exists at %s. Loading existing arrays...", OUT_DIR)
        X_train = np.load(OUT_DIR / "X_train.npy")
        X_val = np.load(OUT_DIR / "X_val.npy")
        X_test = np.load(OUT_DIR / "X_test.npy")
        y_train = np.load(OUT_DIR / "y_train.npy")
        y_val = np.load(OUT_DIR / "y_val.npy")
        y_test = np.load(OUT_DIR / "y_test.npy")
        logger.info("Loaded dataset: Train %s, Val %s, Test %s", X_train.shape, X_val.shape, X_test.shape)
        return

    logger.info("Building sequences from raw video files...")
    # Read classes
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    class_to_id = {c: i for i, c in enumerate(classes)}

    landmarker = HandLandmarker()
    samples = []

    for c in classes:
        clips = list((DATA_DIR / c).glob("*.mp4"))
        for clip in clips:
            seq = extract_landmarks_from_video(clip, landmarker, num_frames=30)
            if seq is not None and seq.shape == (30, 192):
                samples.append({
                    "sequence": seq,
                    "label_idx": class_to_id[c],
                    "class_name": c,
                    "signer_id": get_signer_id(clip.name),
                })

    landmarker.close()
    logger.info("Successfully extracted %d sequences across %d classes.", len(samples), len(classes))

    # Split by signer
    from ml.preprocessing import split_by_signer
    train_s, val_s, test_s = split_by_signer(samples, val_ratio=0.15, test_ratio=0.15)

    X_train = np.array([s["sequence"] for s in train_s], dtype=np.float32)
    y_train = np.array([s["label_idx"] for s in train_s], dtype=np.int32)

    X_val = np.array([s["sequence"] for s in val_s], dtype=np.float32)
    y_val = np.array([s["label_idx"] for s in val_s], dtype=np.int32)

    X_test = np.array([s["sequence"] for s in test_s], dtype=np.float32)
    y_test = np.array([s["label_idx"] for s in test_s], dtype=np.int32)

    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "X_val.npy", X_val)
    np.save(OUT_DIR / "y_val.npy", y_val)
    np.save(OUT_DIR / "X_test.npy", X_test)
    np.save(OUT_DIR / "y_test.npy", y_test)

    summary = {
        "num_classes": len(classes),
        "classes": classes,
        "class_to_id": class_to_id,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "sequence_shape": [30, 192],
    }
    with open(OUT_DIR / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved processed dataset to %s", OUT_DIR)


if __name__ == "__main__":
    build_and_save_dataset()

