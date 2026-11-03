"""
SignBridge AI — High Performance Hand Pose Feature Extraction Pipeline.
Extracts 216-dimensional hand pose features from 642 ISL 40-word video clips.
STRICTLY Hand-Only:
- Absolutely NO full body pose landmarks, shoulder tracking, elbow tracking, or torso tracking.
- Absolutely NO facial landmarks or face mesh tracking.
- Only MediaPipe Hand Landmarker (up to 2 hands, 21 landmarks each).
- Uses extract_two_hand_features from backend.vision.hand_pose_features.
- Produces 30-frame sequence arrays (N, 30, 216) and single-frame static arrays (M, 216).
- Saves processed datasets to data/processed/hand_pose/
- Saves label dictionary to models/hand_pose/labels.json
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision.hand_pose_features import (
    extract_two_hand_features,
    TWO_HAND_FEATURE_DIM,
)
from ml.hand_pose_dataset import (
    HandPoseDataset,
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    MODEL_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 30


class VideoHandPoseExtractor:
    """Extracts MediaPipe Hand Landmarker features from video clips into (30, 216) sequences."""

    def __init__(self, min_confidence: float = 0.3):
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    def close(self):
        try:
            self.mp_hands.close()
        except Exception:
            pass

    def extract_from_video(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Sample 30 uniform frames from video clip and extract (30, 216) hand pose feature array.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        # Determine uniform 30 sampling indices
        if total_frames >= SEQUENCE_LENGTH:
            sample_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
        else:
            sample_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)

        target_set = set(sample_indices)
        frames_dict = {}
        curr = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if curr in target_set:
                frames_dict[curr] = frame
            curr += 1

        cap.release()

        if not frames_dict:
            return None

        sampled_frames = [frames_dict[idx] for idx in sample_indices if idx in frames_dict]
        if len(sampled_frames) < SEQUENCE_LENGTH:
            if len(sampled_frames) == 0:
                return None
            resample_idx = np.linspace(0, len(sampled_frames) - 1, SEQUENCE_LENGTH, dtype=int)
            sampled_frames = [sampled_frames[i] for i in resample_idx]

        sequence_features = []
        last_left_lms = None
        last_right_lms = None

        for frame in sampled_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)

            curr_left = None
            curr_right = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for lm_list, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
                    if label == "Left":
                        curr_left = coords
                    else:
                        curr_right = coords

            # Temporal smoothing: if hand was present recently and momentarily missed, decay previous coords
            if curr_left is not None:
                last_left_lms = curr_left
            elif last_left_lms is not None:
                curr_left = last_left_lms * 0.98

            if curr_right is not None:
                last_right_lms = curr_right
            elif last_right_lms is not None:
                curr_right = last_right_lms * 0.98

            # Extract 216-dim unified hand-pose feature vector
            feat_vec = extract_two_hand_features(curr_left, curr_right)
            sequence_features.append(feat_vec)

        if len(sequence_features) != SEQUENCE_LENGTH:
            return None

        return np.array(sequence_features, dtype=np.float32)  # shape (30, 216)


# Worker process singleton
_worker_extractor: Optional[VideoHandPoseExtractor] = None


def _init_worker():
    global _worker_extractor
    _worker_extractor = VideoHandPoseExtractor(min_confidence=0.3)


def _process_single_video(item: Tuple[str, str, int]) -> Optional[Tuple[np.ndarray, int, str]]:
    global _worker_extractor
    video_path_str, label, label_id = item
    video_path = Path(video_path_str)
    if _worker_extractor is None:
        _worker_extractor = VideoHandPoseExtractor(min_confidence=0.3)
    try:
        seq = _worker_extractor.extract_from_video(video_path)
        if seq is not None:
            return (seq, label_id, label)
        return None
    except Exception as exc:
        logger.error("Error processing %s: %s", video_path, exc)
        return None


def run_feature_extraction(max_workers: Optional[int] = None):
    """
    Run parallel feature extraction across all 642 ISL video clips.
    Saves:
      - data/processed/hand_pose/X_train.npy, y_train.npy
      - data/processed/hand_pose/X_val.npy, y_val.npy
      - data/processed/hand_pose/X_test.npy, y_test.npy
      - data/processed/hand_pose/static_X_train.npy, static_y_train.npy
      - data/processed/hand_pose/static_X_val.npy, static_y_val.npy
      - data/processed/hand_pose/static_X_test.npy, static_y_test.npy
      - models/hand_pose/labels.json
    """
    t0 = time.time()
    logger.info("Initializing Hand Pose Dataset...")
    dataset = HandPoseDataset(RAW_DATASET_DIR)

    PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save class label mapping
    labels_file = MODEL_DIR / "labels.json"
    dataset.save_labels(labels_file)

    train_samples, val_samples, test_samples = dataset.get_stratified_splits(
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )

    splits = {
        "train": [(str(p), lbl, lid) for p, lbl, lid in train_samples],
        "val": [(str(p), lbl, lid) for p, lbl, lid in val_samples],
        "test": [(str(p), lbl, lid) for p, lbl, lid in test_samples],
    }

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    logger.info("Starting parallel extraction with %d workers across %d unique videos...", max_workers, len(dataset.samples))

    for split_name, sample_list in splits.items():
        logger.info("Extracting features for split '%s' (%d videos)...", split_name, len(sample_list))
        X_seq_list = []
        y_list = []
        static_X_list = []
        static_y_list = []

        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
            futures = {executor.submit(_process_single_video, item): item for item in sample_list}
            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                completed += 1
                if completed % 50 == 0 or completed == len(sample_list):
                    logger.info("  [%s] Processed %d/%d videos...", split_name, completed, len(sample_list))
                if res is not None:
                    seq, lid, _ = res
                    X_seq_list.append(seq)
                    y_list.append(lid)

                    # Extract static keyframes (frames 10, 15, 20 from middle of gesture)
                    for kf in [10, 15, 20]:
                        if kf < len(seq):
                            static_X_list.append(seq[kf])
                            static_y_list.append(lid)

        X_arr = np.array(X_seq_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.int32)
        stat_X_arr = np.array(static_X_list, dtype=np.float32)
        stat_y_arr = np.array(static_y_list, dtype=np.int32)

        # For train set: apply simple landmark jitter / augmentation to enrich the sequence dataset
        if split_name == "train" and len(X_arr) > 0:
            logger.info("  Applying data augmentation (jitter + scale) to training sequences...")
            aug_X = []
            aug_y = []
            for seq, label in zip(X_arr, y_arr):
                # 1. Mild Gaussian noise
                noise = np.random.normal(0, 0.015, seq.shape).astype(np.float32)
                aug_X.append(seq + noise)
                aug_y.append(label)

                # 2. Slight scale variance
                scale_factor = np.random.uniform(0.92, 1.08)
                aug_X.append(seq * scale_factor)
                aug_y.append(label)

            X_arr = np.concatenate([X_arr, np.array(aug_X, dtype=np.float32)], axis=0)
            y_arr = np.concatenate([y_arr, np.array(aug_y, dtype=np.int32)], axis=0)

        # Save dynamic sequence data
        np.save(PROCESSED_DATASET_DIR / f"X_{split_name}.npy", X_arr)
        np.save(PROCESSED_DATASET_DIR / f"y_{split_name}.npy", y_arr)

        # Save static keyframe data
        np.save(PROCESSED_DATASET_DIR / f"static_X_{split_name}.npy", stat_X_arr)
        np.save(PROCESSED_DATASET_DIR / f"static_y_{split_name}.npy", stat_y_arr)

        logger.info(
            "Saved split '%s': Dynamic sequences = %s, Static samples = %s",
            split_name,
            X_arr.shape,
            stat_X_arr.shape,
        )

    elapsed = time.time() - t0
    logger.info("✅ Hand pose feature extraction complete in %.1f seconds!", elapsed)


if __name__ == "__main__":
    run_feature_extraction()
