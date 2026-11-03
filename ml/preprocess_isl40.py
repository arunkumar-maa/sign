"""
SignBridge AI - High Performance ISL 40 Words Dataset Preprocessing Pipeline
Processes 642 real ISL video clips using multi-process parallel MediaPipe extraction.
Extracts 192-dim landmark sequences (Hands 126 + Face 42 + Pose 24) across 30 frames.
Splits into Train (70%), Validation (15%), Test (15%) with data augmentation on training set.
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
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision.face_landmarks import USEFUL_FACE_INDICES, NUM_FACE_FEATURES
from backend.vision.pose_landmarks import UPPER_BODY_INDICES, NUM_POSE_FEATURES
from backend.vision.hand_landmarks import NUM_HAND_LANDMARKS, normalize_landmarks

SEQUENCE_LENGTH = 30
FEATURE_DIM = (NUM_HAND_LANDMARKS * 3 * 2) + NUM_FACE_FEATURES + NUM_POSE_FEATURES  # 192


class VideoFeatureExtractor:
    """Extracts MediaPipe landmarks from video clips into (30, 192) sequences."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )

    def close(self):
        try:
            self.mp_hands.close()
            self.mp_pose.close()
            self.mp_face.close()
        except Exception:
            pass

    def process_video(self, video_path: Path) -> Optional[np.ndarray]:
        """Reads video, uniformly samples 30 frames, and extracts combined 192-dim landmarks."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        if total_frames >= SEQUENCE_LENGTH:
            sample_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
        else:
            sample_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)

        frames = []
        current_idx = 0
        target_idx_set = set(sample_indices)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in target_idx_set:
                frames.append((current_idx, frame))
            current_idx += 1

        cap.release()

        if not frames:
            return None

        frames_dict = {idx: f for idx, f in frames}
        sampled_frames = [frames_dict[idx] for idx in sample_indices if idx in frames_dict]

        if len(sampled_frames) < SEQUENCE_LENGTH:
            if len(sampled_frames) == 0:
                return None
            resample_idx = np.linspace(0, len(sampled_frames) - 1, SEQUENCE_LENGTH, dtype=int)
            sampled_frames = [sampled_frames[i] for i in resample_idx]

        sequence_features = []
        last_left = np.zeros(63, dtype=np.float32)
        last_right = np.zeros(63, dtype=np.float32)
        last_pose = np.zeros(NUM_POSE_FEATURES, dtype=np.float32)
        last_face = np.zeros(NUM_FACE_FEATURES, dtype=np.float32)

        for frame in sampled_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Hands
            hands_res = self.mp_hands.process(rgb)
            left_feat = None
            right_feat = None

            if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                for lm_list, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                    label = handedness.classification[0].label
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
                    norm_coords = normalize_landmarks(coords).flatten()
                    if label == "Left":
                        left_feat = norm_coords
                    else:
                        right_feat = norm_coords

            if left_feat is not None:
                last_left = left_feat
            else:
                left_feat = last_left * 0.95

            if right_feat is not None:
                last_right = right_feat
            else:
                right_feat = last_right * 0.95

            # 2. Pose
            pose_res = self.mp_pose.process(rgb)
            pose_feat = np.zeros(NUM_POSE_FEATURES, dtype=np.float32)
            if pose_res.pose_landmarks:
                landmarks = pose_res.pose_landmarks.landmark
                p_coords = []
                for idx in UPPER_BODY_INDICES:
                    lm = landmarks[idx]
                    p_coords.extend([lm.x, lm.y, lm.z])
                pose_feat = np.array(p_coords, dtype=np.float32)
                last_pose = pose_feat
            else:
                pose_feat = last_pose

            # 3. Face
            face_res = self.mp_face.process(rgb)
            face_feat = np.zeros(NUM_FACE_FEATURES, dtype=np.float32)
            if face_res.multi_face_landmarks:
                lm_list = face_res.multi_face_landmarks[0].landmark
                f_coords = []
                for idx in USEFUL_FACE_INDICES:
                    lm = lm_list[idx]
                    f_coords.extend([lm.x, lm.y, lm.z])
                face_feat = np.array(f_coords, dtype=np.float32)
                last_face = face_feat
            else:
                face_feat = last_face

            frame_vec = np.concatenate([left_feat, right_feat, face_feat, pose_feat])
            sequence_features.append(frame_vec)

        seq = np.array(sequence_features, dtype=np.float32)
        if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            return None
        return seq


_worker_extractor: Optional[VideoFeatureExtractor] = None


def _init_worker():
    global _worker_extractor
    _worker_extractor = VideoFeatureExtractor()


def _process_single_video(args: Tuple[str, str, int]) -> Tuple[str, str, int, Optional[np.ndarray]]:
    vpath_str, cls_name, cls_id = args
    global _worker_extractor
    if _worker_extractor is None:
        _worker_extractor = VideoFeatureExtractor()
    seq = _worker_extractor.process_video(Path(vpath_str))
    return (vpath_str, cls_name, cls_id, seq)


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """Applies realistic jitter and scaling data augmentation to a sequence."""
    aug = seq.copy()
    mask = (aug != 0).astype(np.float32)
    noise = np.random.normal(0, 0.015, size=aug.shape) * mask
    aug = aug + noise

    scale = np.random.uniform(0.95, 1.05)
    aug = aug * scale

    if np.random.rand() > 0.5:
        shift = np.random.choice([-1, 1])
        aug = np.roll(aug, shift, axis=0)
        if shift > 0:
            aug[0] = aug[1]
        else:
            aug[-1] = aug[-2]

    return aug.astype(np.float32)


def main():
    start_time = time.time()
    raw_dir = PROJECT_ROOT / "data" / "raw" / "isl_isolated_40words"
    processed_dir = PROJECT_ROOT / "data" / "processed" / "isl40"
    processed_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = PROJECT_ROOT / "models" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = raw_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.error("Metadata not found at %s", metadata_path)
        return

    df = pd.read_csv(metadata_path)
    logger.info("Loaded metadata: %d rows", len(df))

    classes = sorted(df["normalized_word"].unique().tolist())
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}
    id_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

    labels_data = {
        "version": "2.0.0",
        "dataset": "vidit031/isl-isolated-40words",
        "num_classes": len(classes),
        "labels": classes,
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    isl40_labels_path = labels_dir / "isl40_labels.json"
    with open(isl40_labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2)
    logger.info("Saved 40 classes label mapping to %s", isl40_labels_path)

    tasks = []
    for _, row in df.iterrows():
        vpath = raw_dir / row["video_path"]
        cls_name = row["normalized_word"]
        cls_id = class_to_id[cls_name]
        tasks.append((str(vpath), cls_name, cls_id))

    num_workers = min(8, os.cpu_count() or 4)
    logger.info("Extracting landmark sequences for %d videos using %d workers...", len(tasks), num_workers)

    extracted_data = []
    done_count = 0

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_process_single_video, t): t for t in tasks}
        for future in as_completed(futures):
            vpath_str, cls_name, cls_id, seq = future.result()
            done_count += 1
            if seq is not None:
                extracted_data.append({
                    "video_path": vpath_str,
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "sequence": seq,
                })

            if done_count % 50 == 0 or done_count == len(tasks):
                elapsed = time.time() - start_time
                rate = done_count / max(elapsed, 0.1)
                logger.info("Processed %d/%d clips (valid: %d, rate: %.1f clips/sec)",
                            done_count, len(tasks), len(extracted_data), rate)

    logger.info("Feature extraction complete in %.1fs: %d valid sequences out of %d.",
                time.time() - start_time, len(extracted_data), len(df))

    from collections import defaultdict
    class_samples = defaultdict(list)
    for item in extracted_data:
        class_samples[item["class_id"]].append(item)

    train_items = []
    val_items = []
    test_items = []

    np.random.seed(42)

    for cid, items in class_samples.items():
        n = len(items)
        perm = np.random.permutation(n)
        shuffled = [items[i] for i in perm]

        if n == 1:
            train_items.append(shuffled[0])
        elif n == 2:
            train_items.append(shuffled[0])
            test_items.append(shuffled[1])
        elif n == 3:
            train_items.append(shuffled[0])
            val_items.append(shuffled[1])
            test_items.append(shuffled[2])
        else:
            n_test = max(1, int(round(n * 0.15)))
            n_val = max(1, int(round(n * 0.15)))
            n_train = n - n_test - n_val

            train_items.extend(shuffled[:n_train])
            val_items.extend(shuffled[n_train:n_train + n_val])
            test_items.extend(shuffled[n_train + n_val:])

    logger.info("Raw splits: Train=%d, Val=%d, Test=%d (Total=%d)",
                len(train_items), len(val_items), len(test_items),
                len(train_items) + len(val_items) + len(test_items))

    augmented_train_X = []
    augmented_train_y = []

    train_class_counts = defaultdict(int)
    for it in train_items:
        train_class_counts[it["class_id"]] += 1
        augmented_train_X.append(it["sequence"])
        augmented_train_y.append(it["class_id"])

    target_per_class = 25
    for cid, items in class_samples.items():
        train_class_items = [it for it in train_items if it["class_id"] == cid]
        if not train_class_items:
            train_class_items = items

        current_count = train_class_counts[cid]
        needed = max(0, target_per_class - current_count)

        for i in range(needed):
            base_item = train_class_items[i % len(train_class_items)]
            aug_seq = augment_sequence(base_item["sequence"])
            augmented_train_X.append(aug_seq)
            augmented_train_y.append(cid)

    X_train = np.array(augmented_train_X, dtype=np.float32)
    y_train = np.array(augmented_train_y, dtype=np.int32)

    X_val = np.array([it["sequence"] for it in val_items], dtype=np.float32)
    y_val = np.array([it["class_id"] for it in val_items], dtype=np.int32)

    X_test = np.array([it["sequence"] for it in test_items], dtype=np.float32)
    y_test = np.array([it["class_id"] for it in test_items], dtype=np.int32)

    train_perm = np.random.permutation(len(X_train))
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    logger.info("Final dataset arrays:")
    logger.info("  X_train: %s, y_train: %s", X_train.shape, y_train.shape)
    logger.info("  X_val:   %s, y_val:   %s", X_val.shape, y_val.shape)
    logger.info("  X_test:  %s, y_test:  %s", X_test.shape, y_test.shape)

    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)

    summary = {
        "dataset_name": "vidit031/isl-isolated-40words",
        "num_classes": len(classes),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,
        "raw_clips_total": len(df),
        "valid_clips_extracted": len(extracted_data),
        "train_samples_raw": len(train_items),
        "train_samples_augmented": len(X_train),
        "val_samples_pure": len(X_val),
        "test_samples_pure": len(X_test),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "classes": classes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = processed_dir / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Preprocessing complete! Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
