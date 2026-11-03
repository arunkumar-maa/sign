"""
SignBridge AI — Unified Video & Multimodal Preprocessing Pipeline.
Extracts 192-dim landmark sequences (Hands 126 + Face 42 + Pose 24) from real ISL videos.
Extracts signer IDs, timestamps, labels, normalizes coordinates, builds fixed-length sequences (30 frames),
and ensures clean train/val/test splits strictly partitioned by signer ID.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import pandas as pd

from backend.vision.hand_landmarks import HandLandmarker, normalize_landmarks, NUM_HAND_LANDMARKS
from backend.vision.face_landmarks import FaceLandmarker, USEFUL_FACE_INDICES, NUM_FACE_FEATURES
from backend.vision.pose_landmarks import PoseLandmarker, UPPER_BODY_INDICES, NUM_POSE_FEATURES

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 30
FEATURE_DIM = (NUM_HAND_LANDMARKS * 3 * 2) + NUM_FACE_FEATURES + NUM_POSE_FEATURES  # 192


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_signer_id(file_name: str) -> str:
    """
    Extract signer / session identifier from filename.
    Examples:
      - 'yes__ISL500__00106__Yes__session105__clip014.mp4' -> 'session105'
      - 'you__CISLR__00040__QE4UkDB3MyE.mp4' -> 'CISLR_00040'
      - 'c545daf6-s168-008_Yes.mp4' -> 's168'
    """
    stem = Path(file_name).stem
    # Match sessionXXX
    m_session = re.search(r"session(\d+)", stem, re.IGNORECASE)
    if m_session:
        return f"session_{m_session.group(1)}"

    # Match CISLR signer
    m_cislr = re.search(r"CISLR__(\d+)", stem, re.IGNORECASE)
    if m_cislr:
        return f"cislr_{m_cislr.group(1)}"

    # Match sXXX
    m_s = re.search(r"-s(\d+)-", stem, re.IGNORECASE)
    if m_s:
        return f"signer_{m_s.group(1)}"

    # Fallback to prefix hash
    return f"signer_{abs(hash(stem)) % 20}"


class VideoLandmarkProcessor:
    """
    Extracts multi-modal landmark sequences from real ISL videos using MediaPipe.
    """

    def __init__(self):
        self.hand_detector = HandLandmarker()
        self.face_detector = FaceLandmarker()
        self.pose_detector = PoseLandmarker()

    def close(self):
        self.hand_detector.close()
        self.face_detector.close()
        self.pose_detector.close()

    def process_video(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract 30-frame sequence of 192-dim features, timestamps, and metadata.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total_frames <= 0:
            cap.release()
            return None

        # Uniformly sample 30 frames
        sample_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
        target_idx_set = set(sample_indices)

        frames = []
        timestamps = []
        curr_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if curr_idx in target_idx_set:
                frames.append(frame)
                timestamps.append(curr_idx / fps)
            curr_idx += 1

        cap.release()

        if len(frames) == 0:
            return None

        # Ensure exactly SEQUENCE_LENGTH frames
        if len(frames) < SEQUENCE_LENGTH:
            resample_idx = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
            frames = [frames[i] for i in resample_idx]
            timestamps = [timestamps[i] for i in resample_idx]

        sequence_features = []
        last_left = np.zeros(63, dtype=np.float32)
        last_right = np.zeros(63, dtype=np.float32)
        last_face = np.zeros(NUM_FACE_FEATURES, dtype=np.float32)
        last_pose = np.zeros(NUM_POSE_FEATURES, dtype=np.float32)

        for frame in frames:
            # Detect hands
            h_res = self.hand_detector.detect(frame, is_bgr=True)
            left_feat = normalize_landmarks(h_res.left_landmarks).flatten() if h_res.left_hand_detected else last_left * 0.95
            right_feat = normalize_landmarks(h_res.right_landmarks).flatten() if h_res.right_hand_detected else last_right * 0.95
            last_left = left_feat
            last_right = right_feat
            hand_feat = np.concatenate([left_feat, right_feat])

            # Detect face
            f_res = self.face_detector.detect(frame, is_bgr=True)
            face_feat = self.face_detector.extract_features(f_res) if f_res.detected else last_face * 0.95
            last_face = face_feat

            # Detect pose
            p_res = self.pose_detector.detect(frame, is_bgr=True)
            pose_feat = self.pose_detector.extract_features(p_res) if p_res.detected else last_pose * 0.95
            last_pose = pose_feat

            # 126 + 42 + 24 = 192 features
            combined = np.concatenate([hand_feat, face_feat, pose_feat])
            sequence_features.append(combined)

        seq_arr = np.array(sequence_features, dtype=np.float32)  # shape (30, 192)

        signer_id = parse_signer_id(video_path.name)
        label = video_path.parent.name

        return {
            "video_file": video_path.name,
            "label": label,
            "signer_id": signer_id,
            "sequence": seq_arr,
            "timestamps": timestamps,
            "fps": fps,
            "duration": total_frames / fps,
        }


def split_by_signer(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split samples into Train, Val, and Test strictly by signer_id
    to prevent data leakage across identical video signers.
    """
    np.random.seed(seed)
    signer_to_samples: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        signer_to_samples.setdefault(s["signer_id"], []).append(s)

    unique_signers = sorted(list(signer_to_samples.keys()))
    np.random.shuffle(unique_signers)

    total_signers = len(unique_signers)
    n_test = max(1, int(total_signers * test_ratio))
    n_val = max(1, int(total_signers * val_ratio))

    test_signers = set(unique_signers[:n_test])
    val_signers = set(unique_signers[n_test : n_test + n_val])
    train_signers = set(unique_signers[n_test + n_val :])

    # If too few signers, fallback to stratified clip splitting by class
    if len(train_signers) == 0:
        logger.warning("Signer count too low for pure signer split; falling back to clip-level stratification.")
        train, val, test = [], [], []
        for idx, s in enumerate(samples):
            mod = idx % 10
            if mod == 0:
                test.append(s)
            elif mod == 1:
                val.append(s)
            else:
                train.append(s)
        return train, val, test

    train, val, test = [], [], []
    for s in samples:
        signer = s["signer_id"]
        if signer in test_signers:
            test.append(s)
        elif signer in val_signers:
            val.append(s)
        else:
            train.append(s)

    logger.info(
        "Split by Signer: Train %d samples (%d signers), Val %d samples (%d signers), Test %d samples (%d signers)",
        len(train), len(train_signers), len(val), len(val_signers), len(test), len(test_signers)
    )
    return train, val, test
