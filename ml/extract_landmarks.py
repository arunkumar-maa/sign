"""
SignBridge AI — Phase 5: Video Landmark Extraction Pipeline.
Iterates over all verified ISL video clips, uniformly extracts frames,
and computes 21 hand landmarks per hand using MediaPipe.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from backend.vision.hand_landmarks import HandLandmarker
from backend.vision.landmark_preprocessing import (
    normalize_hand_landmarks,
    extract_normalized_hand_features,
    construct_multimodal_vector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "isl_isolated_40words"


def extract_landmarks_from_video(
    video_path: Path,
    landmarker: HandLandmarker,
    num_frames: int = 30,
) -> Optional[np.ndarray]:
    """
    Extract normalized multimodal vectors from video clip sampled uniformly over num_frames.
    Returns: np.ndarray of shape (num_frames, 192) or None.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    sample_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    target_set = set(sample_indices)

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_set:
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        return None

    # Resample if needed
    if len(frames) < num_frames:
        resample_idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in resample_idx]

    seq_features = []
    for frame in frames:
        # Detect hands using BGR frame
        h_res = landmarker.detect(frame, is_bgr=True)
        vec = construct_multimodal_vector(
            left_landmarks=h_res.left_landmarks if h_res.left_hand_detected else None,
            right_landmarks=h_res.right_landmarks if h_res.right_hand_detected else None,
        )
        seq_features.append(vec)

    return np.array(seq_features, dtype=np.float32)  # (num_frames, 192)

