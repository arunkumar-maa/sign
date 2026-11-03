"""
SignBridge AI — Unified Landmark Normalization & Preprocessing Module.
Ensures identical mathematical representation of landmark sequences across
both offline model training and live webcam inference.

Features:
- Relative translation to wrist origin (landmark 0)
- Scale normalization via palm diagonal / wrist-to-middle MCP distance
- Consistent left/right ordering (Left hand: indices 0..62, Right hand: 63..125)
- Support for one-hand and two-hand signs
- Hand-presence flags and safe zero-padding for missing hands
- Resampling of frame sequences into fixed-length arrays (e.g. 30 frames)
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

NUM_LANDMARKS_PER_HAND = 21
HAND_COORDS_DIM = NUM_LANDMARKS_PER_HAND * 3  # 63
TWO_HANDS_DIM = HAND_COORDS_DIM * 2          # 126
FACE_DIM = 42                                # 14 points * 3
POSE_DIM = 24                                # 8 points * 3
TOTAL_MULTIMODAL_DIM = TWO_HANDS_DIM + FACE_DIM + POSE_DIM  # 192


def normalize_hand_landmarks(landmarks: Optional[np.ndarray]) -> np.ndarray:
    """
    Normalize 21 hand landmarks:
    1. Translate coordinates so the wrist (landmark 0) is at origin (0, 0, 0).
    2. Scale invariant: divide by distance between wrist (0) and middle finger MCP (9).
    
    Args:
        landmarks: np.ndarray of shape (21, 3) or None.
        
    Returns:
        Normalized np.ndarray of shape (21, 3).
    """
    if landmarks is None or len(landmarks) == 0:
        return np.zeros((NUM_LANDMARKS_PER_HAND, 3), dtype=np.float32)

    lms = np.array(landmarks, dtype=np.float32).copy()
    if lms.shape != (NUM_LANDMARKS_PER_HAND, 3):
        lms = lms.reshape(NUM_LANDMARKS_PER_HAND, 3)

    # 1. Translate wrist to (0, 0, 0)
    wrist = lms[0].copy()
    lms -= wrist

    # 2. Scale by middle finger metacarpal base (joint 9)
    scale = np.linalg.norm(lms[9])
    if scale > 1e-5:
        lms /= scale
    else:
        # Fallback to bounding box diagonal
        span = np.ptp(lms, axis=0)
        diag = np.linalg.norm(span)
        if diag > 1e-5:
            lms /= diag

    return lms


def extract_normalized_hand_features(
    left_landmarks: Optional[np.ndarray],
    right_landmarks: Optional[np.ndarray],
) -> Tuple[np.ndarray, bool, bool]:
    """
    Extract normalized coordinates for both hands into a continuous 126-float vector.
    
    Returns:
        features: np.ndarray of shape (126,)
        left_present: bool
        right_present: bool
    """
    left_present = left_landmarks is not None and len(left_landmarks) == NUM_LANDMARKS_PER_HAND
    right_present = right_landmarks is not None and len(right_landmarks) == NUM_LANDMARKS_PER_HAND

    norm_left = normalize_hand_landmarks(left_landmarks).flatten() if left_present else np.zeros(HAND_COORDS_DIM, dtype=np.float32)
    norm_right = normalize_hand_landmarks(right_landmarks).flatten() if right_present else np.zeros(HAND_COORDS_DIM, dtype=np.float32)

    features = np.concatenate([norm_left, norm_right])
    return features, left_present, right_present


def construct_multimodal_vector(
    left_landmarks: Optional[np.ndarray],
    right_landmarks: Optional[np.ndarray],
    face_landmarks: Optional[np.ndarray] = None,
    pose_landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct standardized 192-dimensional multi-modal vector:
    - 0..62: Left hand (63)
    - 63..125: Right hand (63)
    - 126..167: Face subset (42)
    - 168..191: Upper body pose (24)
    """
    hand_feat, _, _ = extract_normalized_hand_features(left_landmarks, right_landmarks)

    if face_landmarks is not None and len(face_landmarks) > 0:
        face_feat = face_landmarks.flatten()[:FACE_DIM]
        if len(face_feat) < FACE_DIM:
            pad = np.zeros(FACE_DIM - len(face_feat), dtype=np.float32)
            face_feat = np.concatenate([face_feat, pad])
    else:
        face_feat = np.zeros(FACE_DIM, dtype=np.float32)

    if pose_landmarks is not None and len(pose_landmarks) > 0:
        pose_feat = pose_landmarks.flatten()[:POSE_DIM]
        if len(pose_feat) < POSE_DIM:
            pad = np.zeros(POSE_DIM - len(pose_feat), dtype=np.float32)
            pose_feat = np.concatenate([pose_feat, pad])
    else:
        pose_feat = np.zeros(POSE_DIM, dtype=np.float32)

    return np.concatenate([hand_feat, face_feat, pose_feat]).astype(np.float32)


def resample_sequence(
    sequence: List[np.ndarray],
    target_length: int = 30,
) -> np.ndarray:
    """
    Resample a variable-length list of frame feature vectors into
    a fixed-length numpy array of shape (target_length, feature_dim).
    """
    if not sequence:
        return np.zeros((target_length, TOTAL_MULTIMODAL_DIM), dtype=np.float32)

    raw_arr = np.array(sequence, dtype=np.float32)
    n_frames = len(raw_arr)

    if n_frames == target_length:
        return raw_arr

    # Uniform linear sampling
    indices = np.linspace(0, n_frames - 1, target_length)
    resampled = np.zeros((target_length, raw_arr.shape[1]), dtype=np.float32)

    for i, idx in enumerate(indices):
        low = int(np.floor(idx))
        high = min(int(np.ceil(idx)), n_frames - 1)
        weight = idx - low
        if low == high:
            resampled[i] = raw_arr[low]
        else:
            resampled[i] = (1.0 - weight) * raw_arr[low] + weight * raw_arr[high]

    return resampled

