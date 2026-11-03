"""
Feature extraction: combines hand, face, and pose landmarks into
a unified feature vector for model input.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.vision.hand_landmarks import HandLandmarkResult, extract_flat_features, NUM_HAND_LANDMARKS
from backend.vision.face_landmarks import FaceLandmarkResult, NUM_FACE_FEATURES
from backend.vision.pose_landmarks import PoseLandmarkResult, NUM_POSE_FEATURES

logger = logging.getLogger(__name__)

# Total feature vector length:
# 2 hands × 21 landmarks × 3 coords = 126
# face subset = NUM_FACE_FEATURES (42)
# pose upper body = NUM_POSE_FEATURES (24)
HAND_FEATURES = NUM_HAND_LANDMARKS * 3 * 2  # 126
TOTAL_FEATURES = HAND_FEATURES + NUM_FACE_FEATURES + NUM_POSE_FEATURES


@dataclass
class FrameFeatures:
    """Full feature set for a single frame."""
    hand_features: np.ndarray        # (126,)
    face_features: np.ndarray        # (42,)
    pose_features: np.ndarray        # (24,)
    combined: np.ndarray             # (192,)
    hand_detected: bool = False
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    face_detected: bool = False
    body_detected: bool = False


def extract_frame_features(
    hand_result: HandLandmarkResult,
    face_result: FaceLandmarkResult,
    pose_result: PoseLandmarkResult,
) -> FrameFeatures:
    """
    Extract and combine all landmark features for a single video frame.
    Missing detections produce zero-filled vectors.
    """
    hand_feat = extract_flat_features(hand_result.left_landmarks, hand_result.right_landmarks)

    face_feat = np.zeros(NUM_FACE_FEATURES, dtype=np.float32)
    if face_result.detected and face_result.landmarks is not None:
        face_feat = face_result.landmarks.flatten()

    pose_feat = np.zeros(NUM_POSE_FEATURES, dtype=np.float32)
    if pose_result.detected and pose_result.landmarks is not None:
        pose_feat = pose_result.landmarks.flatten()

    combined = np.concatenate([hand_feat, face_feat, pose_feat])

    return FrameFeatures(
        hand_features=hand_feat,
        face_features=face_feat,
        pose_features=pose_feat,
        combined=combined,
        hand_detected=hand_result.detected,
        left_hand_detected=hand_result.left_hand_detected,
        right_hand_detected=hand_result.right_hand_detected,
        face_detected=face_result.detected,
        body_detected=pose_result.detected,
    )
