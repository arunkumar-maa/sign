import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .config import FEATURE_DIM, HAND_TASK_PATH, HAND_TASK_URL, MAX_HANDS

# MediaPipe hand topology: 21 landmark indices.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_hand_task_model() -> Path:
    HAND_TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HAND_TASK_PATH.exists():
        print("Downloading MediaPipe Hand Landmarker model...")
        urllib.request.urlretrieve(HAND_TASK_URL, HAND_TASK_PATH)
    if HAND_TASK_PATH.stat().st_size < 100_000:
        raise RuntimeError("The downloaded MediaPipe task model looks incomplete. Delete it and retry.")
    return HAND_TASK_PATH


from backend.vision.feature_schema import (
    FEATURE_DIM,
    SINGLE_HAND_DIM,
    compute_joint_angle,
    extract_frame_features,
    extract_single_hand_features,
)


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in [0, 1], normalized from degrees to 180 degrees."""
    return compute_joint_angle(a, b, c)


def _feature_for_hand(hand_landmarks: Any) -> np.ndarray:
    points = np.asarray([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
    return extract_single_hand_features(points)


def result_to_features(result: Any) -> tuple[np.ndarray, int]:
    """Return a fixed 282-float vector via shared Feature Schema (v2.0.0)."""
    return extract_frame_features(result)


def draw_hand_overlay(rgb: np.ndarray, result: Any) -> np.ndarray:
    """Draw actual detected landmarks on an RGB frame and return RGB."""
    bgr = cv2.cvtColor(rgb[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
    hands = getattr(result, "hand_landmarks", []) or []
    height, width = bgr.shape[:2]
    for landmarks in hands:
        points = []
        for landmark in landmarks:
            x = max(0, min(width - 1, int(landmark.x * width)))
            y = max(0, min(height - 1, int(landmark.y * height)))
            points.append((x, y))
        if len(points) != 21:
            continue
        for start, end in HAND_CONNECTIONS:
            cv2.line(bgr, points[start], points[end], (255, 0, 0), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(bgr, point, 4, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(
        bgr, f"Hands: {len(hands)}", (12, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class HandPoseDetector:
    def __init__(self, running_mode: Any = None, num_hands: int = MAX_HANDS):
        model_path = ensure_hand_task_model()
        mode = running_mode or mp.tasks.vision.RunningMode.VIDEO
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect_video(self, rgb: np.ndarray, timestamp_ms: int):
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb[:, :, :3])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._detector.detect_for_video(image, int(timestamp_ms))

    def close(self) -> None:
        self._detector.close()
