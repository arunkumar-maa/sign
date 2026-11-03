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


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in [0, 1], normalized from degrees to 180 degrees."""
    first = a - b
    second = c - b
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine) / np.pi)


def _feature_for_hand(hand_landmarks: Any) -> np.ndarray:
    points = np.asarray([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected 21 x 3 landmarks, received {points.shape}.")
    wrist = points[0].copy()
    palm_scale = float(np.linalg.norm(points[9] - wrist))
    if palm_scale < 1e-4:
        palm_scale = 1e-4
    relative = (points - wrist) / palm_scale

    # Two thumb joint angles, then PIP and DIP angles for index/middle/ring/little.
    angle_triplets = [
        (1, 2, 3), (2, 3, 4),
        (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    ]
    angles = np.asarray([_joint_angle(points[a], points[b], points[c]) for a, b, c in angle_triplets], dtype=np.float32)
    tip_indices = (8, 12, 16, 20)
    thumb_tip = points[4]
    tip_distances = np.asarray(
        [np.linalg.norm(points[index] - thumb_tip) / palm_scale for index in tip_indices],
        dtype=np.float32,
    )
    # 63 absolute coordinates + 63 wrist-relative coordinates + 10 angles + 4 distances.
    return np.concatenate([points.reshape(-1), relative.reshape(-1), angles, tip_distances]).astype(np.float32)


def result_to_features(result: Any) -> tuple[np.ndarray, int]:
    """Return a fixed 282-float vector: two 140-value hand slots plus presence flags."""
    slots: dict[str, np.ndarray | None] = {"Left": None, "Right": None}
    hands = getattr(result, "hand_landmarks", []) or []
    handedness = getattr(result, "handedness", []) or []

    for index, landmarks in enumerate(hands[:MAX_HANDS]):
        side = None
        if index < len(handedness) and handedness[index]:
            try:
                side = handedness[index][0].category_name
            except (AttributeError, IndexError, TypeError):
                side = None
        if side not in slots or slots.get(side) is not None:
            center_x = float(np.mean([point.x for point in landmarks]))
            side = "Left" if center_x < 0.5 else "Right"
            if slots[side] is not None:
                side = "Right" if side == "Left" else "Left"
        slots[side] = _feature_for_hand(landmarks)

    values: list[np.ndarray] = []
    presence: list[float] = []
    for side in ("Left", "Right"):
        vector = slots[side]
        presence.append(1.0 if vector is not None else 0.0)
        values.append(vector if vector is not None else np.zeros(140, dtype=np.float32))

    features = np.concatenate(values + [np.asarray(presence, dtype=np.float32)])
    if features.shape != (FEATURE_DIM,):
        raise RuntimeError(f"Internal feature size error: expected {FEATURE_DIM}, got {features.shape}.")
    return features, int(sum(presence))


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
