"""
SignBridge AI — Hand Landmark Detection using MediaPipe Tasks HandLandmarker.
Integrates the proven detection parameters from hand_detection_test.py:
- Confidence thresholds: 0.3 detection, 0.3 presence, 0.3 tracking
- Green landmark circles (radius 5)
- Blue finger connection lines
- Detection status overlays ("21 LANDMARKS DETECTED", "NO HAND DETECTED", "Hands detected: N", FPS)
- 21 3D coordinates per hand (up to 2 hands) and handedness classification
- Persistent, reusable single detector instance without per-frame reinitialization
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

NUM_HAND_LANDMARKS = 21
FINGERTIP_INDICES: List[int] = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

# Standard 21 Hand Landmark Finger Connections
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                                # Palm base
]

# Model search paths
TASK_MODEL_PATHS = [
    Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task",
    Path(__file__).resolve().parent.parent.parent / "models" / "mediapipe" / "hand_landmarker.task",
]


@dataclass
class HandLandmarkResult:
    """Result from hand landmark detection."""
    detected: bool = False
    num_hands: int = 0
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    left_landmarks: Optional[np.ndarray] = None   # shape (21, 3) [x, y, z]
    right_landmarks: Optional[np.ndarray] = None  # shape (21, 3) [x, y, z]
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    inference_latency_ms: float = 0.0
    fps: float = 0.0
    all_hands: List[Tuple[str, np.ndarray, float]] = field(default_factory=list)
    raw_results: object = field(default=None, repr=False)


class HandLandmarker:
    """
    Persistent MediaPipe Tasks HandLandmarker wrapper for production camera streams.
    Reuses a single detector instance across frames.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.3,
        min_presence_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ):
        self.model_path = None
        if model_path and Path(model_path).exists():
            self.model_path = Path(model_path)
        else:
            for p in TASK_MODEL_PATHS:
                if p.exists():
                    self.model_path = p
                    break

        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._task_landmarker = None
        self._legacy_hands = None
        self._using_tasks_api = False

        self._frame_count = 0
        self._prev_time = time.perf_counter()
        self._current_fps = 0.0

        self._init_detector()

    def _init_detector(self):
        """Initialise Tasks API, falling back to legacy solutions if needed."""
        if self.model_path and self.model_path.exists():
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=str(self.model_path))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_presence_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self._task_landmarker = vision.HandLandmarker.create_from_options(options)
                self._using_tasks_api = True
                logger.info("MediaPipe Tasks HandLandmarker initialized from %s (conf=0.3)", self.model_path)
                return
            except Exception as e:
                logger.warning("Tasks HandLandmarker init failed (%s); attempting legacy fallback.", e)

        try:
            self._legacy_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._using_tasks_api = False
            logger.info("Legacy mp.solutions.hands initialized fallback.")
        except Exception as e:
            logger.error("Failed to initialize any MediaPipe HandLandmarker: %s", e)

    def detect(self, frame: np.ndarray, is_bgr: bool = True) -> HandLandmarkResult:
        """
        Detect hands and extract 21 3D landmarks per hand.

        Args:
            frame: Image array (BGR for OpenCV or RGB for Gradio).
            is_bgr: True if BGR, False if RGB.

        Returns:
            HandLandmarkResult with detected landmarks, handedness, and FPS.
        """
        if frame is None or frame.size == 0:
            return HandLandmarkResult()

        t0 = time.perf_counter()

        # Update FPS counter
        self._frame_count += 1
        dt = t0 - self._prev_time
        if dt >= 0.5:
            self._current_fps = self._frame_count / dt
            self._frame_count = 0
            self._prev_time = t0

        # Ensure frame is in RGB for MediaPipe processing
        if is_bgr:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)

        result = HandLandmarkResult()
        result.fps = self._current_fps

        if self._using_tasks_api and self._task_landmarker is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_res = self._task_landmarker.detect(mp_image)
                result.raw_results = detection_res

                if detection_res and detection_res.hand_landmarks:
                    num_hands = len(detection_res.hand_landmarks)
                    result.detected = num_hands > 0
                    result.num_hands = num_hands

                    detected_hands = []
                    for i in range(num_hands):
                        lms = detection_res.hand_landmarks[i]
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                        score = 0.90
                        label = "Right" if i == 0 else "Left"
                        if detection_res.handedness and i < len(detection_res.handedness):
                            top_category = detection_res.handedness[i][0]
                            label = top_category.category_name or top_category.display_name or label
                            score = float(top_category.score)
                        detected_hands.append((label, coords, score))

                    # Prevent duplicate handedness collision dropping one hand
                    slots = {"left": None, "right": None}
                    for label, coords, score in detected_hands:
                        side = label.lower()
                        if side not in slots or slots[side] is not None:
                            empty_side = "left" if slots["left"] is None else "right"
                            slots[empty_side] = (coords, score, label)
                        else:
                            slots[side] = (coords, score, label)

                    if slots["left"] is not None:
                        result.left_hand_detected = True
                        result.left_landmarks = slots["left"][0]
                        result.left_confidence = slots["left"][1]
                    if slots["right"] is not None:
                        result.right_hand_detected = True
                        result.right_landmarks = slots["right"][0]
                        result.right_confidence = slots["right"][1]

                    for label, coords, score in detected_hands:
                        result.all_hands.append((label, coords, score))
            except Exception as exc:
                logger.error("Tasks HandLandmarker error: %s", exc)

        elif self._legacy_hands is not None:
            try:
                results = self._legacy_hands.process(rgb_frame)
                result.raw_results = results

                if results.multi_hand_landmarks:
                    num_hands = len(results.multi_hand_landmarks)
                    result.detected = num_hands > 0
                    result.num_hands = num_hands

                    detected_hands = []
                    for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        score = float(handedness.classification[0].score)
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
                        detected_hands.append((label, coords, score))

                    slots = {"left": None, "right": None}
                    for label, coords, score in detected_hands:
                        side = label.lower()
                        if side not in slots or slots[side] is not None:
                            empty_side = "left" if slots["left"] is None else "right"
                            slots[empty_side] = (coords, score, label)
                        else:
                            slots[side] = (coords, score, label)

                    if slots["left"] is not None:
                        result.left_hand_detected = True
                        result.left_landmarks = slots["left"][0]
                        result.left_confidence = slots["left"][1]
                    if slots["right"] is not None:
                        result.right_hand_detected = True
                        result.right_landmarks = slots["right"][0]
                        result.right_confidence = slots["right"][1]

                    for label, coords, score in detected_hands:
                        result.all_hands.append((label, coords, score))
            except Exception as exc:
                logger.error("Legacy HandLandmarker error: %s", exc)

        result.inference_latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: HandLandmarkResult,
        is_bgr: bool = True,
    ) -> np.ndarray:
        """
        Draw MediaPipe green landmark circles, blue finger connections, and status text.

        Args:
            frame: Frame array.
            result: HandLandmarkResult.
            is_bgr: True if frame is in BGR, False if RGB.

        Returns:
            Annotated frame array.
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        # Color definitions depending on color space:
        # Green circles: (0, 255, 0) in both RGB and BGR
        # Blue lines: (255, 0, 0) in BGR, (0, 0, 255) in RGB
        # Red text: (0, 0, 255) in BGR, (255, 0, 0) in RGB
        # Yellow text: (0, 255, 255) in BGR, (255, 255, 0) in RGB
        green_color = (0, 255, 0)
        blue_line_color = (255, 0, 0) if is_bgr else (0, 0, 255)
        red_text_color = (0, 0, 255) if is_bgr else (255, 0, 0)
        yellow_text_color = (0, 255, 255) if is_bgr else (255, 255, 0)

        if not result.detected or result.num_hands == 0:
            cv2.putText(
                annotated,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                red_text_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "Show your open palm clearly",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                red_text_color,
                2,
                cv2.LINE_AA,
            )
        else:
            # Collect hands to draw (prefer all_hands from actual detection, or fall back to left/right slots)
            hands_to_draw = []
            if result.all_hands:
                hands_to_draw = [(label, coords) for label, coords, _ in result.all_hands]
            else:
                if result.right_hand_detected and result.right_landmarks is not None:
                    hands_to_draw.append(("Right", result.right_landmarks))
                if result.left_hand_detected and result.left_landmarks is not None:
                    hands_to_draw.append(("Left", result.left_landmarks))

            for hand_index, (hand_label, landmarks) in enumerate(hands_to_draw):
                points = []
                for lm in landmarks:
                    x = int(np.clip(lm[0] * width, 0, width - 1))
                    y = int(np.clip(lm[1] * height, 0, height - 1))
                    points.append((x, y))

                    # Green circle at each landmark (radius 5)
                    cv2.circle(annotated, (x, y), 5, green_color, -1, cv2.LINE_AA)

                # Blue lines connecting finger joints
                for start, end in HAND_CONNECTIONS:
                    if start < len(points) and end < len(points):
                        cv2.line(annotated, points[start], points[end], blue_line_color, 2, cv2.LINE_AA)

                # Programmatic verification of landmark count
                num_lm = len(points)
                if num_lm == 21:
                    hand_text = f"{hand_label}: 21/21 hand landmarks detected"
                else:
                    hand_text = f"{hand_label}: {num_lm}/21 hand landmarks detected"

                cv2.putText(
                    annotated,
                    hand_text,
                    (20, 40 + hand_index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    green_color,
                    2,
                    cv2.LINE_AA,
                )

        # Bottom HUD: Hand count and FPS
        hud_text = f"Hands detected: {result.num_hands} | FPS: {result.fps:.1f} | Latency: {result.inference_latency_ms:.0f}ms"
        cv2.putText(
            annotated,
            hud_text,
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            yellow_text_color,
            2,
            cv2.LINE_AA,
        )

        return annotated

    def close(self):
        """Release Landmarker resources."""
        if self._task_landmarker is not None:
            try:
                self._task_landmarker.close()
            except Exception:
                pass
            self._task_landmarker = None

        if self._legacy_hands is not None:
            try:
                self._legacy_hands.close()
            except Exception:
                pass
            self._legacy_hands = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def normalize_landmarks(landmarks: Optional[np.ndarray]) -> np.ndarray:
    """Normalize hand landmarks relative to wrist and middle MCP distance."""
    if landmarks is None or len(landmarks) == 0:
        return np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)

    normalized = landmarks.copy()
    wrist = normalized[0].copy()
    normalized -= wrist
    scale = np.linalg.norm(normalized[9])
    if scale > 1e-6:
        normalized /= scale
    return normalized


def extract_flat_features(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
) -> np.ndarray:
    """Return flat feature vector of shape (126,) for both hands."""
    left_feat = normalize_landmarks(left).flatten() if left is not None else np.zeros(63, dtype=np.float32)
    right_feat = normalize_landmarks(right).flatten() if right is not None else np.zeros(63, dtype=np.float32)
    return np.concatenate([left_feat, right_feat])
