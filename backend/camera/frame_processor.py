"""
Frame processor: runs MediaPipe detection on each camera frame
and returns annotated frame + extracted features.
Properly handles both OpenCV BGR and Gradio RGB frames without color corruption.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from backend.vision.hand_landmarks import HandLandmarker, HandLandmarkResult
from backend.vision.face_landmarks import FaceLandmarker, FaceLandmarkResult
from backend.vision.pose_landmarks import PoseLandmarker, PoseLandmarkResult
from backend.vision.feature_extraction import FrameFeatures, extract_frame_features

logger = logging.getLogger(__name__)


@dataclass
class TrackingStatus:
    """Human-readable tracking quality information."""
    hand_detected: bool = False
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    face_detected: bool = False
    body_detected: bool = False
    poor_lighting: bool = False
    hands_outside_frame: bool = False
    quality: str = "No hands detected"

    def to_dict(self) -> dict:
        return {
            "hand_detected": self.hand_detected,
            "left_hand_detected": self.left_hand_detected,
            "right_hand_detected": self.right_hand_detected,
            "face_detected": self.face_detected,
            "body_detected": self.body_detected,
            "poor_lighting": self.poor_lighting,
            "hands_outside_frame": self.hands_outside_frame,
            "quality": self.quality,
        }


@dataclass
class ProcessedFrame:
    """Result from processing a single camera frame."""
    annotated_frame: np.ndarray
    features: FrameFeatures
    tracking: TrackingStatus
    processing_time_ms: float


class FrameProcessor:
    """
    Processes individual frames through MediaPipe landmark detectors
    and returns annotated frames with multi-modal features.
    """

    def __init__(self):
        self._hand_detector = HandLandmarker()
        self._face_detector = FaceLandmarker()
        self._pose_detector = PoseLandmarker()
        logger.info("FrameProcessor initialised with MediaPipe Tasks")

    def process(self, frame: np.ndarray, is_bgr: bool = True) -> ProcessedFrame:
        """
        Run all detectors on the frame.

        Args:
            frame: uint8 numpy image array.
            is_bgr: True if frame is OpenCV BGR; False if frame is RGB (e.g. from Gradio).

        Returns:
            ProcessedFrame with annotated image, features, and tracking status.
        """
        t0 = time.perf_counter()

        if frame is None or frame.size == 0:
            empty_feat = FrameFeatures(
                hand_features=np.zeros(126, dtype=np.float32),
                face_features=np.zeros(42, dtype=np.float32),
                pose_features=np.zeros(24, dtype=np.float32),
                combined=np.zeros(192, dtype=np.float32),
            )
            return ProcessedFrame(
                annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                features=empty_feat,
                tracking=TrackingStatus(),
                processing_time_ms=0.0,
            )

        # 1. Hands (primary)
        hand_result = self._hand_detector.detect(frame, is_bgr=is_bgr)

        # 2. Face (isolated)
        try:
            face_result = self._face_detector.detect(frame, is_bgr=is_bgr)
        except Exception as exc:
            logger.warning("Face detector error suppressed: %s", exc)
            face_result = FaceLandmarkResult()

        # 3. Pose (isolated)
        try:
            pose_result = self._pose_detector.detect(frame, is_bgr=is_bgr)
        except Exception as exc:
            logger.warning("Pose detector error suppressed: %s", exc)
            pose_result = PoseLandmarkResult()

        # 4. Feature extraction
        features = extract_frame_features(hand_result, face_result, pose_result)

        # 5. Annotation
        annotated = self._hand_detector.draw_landmarks(frame, hand_result, is_bgr=is_bgr)
        annotated = self._draw_status_overlay(annotated, features, is_bgr=is_bgr)

        # 6. Status assessment
        tracking = self._assess_tracking(frame, features, is_bgr=is_bgr)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ProcessedFrame(
            annotated_frame=annotated,
            features=features,
            tracking=tracking,
            processing_time_ms=elapsed_ms,
        )

    def _assess_tracking(self, frame: np.ndarray, features: FrameFeatures, is_bgr: bool = True) -> TrackingStatus:
        """Assess tracking quality and populate status."""
        status = TrackingStatus(
            hand_detected=features.hand_detected,
            left_hand_detected=features.left_hand_detected,
            right_hand_detected=features.right_hand_detected,
            face_detected=features.face_detected,
            body_detected=features.body_detected,
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY)
        mean_brightness = float(gray.mean())
        if mean_brightness < 50:
            status.poor_lighting = True

        if features.hand_detected:
            if features.left_hand_detected and features.right_hand_detected:
                status.quality = "Both hands tracked"
            elif features.right_hand_detected:
                status.quality = "Right hand tracked"
            elif features.left_hand_detected:
                status.quality = "Left hand tracked"
        elif status.poor_lighting:
            status.quality = "⚠ Poor lighting — improve illumination"
        else:
            status.quality = "No hands detected — show hands in frame"

        return status

    def _draw_status_overlay(self, frame: np.ndarray, features: FrameFeatures, is_bgr: bool = True) -> np.ndarray:
        """Draw tracking status indicator badges on the bottom-left of the frame."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        indicators = [
            ("L", features.left_hand_detected, (35, h - 35)),
            ("R", features.right_hand_detected, (75, h - 35)),
            ("F", features.face_detected, (115, h - 35)),
            ("B", features.body_detected, (155, h - 35)),
        ]

        # In BGR: Active is (0, 200, 0), Inactive is (0, 0, 180)
        # In RGB: Active is (0, 200, 0), Inactive is (180, 0, 0)
        for label, active, pos in indicators:
            if is_bgr:
                color = (0, 220, 0) if active else (0, 0, 180)
            else:
                color = (0, 220, 0) if active else (180, 0, 0)

            cv2.circle(overlay, pos, 14, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, pos, 15, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                label,
                (pos[0] - 5, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return overlay

    def close(self):
        self._hand_detector.close()
        self._face_detector.close()
        self._pose_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
