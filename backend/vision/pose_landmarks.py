"""
Upper-body pose landmark detection using the official MediaPipe Tasks Pose Landmarker API.
Extracts shoulder, elbow, wrist, and torso landmarks to provide spatial frame-of-reference
for signs performed relative to the body (e.g., chest level, chin level, overhead).
Operates completely independently so pose errors never crash hand tracking.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# 8 Upper-body landmark indices:
# 11, 12: shoulders
# 13, 14: elbows
# 15, 16: wrists
# 23, 24: hips
UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
NUM_POSE_FEATURES = len(UPPER_BODY_INDICES) * 3  # 24 features

DEFAULT_POSE_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "mediapipe" / "pose_landmarker.task"


@dataclass
class PoseLandmarkResult:
    """Result from upper-body pose landmark detection."""
    detected: bool = False
    landmarks: Optional[np.ndarray] = None  # shape (8, 3) [x, y, z]
    confidence: float = 0.0
    inference_latency_ms: float = 0.0
    raw_results: object = field(default=None, repr=False)


class PoseLandmarker:
    """
    Wraps MediaPipe Tasks PoseLandmarker for upper-body posture extraction.
    Independent and fully isolated from hand and face detection.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_detection_confidence: float = 0.5,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_POSE_MODEL_PATH
        self.min_detection_confidence = min_detection_confidence

        self._task_landmarker = None
        self._legacy_pose = None
        self._using_tasks_api = False

        self._init_detector()

    def _init_detector(self):
        """Initialise Tasks API with fallback to legacy solutions."""
        if self.model_path.exists():
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=str(self.model_path))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=self.min_detection_confidence,
                    min_pose_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_detection_confidence,
                )
                self._task_landmarker = vision.PoseLandmarker.create_from_options(options)
                self._using_tasks_api = True
                logger.info("PoseLandmarker Tasks API initialized from %s", self.model_path)
                return
            except Exception as e:
                logger.warning("Tasks PoseLandmarker init failed: %s. Falling back to legacy.", e)

        try:
            self._legacy_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_detection_confidence,
            )
            self._using_tasks_api = False
            logger.info("Legacy mp.solutions.pose initialized.")
        except Exception as e:
            logger.error("Failed to initialize PoseLandmarker: %s", e)

    def detect(self, frame: np.ndarray, is_bgr: bool = True) -> PoseLandmarkResult:
        """
        Run pose detection on an image frame.
        Safe against any exception so pipeline never crashes.
        """
        if frame is None or frame.size == 0:
            return PoseLandmarkResult()

        t0 = time.perf_counter()
        result = PoseLandmarkResult()

        try:
            if is_bgr:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)

            if self._using_tasks_api and self._task_landmarker is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection = self._task_landmarker.detect(mp_image)
                result.raw_results = detection

                if detection and detection.pose_landmarks and len(detection.pose_landmarks) > 0:
                    pose = detection.pose_landmarks[0]
                    coords = np.array(
                        [[pose[i].x, pose[i].y, pose[i].z] for i in UPPER_BODY_INDICES],
                        dtype=np.float32,
                    )
                    result.detected = True
                    result.landmarks = coords
                    result.confidence = 0.90

            elif self._legacy_pose is not None:
                res = self._legacy_pose.process(rgb_frame)
                result.raw_results = res
                if res.pose_landmarks:
                    pose = res.pose_landmarks
                    coords = np.array(
                        [[pose.landmark[i].x, pose.landmark[i].y, pose.landmark[i].z] for i in UPPER_BODY_INDICES],
                        dtype=np.float32,
                    )
                    result.detected = True
                    result.landmarks = coords
                    result.confidence = 0.85

        except Exception as exc:
            logger.warning("PoseLandmarker error (suppressed to preserve hand pipeline): %s", exc)

        result.inference_latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def extract_features(self, result: PoseLandmarkResult) -> np.ndarray:
        """Return flat feature vector of shape (NUM_POSE_FEATURES,) = (24,)."""
        if result.detected and result.landmarks is not None:
            return result.landmarks.flatten()
        return np.zeros(NUM_POSE_FEATURES, dtype=np.float32)

    def close(self):
        if self._task_landmarker is not None:
            try:
                self._task_landmarker.close()
            except Exception:
                pass
            self._task_landmarker = None

        if self._legacy_pose is not None:
            try:
                self._legacy_pose.close()
            except Exception:
                pass
            self._legacy_pose = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
