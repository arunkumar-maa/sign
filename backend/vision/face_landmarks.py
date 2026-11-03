"""
Face landmark detection using the official MediaPipe Tasks Face Landmarker API.
Extracts selected facial landmarks relevant to non-manual ISL grammatical markers
(eyebrows, eye apertures, mouth expressions, head/chin anchor).
Operates completely independently so face detection errors never crash hand tracking.
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

# 14 Landmark indices: Eye corners, Eyebrows, Mouth corners, Lips, Nose tip, Chin
# 14 points × 3 (x,y,z) = 42 features
USEFUL_FACE_INDICES = [
    33, 133, 362, 263,   # eye corners (inner/outer for both eyes)
    70, 300,              # eyebrow outer
    105, 334,             # eyebrow inner
    61, 291,              # mouth corners
    13, 14,               # lips (upper and lower midpoints)
    1,                    # nose tip
    152,                  # chin point
]
NUM_FACE_FEATURES = len(USEFUL_FACE_INDICES) * 3  # 42

DEFAULT_FACE_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "mediapipe" / "face_landmarker.task"


@dataclass
class FaceLandmarkResult:
    """Result from face landmark detection."""
    detected: bool = False
    landmarks: Optional[np.ndarray] = None  # shape (14, 3) [x, y, z]
    confidence: float = 0.0
    inference_latency_ms: float = 0.0
    raw_results: object = field(default=None, repr=False)


class FaceLandmarker:
    """
    Wraps MediaPipe Tasks FaceLandmarker for facial landmark extraction.
    Independent and fully isolated from hand and pose detection.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_FACE_MODEL_PATH
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence

        self._task_landmarker = None
        self._legacy_mesh = None
        self._using_tasks_api = False

        self._init_detector()

    def _init_detector(self):
        """Initialise Tasks API with fallback to legacy solutions."""
        if self.model_path.exists():
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=str(self.model_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=self.max_num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_face_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_detection_confidence,
                )
                self._task_landmarker = vision.FaceLandmarker.create_from_options(options)
                self._using_tasks_api = True
                logger.info("FaceLandmarker Tasks API initialized from %s", self.model_path)
                return
            except Exception as e:
                logger.warning("Tasks FaceLandmarker init failed: %s. Falling back to legacy.", e)

        try:
            self._legacy_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_num_faces,
                refine_landmarks=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_detection_confidence,
            )
            self._using_tasks_api = False
            logger.info("Legacy mp.solutions.face_mesh initialized.")
        except Exception as e:
            logger.error("Failed to initialize FaceLandmarker: %s", e)

    def detect(self, frame: np.ndarray, is_bgr: bool = True) -> FaceLandmarkResult:
        """
        Run facial landmark detection on an image frame.
        Safe against any exception so pipeline never crashes.
        """
        if frame is None or frame.size == 0:
            return FaceLandmarkResult()

        t0 = time.perf_counter()
        result = FaceLandmarkResult()

        try:
            if is_bgr:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)

            if self._using_tasks_api and self._task_landmarker is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection = self._task_landmarker.detect(mp_image)
                result.raw_results = detection

                if detection and detection.face_landmarks and len(detection.face_landmarks) > 0:
                    face = detection.face_landmarks[0]
                    coords = np.array(
                        [[face[i].x, face[i].y, face[i].z] for i in USEFUL_FACE_INDICES],
                        dtype=np.float32,
                    )
                    result.detected = True
                    result.landmarks = coords
                    result.confidence = 0.90

            elif self._legacy_mesh is not None:
                res = self._legacy_mesh.process(rgb_frame)
                result.raw_results = res
                if res.multi_face_landmarks and len(res.multi_face_landmarks) > 0:
                    face = res.multi_face_landmarks[0]
                    coords = np.array(
                        [[face.landmark[i].x, face.landmark[i].y, face.landmark[i].z] for i in USEFUL_FACE_INDICES],
                        dtype=np.float32,
                    )
                    result.detected = True
                    result.landmarks = coords
                    result.confidence = 0.85

        except Exception as exc:
            logger.warning("FaceLandmarker error (suppressed to preserve hand pipeline): %s", exc)

        result.inference_latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def extract_features(self, result: FaceLandmarkResult) -> np.ndarray:
        """Return flat feature vector of shape (NUM_FACE_FEATURES,) = (42,)."""
        if result.detected and result.landmarks is not None:
            return result.landmarks.flatten()
        return np.zeros(NUM_FACE_FEATURES, dtype=np.float32)

    def close(self):
        if self._task_landmarker is not None:
            try:
                self._task_landmarker.close()
            except Exception:
                pass
            self._task_landmarker = None

        if self._legacy_mesh is not None:
            try:
                self._legacy_mesh.close()
            except Exception:
                pass
            self._legacy_mesh = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
