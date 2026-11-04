"""
SignBridge AI — Phase 3: Indian Sign Language (ISL) Number Recognizer (0–9).
Dedicated recognizer for Indian Sign Language numbers (10 classes: 0 through 9).
Evaluates 30-frame hand landmark sequences using models/isl_number_model.keras
conforming strictly to Feature Schema v2.0.0 (282 dimensions).
"""

from collections import deque
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from backend.vision.feature_schema import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    IncompatibleModelError,
    resample_sequence,
    validate_features,
    verify_model_compatibility,
)
from backend.vision.hand_pose_features import (
    calculate_angle_3d,
    WRIST,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
    INDEX_MCP,
    MIDDLE_MCP,
    RING_MCP,
    PINKY_MCP,
    FINGERTIP_INDICES,
)
from backend.recognition.finger_counter import FingerCounter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "isl_number_model.keras"
CAT_MODEL_PATH = PROJECT_ROOT / "models" / "number" / "hand_sequence_lstm.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "number_labels.json"
CAT_LABELS_PATH = PROJECT_ROOT / "models" / "number" / "labels.json"


@dataclass
class ISLNumberResult:
    """Output from ISL Number recognition."""
    number: Optional[int] = None
    label: str = "Unknown Sign"
    confidence: float = 0.0
    is_valid: bool = False
    description: str = "Waiting for hand sign..."
    hand_detected: bool = False

    def summary(self) -> str:
        if not self.hand_detected:
            return "ISL Number: — (No hand detected)"
        if not self.is_valid or self.number is None:
            return f"ISL Number: Unknown Sign (Confidence: {self.confidence:.0%})"
        return f"ISL Number: {self.number} ({self.label}) — Conf: {self.confidence:.0%}"


class ISLNumberRecognizer:
    """
    Dedicated Indian Sign Language (ISL) Number Recognizer (Digits 0-9).
    Evaluates 282-dim sequences via trained BiLSTM neural model with margin gating.
    Never uses heuristic finger counts to masquerade as trained model predictions.
    """

    ISL_NUMBER_NAMES = {
        0: "ZERO",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
        6: "SIX",
        7: "SEVEN",
        8: "EIGHT",
        9: "NINE",
    }

    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        min_confidence: float = 0.50,
        margin_threshold: float = 0.10,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        self.model_path = model_path or (DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else CAT_MODEL_PATH)
        self.labels_path = labels_path or (DEFAULT_LABELS_PATH if DEFAULT_LABELS_PATH.exists() else CAT_LABELS_PATH)
        self.min_confidence = min_confidence
        self.margin_threshold = margin_threshold
        self.sequence_length = sequence_length

        self.finger_counter = FingerCounter()
        self.classes: List[str] = [str(i) for i in range(10)]
        self.id_to_label: Dict[int, str] = {i: str(i) for i in range(10)}
        self.label_to_id: Dict[str, int] = {str(i): i for i in range(10)}
        self._load_labels()

        self.trained_model = None
        self._load_model()

        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._last_number: Optional[int] = None
        self._last_conf: float = 0.0

    def _load_labels(self):
        """Load number class mappings."""
        for p in [self.labels_path, DEFAULT_LABELS_PATH, CAT_LABELS_PATH]:
            if p and Path(p).exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.classes = data.get("classes") or data.get("labels") or self.classes
                    if "id_to_label" in data:
                        self.id_to_label = {int(k): str(v) for k, v in data["id_to_label"].items()}
                    else:
                        self.id_to_label = {i: str(c) for i, c in enumerate(self.classes)}
                    self.label_to_id = {str(k): int(v) for k, v in data.get("class_to_id", {c: i for i, c in enumerate(self.classes)}).items()}
                    logger.info("ISLNumberRecognizer: Loaded %d number classes from %s", len(self.classes), p)
                    return
                except Exception as e:
                    logger.warning("ISLNumberRecognizer: Failed loading labels from %s: %s", p, e)

    def _load_model(self):
        """Load trained Keras number model and verify schema compatibility."""
        import tensorflow as tf
        candidates = [self.model_path, DEFAULT_MODEL_PATH, CAT_MODEL_PATH]
        for p in candidates:
            if p and Path(p).exists():
                try:
                    loaded = tf.keras.models.load_model(str(p), compile=False)
                    verify_model_compatibility(loaded, expected_dim=FEATURE_DIM, expected_seq_len=self.sequence_length)
                    if hasattr(loaded, "output_shape") and loaded.output_shape is not None:
                        out_dim = loaded.output_shape[-1]
                        if out_dim != 10:
                            raise IncompatibleModelError(f"Number model output shape {loaded.output_shape} != 10 classes.")
                    self.trained_model = loaded
                    logger.info("ISLNumberRecognizer: Loaded compatible model from %s", p)
                    return
                except IncompatibleModelError as ime:
                    logger.error("ISLNumberRecognizer: Rejected incompatible model at %s: %s", p, ime)
                except Exception as e:
                    logger.warning("ISLNumberRecognizer: Could not load model from %s: %s", p, e)
        logger.info("ISLNumberRecognizer: No compatible trained number model found at %s", self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.trained_model is not None

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def reset_buffer(self):
        self._buffer.clear()
        self._last_number = None
        self._last_conf = 0.0

    def add_frame(self, feature_vector: Optional[np.ndarray], has_hands: bool):
        """Feed a 282-dim hand landmark feature vector into the sliding window."""
        if not has_hands or feature_vector is None:
            self._buffer.clear()
            return
        if not validate_features(feature_vector):
            logger.debug("ISLNumberRecognizer: Rejected invalid feature vector of shape %s", getattr(feature_vector, 'shape', None))
            return
        self._buffer.append(feature_vector.astype(np.float32))

    def predict(self) -> Dict[str, Any]:
        """Classify ISL Number from current sequence buffer."""
        if not self.is_loaded:
            return {
                "number": None,
                "label": "Model Not Available",
                "confidence": 0.0,
                "status": "Model Not Available",
                "is_model_available": False,
            }

        if len(self._buffer) < 10:
            return {
                "number": None,
                "label": "Collecting",
                "confidence": 0.0,
                "status": f"Collecting number frames ({len(self._buffer)}/{self.sequence_length})",
                "is_model_available": True,
            }

        seq = resample_sequence(np.asarray(self._buffer, dtype=np.float32), target_length=self.sequence_length)
        batch = np.expand_dims(seq, axis=0)
        try:
            probs = self.trained_model.predict(batch, verbose=0)[0]
            if len(probs) != len(self.classes):
                logger.error("Number model output dimension mismatch: %d != %d", len(probs), len(self.classes))
                return {
                    "number": None,
                    "label": "Error",
                    "confidence": 0.0,
                    "status": "Model Dimension Mismatch",
                    "is_model_available": True,
                }

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            top_conf = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = top_conf - second_conf

            if top_conf < self.min_confidence or margin < self.margin_threshold:
                return {
                    "number": None,
                    "label": "Unknown",
                    "confidence": top_conf,
                    "status": "Unknown Sign",
                    "is_model_available": True,
                }

            self._last_number = top_idx
            self._last_conf = top_conf
            num_str = str(top_idx)
            num_name = self.ISL_NUMBER_NAMES.get(top_idx, num_str)
            return {
                "number": top_idx,
                "label": num_str,
                "confidence": top_conf,
                "status": f"Number {num_str} ({num_name})",
                "is_model_available": True,
            }
        except Exception as e:
            logger.error("ISLNumberRecognizer: Inference error: %s", e)
            return {
                "number": None,
                "label": "Error",
                "confidence": 0.0,
                "status": f"Inference Error: {e}",
                "is_model_available": True,
            }

    def recognize_geometric(
        self,
        landmarks: Optional[np.ndarray],
        handedness: str = "Right",
    ) -> ISLNumberResult:
        """
        Recognize ISL Number sign from 21 MediaPipe hand landmarks using geometric kinematics.
        Diagnostic fallback helper; not called during neural prediction.
        """
        if landmarks is None or len(landmarks) < 21:
            return ISLNumberResult(hand_detected=False)

        lms = np.asarray(landmarks, dtype=np.float32)[:21, :3]
        wrist = lms[WRIST]
        palm_scale = float(np.linalg.norm(lms[MIDDLE_MCP] - wrist))
        if palm_scale < 1e-5:
            palm_scale = 1.0

        f_res = self.finger_counter.count_fingers(lms, handedness=handedness)
        ext = f_res.extended_fingers
        thumb = ext["thumb"]
        index = ext["index"]
        middle = ext["middle"]
        ring = ext["ring"]
        pinky = ext["pinky"]
        count = f_res.count

        thumb_tip = lms[THUMB_TIP]
        d_thumb_index = float(np.linalg.norm(thumb_tip - lms[INDEX_TIP])) / palm_scale
        d_thumb_middle = float(np.linalg.norm(thumb_tip - lms[MIDDLE_TIP])) / palm_scale
        d_thumb_ring = float(np.linalg.norm(thumb_tip - lms[RING_TIP])) / palm_scale
        d_thumb_pinky = float(np.linalg.norm(thumb_tip - lms[PINKY_TIP])) / palm_scale

        TOUCH_THRESHOLD = 0.35
        touch_index = d_thumb_index < TOUCH_THRESHOLD
        touch_middle = d_thumb_middle < TOUCH_THRESHOLD
        touch_ring = d_thumb_ring < TOUCH_THRESHOLD
        touch_pinky = d_thumb_pinky < TOUCH_THRESHOLD

        if (pinky and not index and not middle and not ring) or (thumb and not index and not middle and not ring and not pinky):
            return ISLNumberResult(number=None, label="Unknown Sign", confidence=0.20, is_valid=False, description="Non-number handshape", hand_detected=True)

        if count == 0 and not thumb and not index and not middle and not ring and not pinky:
            return ISLNumberResult(number=0, label="ZERO", confidence=0.90, is_valid=True, description="Fist / Zero", hand_detected=True)

        if count == 0 and touch_index and not middle and not ring and not pinky:
            return ISLNumberResult(number=0, label="ZERO", confidence=0.90, is_valid=True, description="Circular O shape / Zero", hand_detected=True)

        if index and not middle and not ring and not pinky:
            return ISLNumberResult(number=1, label="ONE", confidence=0.92, is_valid=True, description="Index extended", hand_detected=True)

        if index and middle and not ring and not pinky and not thumb:
            return ISLNumberResult(number=2, label="TWO", confidence=0.92, is_valid=True, description="Index + Middle extended (V-sign)", hand_detected=True)

        if thumb and index and middle and not ring and not pinky:
            return ISLNumberResult(number=3, label="THREE", confidence=0.90, is_valid=True, description="Thumb + Index + Middle extended", hand_detected=True)

        if index and middle and ring and pinky and not thumb:
            return ISLNumberResult(number=4, label="FOUR", confidence=0.92, is_valid=True, description="Four fingers extended, thumb folded", hand_detected=True)

        if count == 5:
            return ISLNumberResult(number=5, label="FIVE", confidence=0.95, is_valid=True, description="All 5 fingers extended open", hand_detected=True)

        if touch_pinky and (index or middle):
            return ISLNumberResult(number=6, label="SIX", confidence=0.88, is_valid=True, description="Thumb touching pinky", hand_detected=True)

        if touch_ring and (index or pinky):
            return ISLNumberResult(number=7, label="SEVEN", confidence=0.88, is_valid=True, description="Thumb touching ring", hand_detected=True)

        if touch_middle and (index or pinky):
            return ISLNumberResult(number=8, label="EIGHT", confidence=0.88, is_valid=True, description="Thumb touching middle", hand_detected=True)

        if touch_index and (middle or ring):
            return ISLNumberResult(number=9, label="NINE", confidence=0.88, is_valid=True, description="Thumb touching index", hand_detected=True)

        return ISLNumberResult(number=None, label="Unknown Sign", confidence=0.30, is_valid=False, description="Unrecognized handshape", hand_detected=True)

    # Alias for backward compatibility
    recognize = recognize_geometric
