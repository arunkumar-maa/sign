"""
SignBridge AI — Phase 3: Indian Sign Language (ISL) Number Recognizer.
Recognizes actual ISL number signs (0 through 9) using verified ISL standards:
- ISL 0: Closed fist or circular 'O' shape.
- ISL 1: Index finger extended, palm facing forward, others folded.
- ISL 2: Index and Middle fingers extended in V-shape, others folded.
- ISL 3: Thumb, Index, and Middle fingers extended (standard ISL 3).
- ISL 4: Four fingers (Index, Middle, Ring, Pinky) extended, thumb folded across palm.
- ISL 5: All 5 fingers extended open palm facing forward.
- ISL 6: Thumb touches Pinky fingertip, 3 other fingers extended upright.
- ISL 7: Thumb touches Ring fingertip, 3 other fingers extended upright.
- ISL 8: Thumb touches Middle fingertip, 3 other fingers extended upright.
- ISL 9: Thumb touches Index fingertip, 3 other fingers extended upright.

STRICT REQUIREMENTS:
- Separate from finger counter (Finger count diagnostic vs. semantic ISL number).
- Does not use ASL number conventions.
- Displays 'Unknown Sign' for ambiguous or non-number handshapes.
- Supports optional trained Keras model with geometric fallback.
"""

from dataclasses import dataclass, field
from collections import deque
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "isl_number_model.keras"
CAT_MODEL_PATH = PROJECT_ROOT / "models" / "number" / "hand_sequence_lstm.keras"
HAND_POSE_MODEL_PATH = PROJECT_ROOT / "models" / "hand_pose" / "isl_numbers.keras"

DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "number_labels.json"
CAT_LABELS_PATH = PROJECT_ROOT / "models" / "number" / "labels.json"
HAND_POSE_LABELS_PATH = PROJECT_ROOT / "models" / "hand_pose" / "number_labels.json"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 282


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
    Combines trained number classifier with strict geometric syntactic verification.
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
        min_confidence: float = 0.55,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.min_confidence = min_confidence
        self.sequence_length = sequence_length
        self.finger_counter = FingerCounter()
        self.trained_model = None
        self._load_model()
        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._last_number: Optional[int] = None
        self._last_conf: float = 0.0

    def _load_model(self):
        """Attempt to load trained Keras number model if available."""
        import tensorflow as tf
        candidates = [self.model_path, DEFAULT_MODEL_PATH, CAT_MODEL_PATH, HAND_POSE_MODEL_PATH]
        for p in candidates:
            if p and p.exists():
                try:
                    self.trained_model = tf.keras.models.load_model(str(p), compile=False)
                    logger.info("ISLNumberRecognizer: Loaded trained model from %s", p)
                    return
                except Exception as e:
                    logger.warning("ISLNumberRecognizer: Could not load model from %s: %s", p, e)
        logger.info("ISLNumberRecognizer: No trained number model found; will report Model Not Available")

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
        if len(feature_vector) != FEATURE_DIM:
            vec = np.zeros(FEATURE_DIM, dtype=np.float32)
            n = min(len(feature_vector), FEATURE_DIM)
            vec[:n] = feature_vector[:n]
            feature_vector = vec
        self._buffer.append(feature_vector.astype(np.float32))

    def predict(self) -> Dict[str, Any]:
        """Classify ISL Number from current sequence buffer."""
        if not self.is_loaded:
            return {
                "number": None,
                "confidence": 0.0,
                "status": "Model Not Available",
                "is_model_available": False,
            }

        if len(self._buffer) < 10:
            return {
                "number": None,
                "confidence": 0.0,
                "status": f"Collecting number frames ({len(self._buffer)}/{self.sequence_length})",
                "is_model_available": True,
            }

        seq = np.asarray(self._buffer, dtype=np.float32)
        if len(seq) != self.sequence_length:
            positions = np.linspace(0, len(seq) - 1, self.sequence_length)
            left = np.floor(positions).astype(int)
            right = np.ceil(positions).astype(int)
            weight = (positions - left)[:, None].astype(np.float32)
            seq = (seq[left] * (1.0 - weight) + seq[right] * weight).astype(np.float32)

        batch = np.expand_dims(seq, axis=0)
        try:
            probs = self.trained_model.predict(batch, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])

            if top_conf < self.min_confidence:
                return {
                    "number": None,
                    "confidence": top_conf,
                    "status": "Unknown Sign",
                    "is_model_available": True,
                }

            self._last_number = top_idx
            self._last_conf = top_conf
            num_name = self.ISL_NUMBER_NAMES.get(top_idx, str(top_idx))
            return {
                "number": top_idx,
                "confidence": top_conf,
                "status": f"Number {top_idx} ({num_name})",
                "is_model_available": True,
            }
        except Exception as e:
            logger.error("ISLNumberRecognizer: Inference error: %s", e)
            return {
                "number": None,
                "confidence": 0.0,
                "status": f"Inference Error: {e}",
                "is_model_available": True,
            }

    def recognize(
        self,
        landmarks: Optional[np.ndarray],
        handedness: str = "Right",
    ) -> ISLNumberResult:
        """
        Recognize ISL Number sign from 21 MediaPipe hand landmarks.

        Args:
            landmarks: shape (21, 3)
            handedness: 'Left' or 'Right'

        Returns:
            ISLNumberResult
        """
        if landmarks is None or len(landmarks) < 21:
            return ISLNumberResult(hand_detected=False)

        lms = np.asarray(landmarks, dtype=np.float32)[:21, :3]
        wrist = lms[WRIST]
        palm_scale = float(np.linalg.norm(lms[MIDDLE_MCP] - wrist))
        if palm_scale < 1e-5:
            palm_scale = 1.0

        # Step 1: Geometric finger extension state
        f_res = self.finger_counter.count_fingers(lms, handedness=handedness)
        ext = f_res.extended_fingers
        thumb = ext["thumb"]
        index = ext["index"]
        middle = ext["middle"]
        ring = ext["ring"]
        pinky = ext["pinky"]
        count = f_res.count

        # Step 2: Fingertip touch detection (Thumb tip to other fingertips)
        thumb_tip = lms[THUMB_TIP]
        d_thumb_index = float(np.linalg.norm(thumb_tip - lms[INDEX_TIP])) / palm_scale
        d_thumb_middle = float(np.linalg.norm(thumb_tip - lms[MIDDLE_TIP])) / palm_scale
        d_thumb_ring = float(np.linalg.norm(thumb_tip - lms[RING_TIP])) / palm_scale
        d_thumb_pinky = float(np.linalg.norm(thumb_tip - lms[PINKY_TIP])) / palm_scale

        TOUCH_THRESHOLD = 0.32  # Normalized touch threshold

        touch_index = d_thumb_index < TOUCH_THRESHOLD
        touch_middle = d_thumb_middle < TOUCH_THRESHOLD
        touch_ring = d_thumb_ring < TOUCH_THRESHOLD
        touch_pinky = d_thumb_pinky < TOUCH_THRESHOLD

        pred_num: Optional[int] = None
        conf: float = 0.0
        desc: str = ""

        # ISL 9: Thumb touching Index tip, remaining 3 extended (Middle, Ring, Pinky)
        if touch_index and middle and ring and pinky:
            pred_num = 9
            conf = 0.92
            desc = "ISL 9: Thumb touching Index tip, 3 fingers upright"

        # ISL 8: Thumb touching Middle tip, remaining 3 extended (Index, Ring, Pinky)
        elif touch_middle and index and ring and pinky:
            pred_num = 8
            conf = 0.92
            desc = "ISL 8: Thumb touching Middle tip, 3 fingers upright"

        # ISL 7: Thumb touching Ring tip, remaining 3 extended (Index, Middle, Pinky)
        elif touch_ring and index and middle and pinky:
            pred_num = 7
            conf = 0.92
            desc = "ISL 7: Thumb touching Ring tip, 3 fingers upright"

        # ISL 6: Thumb touching Pinky tip, remaining 3 extended (Index, Middle, Ring)
        elif touch_pinky and index and middle and ring:
            pred_num = 6
            conf = 0.92
            desc = "ISL 6: Thumb touching Pinky tip, 3 fingers upright"

        # ISL 5: All 5 fingers extended open palm
        elif count == 5 and thumb and index and middle and ring and pinky:
            pred_num = 5
            conf = 0.95
            desc = "ISL 5: Five fingers extended open palm"

        # ISL 4: Four fingers (Index, Middle, Ring, Pinky) extended, thumb folded across palm
        elif count == 4 and not thumb and index and middle and ring and pinky:
            pred_num = 4
            conf = 0.94
            desc = "ISL 4: Four fingers upright, thumb folded across palm"

        # ISL 3: Thumb, Index, and Middle fingers extended (Standard ISL 3)
        elif (count == 3 and thumb and index and middle and not ring and not pinky) or (
            count == 3 and not thumb and index and middle and ring and not pinky
        ):
            pred_num = 3
            conf = 0.90
            desc = "ISL 3: Three fingers extended"

        # ISL 2: Index and Middle fingers extended upright (V-shape), others folded
        elif count == 2 and not thumb and index and middle and not ring and not pinky:
            pred_num = 2
            conf = 0.95
            desc = "ISL 2: Index and Middle extended (V-shape)"

        # ISL 1: Only Index finger extended upright, thumb over folded fingers
        elif count == 1 and not thumb and index and not middle and not ring and not pinky:
            pred_num = 1
            conf = 0.95
            desc = "ISL 1: Single index finger extended upright"

        # ISL 0: Closed fist (all fingers folded close to palm)
        elif count == 0 and not thumb and not index and not middle and not ring and not pinky:
            # Anatomical plausibility: in a true fist, fingertips are curled in towards palm/wrist
            tip_dists = [
                float(np.linalg.norm(lms[tip] - lms[WRIST])) / palm_scale
                for tip in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            ]
            if np.mean(tip_dists) < 1.35:
                pred_num = 0
                conf = 0.92
                desc = "ISL 0: Closed fist"

        # Non-matching or ambiguous handshape -> Unknown Sign
        if pred_num is None or conf < self.min_confidence:
            return ISLNumberResult(
                number=None,
                label="Unknown Sign",
                confidence=round(conf, 3),
                is_valid=False,
                description="Handshape does not match standard ISL numbers (0-9)",
                hand_detected=True,
            )

        return ISLNumberResult(
            number=pred_num,
            label=self.ISL_NUMBER_NAMES[pred_num],
            confidence=round(conf, 3),
            is_valid=True,
            description=desc,
            hand_detected=True,
        )
