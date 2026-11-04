"""
SignBridge AI — ISL Fingerspelling / Alphabet Recognizer (A–Z).
Dedicated recognizer for Indian Sign Language Alphabet / Fingerspelling (26 classes).
Evaluates 30-frame hand landmark sequences using models/isl_fingerspelling_model.keras
conforming strictly to Feature Schema v2.0.0 (282 dimensions).
"""

from collections import deque
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

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "isl_fingerspelling_model.keras"
CAT_MODEL_PATH = PROJECT_ROOT / "models" / "alphabet" / "hand_sequence_lstm.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "fingerspelling_labels.json"
CAT_LABELS_PATH = PROJECT_ROOT / "models" / "alphabet" / "labels.json"


class ISLFingerspellingRecognizer:
    """
    Dedicated ISL Alphabet / Fingerspelling (A-Z) classifier.
    Operates on a sliding 30-frame window of 282-dim hand landmarks.
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        min_confidence: float = 0.50,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
    ):
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.model_path = model_path or (DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else CAT_MODEL_PATH)
        self.labels_path = labels_path or (DEFAULT_LABELS_PATH if DEFAULT_LABELS_PATH.exists() else CAT_LABELS_PATH)

        self.classes: List[str] = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        self.id_to_label: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}
        self.label_to_id: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self._load_labels()

        self.model = None
        self._load_model()

        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._last_letter: Optional[str] = None
        self._last_conf: float = 0.0

    def _load_labels(self):
        """Load alphabet class mappings."""
        for p in [self.labels_path, DEFAULT_LABELS_PATH, CAT_LABELS_PATH]:
            if p and Path(p).exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.classes = data.get("classes") or data.get("labels") or self.classes
                    if "id_to_label" in data:
                        self.id_to_label = {int(k): v for k, v in data["id_to_label"].items()}
                    else:
                        self.id_to_label = {i: c for i, c in enumerate(self.classes)}
                    self.label_to_id = data.get("label_to_id", {c: i for i, c in enumerate(self.classes)})
                    logger.info("ISLFingerspellingRecognizer: Loaded %d classes from %s", len(self.classes), p)
                    return
                except Exception as e:
                    logger.warning("ISLFingerspellingRecognizer: Failed loading labels from %s: %s", p, e)

    def _load_model(self):
        """Load trained Keras fingerspelling model and verify schema compatibility."""
        import tensorflow as tf
        for p in [self.model_path, DEFAULT_MODEL_PATH, CAT_MODEL_PATH]:
            if p and Path(p).exists():
                try:
                    loaded = tf.keras.models.load_model(str(p), compile=False)
                    verify_model_compatibility(loaded, expected_dim=FEATURE_DIM, expected_seq_len=self.sequence_length)
                    self.model = loaded
                    logger.info("ISLFingerspellingRecognizer: Successfully loaded compatible model from %s", p)
                    return
                except IncompatibleModelError as ime:
                    logger.error("ISLFingerspellingRecognizer: Incompatible model rejected at %s: %s", p, ime)
                except Exception as e:
                    logger.warning("ISLFingerspellingRecognizer: Failed loading model from %s: %s", p, e)
        logger.info("ISLFingerspellingRecognizer: Model not found or incompatible at %s", self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def reset_buffer(self):
        self._buffer.clear()
        self._last_letter = None
        self._last_conf = 0.0

    def add_frame(self, feature_vector: Optional[np.ndarray], has_hands: bool):
        """Add a 282-dim hand landmark feature vector to the sequence buffer."""
        if not has_hands or feature_vector is None:
            self._buffer.clear()
            return

        if not validate_features(feature_vector):
            logger.debug("ISLFingerspellingRecognizer: Rejected invalid feature vector of shape %s", getattr(feature_vector, 'shape', None))
            return

        self._buffer.append(feature_vector.astype(np.float32))

    def predict(self) -> Dict[str, Any]:
        """Perform fingerspelling prediction on the current buffer."""
        if not self.is_loaded:
            return {
                "letter": None,
                "word": None,
                "label": "Model Not Available",
                "is_new_sign": False,
                "confidence": 0.0,
                "status": "Model Not Available",
                "is_model_available": False,
            }

        if len(self._buffer) < 10:
            return {
                "letter": None,
                "word": None,
                "label": "Collecting",
                "is_new_sign": False,
                "confidence": 0.0,
                "status": f"Collecting letter frames ({len(self._buffer)}/{self.sequence_length})",
                "is_model_available": True,
            }

        # Interpolate / resample buffer to target SEQUENCE_LENGTH using shared schema
        seq = resample_sequence(np.asarray(self._buffer, dtype=np.float32), target_length=self.sequence_length)
        batch = np.expand_dims(seq, axis=0)  # Shape (1, 30, 282)

        try:
            probs = self.model.predict(batch, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            pred_letter = self.id_to_label.get(top_idx, self.classes[top_idx] if top_idx < len(self.classes) else "?")

            if top_conf < self.min_confidence:
                return {
                    "letter": None,
                    "word": None,
                    "label": "Unknown",
                    "confidence": top_conf,
                    "status": "Unknown Sign",
                    "is_new_sign": False,
                    "is_model_available": True,
                }

            self._last_letter = pred_letter
            self._last_conf = top_conf

            return {
                "letter": pred_letter,
                "word": pred_letter,
                "label": pred_letter,
                "is_new_sign": True,
                "confidence": top_conf,
                "status": f"Letter {pred_letter} ({top_conf:.1%})",
                "is_model_available": True,
            }
        except Exception as e:
            logger.error("ISLFingerspellingRecognizer: Inference error: %s", e)
            return {
                "letter": None,
                "word": None,
                "label": "Error",
                "confidence": 0.0,
                "status": f"Inference Error: {e}",
                "is_new_sign": False,
                "is_model_available": True,
            }
