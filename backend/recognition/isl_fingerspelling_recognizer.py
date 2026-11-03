"""
SignBridge AI — ISL Fingerspelling Gesture Recognizer.
Dedicated recognizer for Indian Sign Language Alphabet / Fingerspelling (A through Z).
Evaluates 30-frame hand landmark sequences using models/isl_fingerspelling_model.keras.
"""

from collections import deque
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "isl_fingerspelling_model.keras"
CAT_MODEL_PATH = PROJECT_ROOT / "models" / "alphabet" / "hand_sequence_lstm.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "fingerspelling_labels.json"
CAT_LABELS_PATH = PROJECT_ROOT / "models" / "alphabet" / "labels.json"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 282


class ISLFingerspellingRecognizer:
    """
    Dedicated ISL Alphabet / Fingerspelling (A-Z) classifier.
    Operates on a sliding 30-frame window of 282-dim hand landmarks.
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        min_confidence: float = 0.45,
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
            if p and p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.classes = data.get("classes") or data.get("labels") or self.classes
                    self.id_to_label = {int(k): v for k, v in data.get("id_to_label", {}).items()} if "id_to_label" in data else {i: c for i, c in enumerate(self.classes)}
                    self.label_to_id = data.get("label_to_id", {c: i for i, c in enumerate(self.classes)})
                    logger.info("ISLFingerspellingRecognizer: Loaded %d classes from %s", len(self.classes), p)
                    return
                except Exception as e:
                    logger.warning("ISLFingerspellingRecognizer: Failed loading labels from %s: %s", p, e)

    def _load_model(self):
        """Load trained Keras fingerspelling model."""
        import tensorflow as tf
        for p in [self.model_path, DEFAULT_MODEL_PATH, CAT_MODEL_PATH]:
            if p and p.exists():
                try:
                    self.model = tf.keras.models.load_model(str(p), compile=False)
                    logger.info("ISLFingerspellingRecognizer: Successfully loaded model from %s", p)
                    return
                except Exception as e:
                    logger.warning("ISLFingerspellingRecognizer: Failed loading model from %s: %s", p, e)
        logger.info("ISLFingerspellingRecognizer: Model not found at %s", self.model_path)

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

        if len(feature_vector) != FEATURE_DIM:
            vec = np.zeros(FEATURE_DIM, dtype=np.float32)
            n = min(len(feature_vector), FEATURE_DIM)
            vec[:n] = feature_vector[:n]
            feature_vector = vec

        self._buffer.append(feature_vector.astype(np.float32))

    def predict(self) -> Dict[str, Any]:
        """Perform fingerspelling prediction on the current buffer."""
        if not self.is_loaded:
            return {
                "letter": None,
                "confidence": 0.0,
                "status": "Model Not Available",
                "is_model_available": False,
            }

        if len(self._buffer) < 10:
            return {
                "letter": None,
                "confidence": 0.0,
                "status": f"Collecting letter frames ({len(self._buffer)}/{self.sequence_length})",
                "is_model_available": True,
            }

        # Interpolate / resample buffer to target SEQUENCE_LENGTH
        seq = np.asarray(self._buffer, dtype=np.float32)
        if len(seq) != self.sequence_length:
            positions = np.linspace(0, len(seq) - 1, self.sequence_length)
            left = np.floor(positions).astype(int)
            right = np.ceil(positions).astype(int)
            weight = (positions - left)[:, None].astype(np.float32)
            seq = (seq[left] * (1.0 - weight) + seq[right] * weight).astype(np.float32)

        batch = np.expand_dims(seq, axis=0)  # Shape (1, 30, 282)

        try:
            probs = self.model.predict(batch, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            pred_letter = self.id_to_label.get(top_idx, self.classes[top_idx] if top_idx < len(self.classes) else "?")

            if top_conf < self.min_confidence:
                return {
                    "letter": None,
                    "confidence": top_conf,
                    "status": "Unknown Sign",
                    "is_model_available": True,
                }

            self._last_letter = pred_letter
            self._last_conf = top_conf

            return {
                "letter": pred_letter,
                "confidence": top_conf,
                "status": f"Letter {pred_letter} ({top_conf:.1%})",
                "is_model_available": True,
            }
        except Exception as e:
            logger.error("ISLFingerspellingRecognizer: Inference error: %s", e)
            return {
                "letter": None,
                "confidence": 0.0,
                "status": f"Inference Error: {e}",
                "is_model_available": True,
            }

