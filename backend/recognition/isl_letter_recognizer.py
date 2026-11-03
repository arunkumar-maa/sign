"""
SignBridge AI — Phase 8: ISL Fingerspelling & Letter Recognizer.
Independent from Word Recognition.
Enforces strict ISL provenance (rejects American Sign Language rules/heuristics).
When a verified ISL letter model is not present, clearly reports:
'Training data required: Verified ISL alphabet dataset needed.'
Never outputs arbitrary letters or mock predictions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LETTER_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "isl_letter_model.keras"
DEFAULT_LETTER_LABELS_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "letter_labels.json"


class ISLLetterRecognizer:
    """
    Dedicated classifier for Indian Sign Language (ISL) fingerspelling and alphabet letters.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        confidence_threshold: float = 0.70,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_LETTER_MODEL_PATH
        self.labels_path = Path(labels_path) if labels_path else DEFAULT_LETTER_LABELS_PATH
        self.confidence_threshold = confidence_threshold

        self._model = None
        self._labels: List[str] = []
        self._is_loaded = False

        self._load_model()

    def _load_model(self):
        """Attempt to load verified ISL alphabet model."""
        if not self.model_path.exists() or not self.labels_path.exists():
            logger.info("ISL letter model not yet trained. Status: Training data required.")
            self._is_loaded = False
            return

        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(str(self.model_path))
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._labels = data.get("labels", [])
            self._is_loaded = True
            logger.info("ISL Letter Recognizer loaded successfully (%d letter classes)", len(self._labels))
        except Exception as exc:
            logger.warning("Could not load ISL letter model: %s", exc)
            self._model = None
            self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None and len(self._labels) > 0

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single static frame or sequence of features.
        Returns 'Training data required' if no verified model is loaded.
        """
        t0 = time.perf_counter()

        if not self.is_loaded:
            return {
                "letter": None,
                "confidence": 0.0,
                "status": "Letter Recognition: Training data required",
                "status_detail": "training_data_required",
                "message": "Verified ISL fingerspelling dataset required for alphabet training.",
                "latency_ms": 0.0,
            }

        if feature_vector is None or len(feature_vector) == 0:
            return {
                "letter": None,
                "confidence": 0.0,
                "status": "No hand detected",
                "status_detail": "no_hand",
                "latency_ms": 0.0,
            }

        try:
            arr = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
            probs = self._model.predict(arr, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            conf = float(probs[top_idx])
            latency = (time.perf_counter() - t0) * 1000.0

            if conf < self.confidence_threshold:
                return {
                    "letter": None,
                    "confidence": conf,
                    "status": "Unknown letter sign",
                    "status_detail": "low_confidence",
                    "latency_ms": latency,
                }

            letter = self._labels[top_idx]
            return {
                "letter": letter,
                "confidence": conf,
                "status": f"Letter recognized: {letter}",
                "status_detail": "recognized",
                "latency_ms": latency,
            }
        except Exception as exc:
            logger.error("Letter recognition error: %s", exc)
            return {
                "letter": None,
                "confidence": 0.0,
                "status": f"Error: {exc}",
                "status_detail": "error",
                "latency_ms": 0.0,
            }

