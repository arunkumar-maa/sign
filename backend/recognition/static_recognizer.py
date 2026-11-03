"""
Static Indian Sign Language gesture recognizer.
Performs neural inference using trained landmark neural network.
Enforces confidence thresholding, unknown-sign rejection, and graceful degradation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_UNAVAILABLE_MSG = "Hand tracking is working, but ISL recognition model is unavailable."


class StaticRecognizer:
    """
    Inference engine for static Indian Sign Language gestures.
    """

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        confidence_threshold: float = 0.65,
    ):
        self._model = None
        self._labels: List[str] = []
        self._threshold = confidence_threshold
        self._model_path = Path(model_path)
        self._labels_path = Path(labels_path)
        self._model_version = "not_loaded"

        self._load()

    def _load(self):
        """Attempt to load model and labels. Fail gracefully."""
        if not self._model_path.exists():
            logger.warning("Static model not found at %s. %s", self._model_path, _MODEL_UNAVAILABLE_MSG)
            return
        if not self._labels_path.exists():
            logger.warning("Labels file not found at %s.", self._labels_path)
            return

        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(str(self._model_path))
            with open(self._labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._labels = data.get("labels", data) if isinstance(data, dict) else data
            self._model_version = data.get("version", "1.0") if isinstance(data, dict) else "1.0"
            logger.info(
                "Static model loaded from %s — %d classes, version %s",
                self._model_path, len(self._labels), self._model_version,
            )
        except Exception as exc:
            logger.error("Failed to load static model: %s", exc)
            self._model = None

    @property
    def is_loaded(self) -> bool:
        """True if model and labels are loaded and operational."""
        return self._model is not None and len(self._labels) > 0

    @property
    def unavailable_message(self) -> str:
        return _MODEL_UNAVAILABLE_MSG

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on landmark feature vector.

        Args:
            features: 1D numpy array of landmark coordinates (e.g., shape (126,) or (192,)).

        Returns:
            Dict containing gesture, confidence, status, and metadata.
        """
        timestamp = time.time()

        if not self.is_loaded:
            return {
                "gesture": None,
                "confidence": 0.0,
                "status": "model_unavailable",
                "message": _MODEL_UNAVAILABLE_MSG,
                "recognition_type": "static",
                "model_version": self._model_version,
                "timestamp": timestamp,
            }

        # Validate input features
        if features is None or features.size == 0 or np.all(np.abs(features[:63]) < 1e-5):
            return {
                "gesture": None,
                "confidence": 0.0,
                "status": "no_hands",
                "recognition_type": "static",
                "model_version": self._model_version,
                "timestamp": timestamp,
            }

        try:
            # Match model input feature dimensions
            expected_dim = self._model.input_shape[-1]
            if features.shape[0] < expected_dim:
                padded = np.zeros(expected_dim, dtype=np.float32)
                padded[:features.shape[0]] = features
                input_arr = padded.reshape(1, -1)
            elif features.shape[0] > expected_dim:
                input_arr = features[:expected_dim].reshape(1, -1).astype(np.float32)
            else:
                input_arr = features.reshape(1, -1).astype(np.float32)

            probs = self._model.predict(input_arr, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])

            if confidence >= self._threshold:
                return {
                    "gesture": self._labels[top_idx],
                    "confidence": confidence,
                    "status": "recognized",
                    "recognition_type": "static",
                    "model_version": self._model_version,
                    "timestamp": timestamp,
                }
            else:
                return {
                    "gesture": self._labels[top_idx] if confidence > 0.45 else None,
                    "confidence": confidence,
                    "status": "low_confidence",
                    "recognition_type": "static",
                    "model_version": self._model_version,
                    "timestamp": timestamp,
                }
        except Exception as exc:
            logger.error("Static neural inference error: %s", exc)
            return {
                "gesture": None,
                "confidence": 0.0,
                "status": "error",
                "error": str(exc),
                "recognition_type": "static",
                "model_version": self._model_version,
                "timestamp": timestamp,
            }

    def get_labels(self) -> List[str]:
        return list(self._labels)
