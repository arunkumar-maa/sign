"""
Dynamic gesture recognizer using LSTM/GRU sequence model.
"""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_UNAVAILABLE_MSG = "Hand tracking is working, but ISL recognition model is unavailable."


class DynamicRecognizer:
    """
    Maintains a sliding window of frame features and runs LSTM inference
    when the window is full.
    """

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        sequence_length: int = 30,
        confidence_threshold: float = 0.70,
    ):
        self._model = None
        self._labels: list[str] = []
        self._threshold = confidence_threshold
        self._sequence_length = sequence_length
        self._model_path = model_path
        self._labels_path = labels_path
        self._model_version = "not_loaded"
        self._buffer: deque = deque(maxlen=sequence_length)

        self._load()

    def _load(self):
        if not self._model_path.exists():
            logger.warning("Dynamic model not found at %s. %s", self._model_path, _MODEL_UNAVAILABLE_MSG)
            return
        if not self._labels_path.exists():
            logger.warning("Dynamic labels not found at %s.", self._labels_path)
            return
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(str(self._model_path))
            with open(self._labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._labels = data.get("labels", data) if isinstance(data, dict) else data
            self._model_version = data.get("version", "1.0") if isinstance(data, dict) else "1.0"
            logger.info(
                "Dynamic model loaded — %d classes, seq_len=%d, version=%s",
                len(self._labels), self._sequence_length, self._model_version,
            )
        except Exception as exc:
            logger.error("Failed to load dynamic model: %s", exc)
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and len(self._labels) > 0

    @property
    def buffer_ready(self) -> bool:
        return len(self._buffer) >= self._sequence_length

    def add_frame(self, features: np.ndarray):
        """Add one frame's feature vector to the sliding window buffer."""
        self._buffer.append(features.astype(np.float32))

    def reset_buffer(self):
        """Clear the sequence buffer (call after a sign is confirmed)."""
        self._buffer.clear()

    def predict(self) -> Dict[str, Any]:
        """
        Run inference on the current buffer if full.
        
        Returns:
            Recognition result dict.
        """
        timestamp = time.time()

        if not self.is_loaded:
            return {
                "gesture": None,
                "confidence": 0.0,
                "status": "model_unavailable",
                "recognition_type": "dynamic",
                "model_version": self._model_version,
                "timestamp": timestamp,
                "message": _MODEL_UNAVAILABLE_MSG,
            }

        if not self.buffer_ready:
            return {
                "gesture": None,
                "confidence": 0.0,
                "status": "buffering",
                "recognition_type": "dynamic",
                "model_version": self._model_version,
                "timestamp": timestamp,
            }

        try:
            sequence = np.array(list(self._buffer), dtype=np.float32)
            sequence = sequence.reshape(1, self._sequence_length, -1)
            probs = self._model.predict(sequence, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])

            if confidence < self._threshold:
                return {
                    "gesture": None,
                    "confidence": confidence,
                    "status": "low_confidence",
                    "recognition_type": "dynamic",
                    "model_version": self._model_version,
                    "timestamp": timestamp,
                }

            return {
                "gesture": self._labels[top_idx],
                "confidence": confidence,
                "status": "recognized",
                "recognition_type": "dynamic",
                "model_version": self._model_version,
                "timestamp": timestamp,
            }
        except Exception as exc:
            logger.error("Dynamic inference error: %s", exc)
            return {
                "gesture": None,
                "confidence": 0.0,
                "status": "error",
                "recognition_type": "dynamic",
                "model_version": self._model_version,
                "timestamp": timestamp,
                "error": str(exc),
            }
