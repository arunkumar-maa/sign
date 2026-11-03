"""
SignBridge AI — Phase 7: Live ISL Sequence Recognizer.
Maintains a 30-frame sliding window of normalized 192-dimensional multi-modal features,
runs LSTM inference using the trained 40-word ISL model, and applies prediction smoothing,
temporal stability verification, confidence thresholding, and cooldown duplicate prevention.
"""

import collections
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LiveISLRecognizer:
    """
    Live real-time inference engine for 40-word Indian Sign Language (ISL) recognition.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        sequence_length: int = 30,
        confidence_threshold: float = 0.38,
        stability_count: int = 3,
        cooldown_seconds: float = 1.5,
    ):
        self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "models" / "isl_word_model.keras"
        self.labels_path = Path(labels_path) if labels_path else PROJECT_ROOT / "models" / "word_labels.json"
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.stability_count = stability_count
        self.cooldown_seconds = cooldown_seconds

        self._model = None
        self._labels: List[str] = []
        self._is_loaded = False

        # Sliding sequence buffer: stores normalized (192,) feature vectors
        self._buffer: collections.deque = collections.deque(maxlen=self.sequence_length)

        # Temporal smoothing deque: stores recent predicted class indices
        self._recent_predictions: collections.deque = collections.deque(maxlen=self.stability_count)

        # Duplicate cooldown tracking
        self._last_emitted_word: Optional[str] = None
        self._last_emitted_time: float = 0.0

        self._load()

    def _load(self):
        """Load trained ISL sequence model and label mapping."""
        if not self.model_path.exists() or not self.labels_path.exists():
            logger.warning("ISL model or labels not found at %s / %s", self.model_path, self.labels_path)
            self._is_loaded = False
            return

        try:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._labels = data.get("labels", data) if isinstance(data, dict) else data

            import os
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            self._model = tf.keras.models.load_model(str(self.model_path))
            self._is_loaded = True
            logger.info("LiveISLRecognizer loaded successfully with %d classes", len(self._labels))
        except Exception as exc:
            logger.error("Failed to load ISL model: %s", exc)
            self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None and len(self._labels) > 0

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def reset_buffer(self):
        """Clear sequence and smoothing buffers."""
        self._buffer.clear()
        self._recent_predictions.clear()

    def add_frame(self, frame_features: Optional[np.ndarray], has_hands: bool):
        """
        Add a 192-dim multi-modal feature vector from the current frame.
        If no hands are detected, clears the buffer to prevent evaluating stale poses.
        """
        if not has_hands or frame_features is None:
            if len(self._buffer) > 0:
                self.reset_buffer()
            return

        feat = np.asarray(frame_features, dtype=np.float32).flatten()
        if len(feat) != 192:
            if len(feat) < 192:
                feat = np.pad(feat, (0, 192 - len(feat)), mode="constant")
            else:
                feat = feat[:192]

        self._buffer.append(feat)

    def predict(self) -> Dict[str, Any]:
        """
        Evaluate current 30-frame sequence buffer.

        Returns:
            Dict containing:
                - word: recognized word string or None
                - confidence: float score
                - status: human-readable status banner
                - is_new_sign: True if a fresh, stable, non-cooldown sign was confirmed
        """
        if not self.is_loaded:
            return {
                "word": None,
                "confidence": 0.0,
                "status": "ISL recognition model not trained",
                "is_new_sign": False,
            }

        # Check if enough frames have been collected
        if len(self._buffer) < self.sequence_length:
            return {
                "word": None,
                "confidence": 0.0,
                "status": f"Collecting sign frames ({len(self._buffer)}/{self.sequence_length})",
                "is_new_sign": False,
            }

        # Prepare batch input: shape (1, 30, 192)
        sequence_array = np.array(self._buffer, dtype=np.float32).reshape(1, self.sequence_length, 192)

        try:
            preds = self._model.predict(sequence_array, verbose=0)[0]
            best_idx = int(np.argmax(preds))
            confidence = float(preds[best_idx])
            raw_word = self._labels[best_idx] if best_idx < len(self._labels) else "unknown"

            # Check confidence threshold
            if confidence < self.confidence_threshold:
                self._recent_predictions.clear()
                return {
                    "word": None,
                    "confidence": confidence,
                    "status": "Unknown sign",
                    "is_new_sign": False,
                }

            # Temporal stability smoothing
            self._recent_predictions.append(best_idx)
            if len(self._recent_predictions) < self.stability_count:
                return {
                    "word": None,
                    "confidence": confidence,
                    "status": "Verifying sign stability...",
                    "is_new_sign": False,
                }

            # Check if all recent predictions agree
            all_agree = all(idx == best_idx for idx in self._recent_predictions)
            if not all_agree:
                return {
                    "word": None,
                    "confidence": confidence,
                    "status": "Recognizing sign...",
                    "is_new_sign": False,
                }

            # Cooldown check: prevent emitting the same sign repeatedly while held
            now = time.time()
            if raw_word == self._last_emitted_word and (now - self._last_emitted_time) < self.cooldown_seconds:
                return {
                    "word": raw_word.upper(),
                    "confidence": confidence,
                    "status": f"Sign held: {raw_word.upper()}",
                    "is_new_sign": False,
                }

            # Verified new sign lock
            self._last_emitted_word = raw_word
            self._last_emitted_time = now

            return {
                "word": raw_word.upper(),
                "confidence": confidence,
                "status": "Sign recognized",
                "is_new_sign": True,
            }

        except Exception as exc:
            logger.error("LiveISLRecognizer prediction error: %s", exc)
            return {
                "word": None,
                "confidence": 0.0,
                "status": f"Prediction error: {exc}",
                "is_new_sign": False,
            }

