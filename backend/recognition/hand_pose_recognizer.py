"""
SignBridge AI — Hand Pose Based ISL Recognizer.
Performs real-time Indian Sign Language recognition using hand pose features only:
- Dual Recognition Paths:
  1. Dynamic Hand Movement Recognizer (Bidirectional LSTM on 30-frame hand pose sequence).
  2. Static Hand Pose Classifier (Dense MLP on single-frame handshape, angles & distances).
- Strictly Hand-Only: ZERO face, body pose, shoulder, or torso tracking.
- Features:
  - 216-dimensional unified two-hand representation.
  - Motion energy estimation to distinguish static handshapes from dynamic movement.
  - Sliding sequence buffer with safe missing-hand handling.
  - Confidence gating, temporal voting, and repeat sign cooldown.
  - Unknown sign rejection.
"""

from collections import deque
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LABELS_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "hand_pose" / "labels.json"
DYNAMIC_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "hand_pose" / "isl_dynamic_handpose.keras"
STATIC_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "hand_pose" / "isl_static_handpose.keras"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 216


class HandPoseRecognizer:
    """
    Real-time Hand Pose ISL Recognizer.
    Maintains a sliding 30-frame feature buffer and evaluates dynamic & static models.
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        confidence_threshold: float = 0.35,
        cooldown_seconds: float = 1.5,
        labels_path: Optional[Path] = None,
        dynamic_model_path: Optional[Path] = None,
        static_model_path: Optional[Path] = None,
    ):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds

        self.labels_path = labels_path or DEFAULT_LABELS_PATH
        self.dynamic_model_path = dynamic_model_path or DYNAMIC_MODEL_PATH
        self.static_model_path = static_model_path or STATIC_MODEL_PATH

        self.classes: List[str] = []
        self.id_to_label: Dict[int, str] = {}
        self.label_to_id: Dict[str, int] = {}
        self._load_labels()

        self.dynamic_model = None
        self.static_model = None
        self._load_models()

        # Sliding sequence buffer
        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._hand_present_buffer: deque = deque(maxlen=self.sequence_length)

        # Recognition state
        self._last_predicted_word: Optional[str] = None
        self._last_prediction_time: float = 0.0
        self._last_emitted_word: Optional[str] = None
        self._recent_predictions: deque = deque(maxlen=5)

    def _load_labels(self):
        """Load 40-word ISL classes from labels.json."""
        if self.labels_path.exists():
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.classes = data.get("classes", [])
                self.label_to_id = data.get("label_to_id", {})
                self.id_to_label = {int(k): v for k, v in data.get("id_to_label", {}).items()}
                logger.info("HandPoseRecognizer: Loaded %d ISL classes from %s", len(self.classes), self.labels_path)
                return
            except Exception as e:
                logger.error("HandPoseRecognizer: Failed loading labels from %s: %s", self.labels_path, e)

        # Fallback to default 40 ISL vocabulary classes
        self.classes = [
            "brother", "come", "drink", "eat", "father", "food", "friend", "go",
            "goodbye", "he", "hello", "help", "home", "hospital", "market", "me",
            "mother", "no", "okay", "please", "read", "school", "she", "sister",
            "sit", "sorry", "stand", "stop", "student", "tea", "teacher", "thank_you",
            "today", "water", "what", "when", "where", "write", "yes", "you"
        ]
        self.id_to_label = {i: c for i, c in enumerate(self.classes)}
        self.label_to_id = {c: i for i, c in enumerate(self.classes)}

    def _load_models(self):
        """Load trained dynamic and static hand pose models."""
        import tensorflow as tf

        if self.dynamic_model_path.exists():
            try:
                self.dynamic_model = tf.keras.models.load_model(str(self.dynamic_model_path), compile=False)
                logger.info("HandPoseRecognizer: Dynamic model loaded from %s", self.dynamic_model_path)
            except Exception as e:
                logger.warning("HandPoseRecognizer: Failed to load dynamic model (%s)", e)

        if self.static_model_path.exists():
            try:
                self.static_model = tf.keras.models.load_model(str(self.static_model_path), compile=False)
                logger.info("HandPoseRecognizer: Static model loaded from %s", self.static_model_path)
            except Exception as e:
                logger.warning("HandPoseRecognizer: Failed to load static model (%s)", e)

    def reset_buffer(self):
        """Clear sequence buffer and cooldown history."""
        self._buffer.clear()
        self._hand_present_buffer.clear()
        self._last_predicted_word = None
        self._recent_predictions.clear()

    @property
    def is_loaded(self) -> bool:
        return (self.dynamic_model is not None) or (self.static_model is not None)

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def add_frame(self, feature_vector: Optional[np.ndarray], has_hands: bool):
        """
        Append a frame's 216-dim hand pose feature vector into sliding buffer.

        Args:
            feature_vector: shape (216,) or None
            has_hands: True if at least one hand is detected in current frame
        """
        if not has_hands or feature_vector is None:
            # If no hand in view, reset buffer to prevent mixing separate gestures
            self._buffer.clear()
            self._hand_present_buffer.clear()
            return

        if len(feature_vector) != FEATURE_DIM:
            vec = np.zeros(FEATURE_DIM, dtype=np.float32)
            n = min(len(feature_vector), FEATURE_DIM)
            vec[:n] = feature_vector[:n]
            feature_vector = vec

        self._buffer.append(feature_vector)
        self._hand_present_buffer.append(True)

    def calculate_motion_energy(self) -> float:
        """
        Compute standard deviation of wrist and fingertip movements over buffer.
        Helps differentiate stationary static handshapes from active dynamic movements.
        """
        if len(self._buffer) < 5:
            return 0.0

        recent = np.array(list(self._buffer)[-15:])  # (T, 216)
        # Wrist coordinates are at indices 98:101 (left) and 202:205 (right)
        lw_diff = np.diff(recent[:, 98:101], axis=0)
        rw_diff = np.diff(recent[:, 202:205], axis=0)
        motion = float(np.mean(np.linalg.norm(lw_diff, axis=-1)) + np.mean(np.linalg.norm(rw_diff, axis=-1)))
        return motion

    def predict(self) -> Dict[str, Any]:
        """
        Perform inference on the current sliding buffer.

        Returns:
            Dictionary with:
              - word: predicted ISL word string (or None)
              - confidence: float score
              - is_new_sign: bool (True if ready to append to sentence)
              - mode: 'dynamic' | 'static' | 'none'
              - status: human-readable HUD message
        """
        now = time.time()

        # Check if enough frames have accumulated
        if len(self._buffer) < self.sequence_length:
            return {
                "word": None,
                "confidence": 0.0,
                "is_new_sign": False,
                "mode": "none",
                "status": f"Collecting sign frames... ({len(self._buffer)}/{self.sequence_length})",
            }

        seq_arr = np.array(self._buffer, dtype=np.float32)  # (30, 216)
        motion = self.calculate_motion_energy()

        pred_word = None
        pred_conf = 0.0
        mode = "none"

        # Route 1: Dynamic Sequence Model (Primary for ISL gestures)
        if self.dynamic_model is not None:
            input_batch = np.expand_dims(seq_arr, axis=0)  # (1, 30, 216)
            probs = self.dynamic_model.predict(input_batch, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            pred_word = self.id_to_label.get(top_idx, "Unknown")
            pred_conf = top_conf
            mode = "dynamic"

        # Route 2: Static Model fallback if motion is very low and static model is loaded
        if (self.static_model is not None) and (motion < 0.008 or pred_conf < self.confidence_threshold):
            latest_feat = np.expand_dims(seq_arr[-1], axis=0)  # (1, 216)
            stat_probs = self.static_model.predict(latest_feat, verbose=0)[0]
            stat_idx = int(np.argmax(stat_probs))
            stat_conf = float(stat_probs[stat_idx])

            if stat_conf > pred_conf:
                pred_word = self.id_to_label.get(stat_idx, "Unknown")
                pred_conf = stat_conf
                mode = "static"

        # Confidence gating
        if pred_conf < self.confidence_threshold:
            return {
                "word": None,
                "confidence": pred_conf,
                "is_new_sign": False,
                "mode": mode,
                "status": "Unknown sign (Try clear ISL gesture)",
            }

        # Temporal smoothing: check agreement across recent predictions
        self._recent_predictions.append(pred_word)
        # Determine dominant prediction in window
        counts = {}
        for w in self._recent_predictions:
            counts[w] = counts.get(w, 0) + 1
        most_common_word, count = max(counts.items(), key=lambda x: x[1])

        # Require at least 2 consecutive / consistent frames
        if count < 2:
            return {
                "word": pred_word,
                "confidence": pred_conf,
                "is_new_sign": False,
                "mode": mode,
                "status": f"Stabilizing: {pred_word}",
            }

        # Check repeat cooldown
        is_repeat = (most_common_word == self._last_predicted_word)
        within_cooldown = (now - self._last_prediction_time) < self.cooldown_seconds

        if is_repeat and within_cooldown:
            return {
                "word": most_common_word,
                "confidence": pred_conf,
                "is_new_sign": False,
                "mode": mode,
                "status": f"Sign held: {most_common_word} (Cooldown active)",
            }

        # Verified new sign!
        self._last_predicted_word = most_common_word
        self._last_prediction_time = now
        self._last_emitted_word = most_common_word

        return {
            "word": most_common_word,
            "confidence": pred_conf,
            "is_new_sign": True,
            "mode": mode,
            "status": f"Recognized ({mode}): {most_common_word}",
        }
