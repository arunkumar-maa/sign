"""
SignBridge AI — Phase 4: ISL 40-Word Gesture Recognizer.
Dedicated recognizer for the 40 Indian Sign Language vocabulary signs.
Completely decoupled from finger counting and number recognition.

Key Fixes & Guards:
1. Motion Energy Gating: Rejects dynamic word predictions (e.g. HELLO, GO, SORRY, STUDENT)
   when the user is holding a stationary hand pose (such as 2 fingers).
2. Margin & Entropy Gating: If top-1 confidence is below threshold (< 0.52) or margin
   between top-1 and top-2 is < 0.15, classifies as 'Unknown Sign'.
3. Temporal smoothing over sliding 30-frame window.
4. Duplicate cooldown: Prevents repeating the same word into the sentence repeatedly.
"""

from collections import deque
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WORD_MODEL = PROJECT_ROOT / "models" / "isl_word_model.keras"
CAT_WORD_MODEL = PROJECT_ROOT / "models" / "word" / "hand_sequence_lstm.keras"
HAND_POSE_DYNAMIC_MODEL = PROJECT_ROOT / "models" / "hand_pose" / "isl_dynamic_handpose.keras"

DEFAULT_WORD_LABELS = PROJECT_ROOT / "models" / "word_labels.json"
CAT_WORD_LABELS = PROJECT_ROOT / "models" / "word" / "labels.json"
HAND_POSE_LABELS = PROJECT_ROOT / "models" / "hand_pose" / "labels.json"

STATIC_MODEL_PATH = PROJECT_ROOT / "models" / "hand_pose" / "isl_static_handpose.keras"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 282


class ISLWordRecognizer:
    """
    Dedicated 40-Word ISL Classifier.
    Evaluates dynamic sequence gesture features with strict motion and confidence gating.
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        min_confidence: float = 0.52,
        min_margin: float = 0.15,
        min_motion_energy: float = 0.012,
        cooldown_seconds: float = 1.8,
        labels_path: Optional[Path] = None,
        dynamic_model_path: Optional[Path] = None,
        static_model_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.sequence_length = sequence_length
        self.min_confidence = confidence_threshold if confidence_threshold is not None else min_confidence
        self.confidence_threshold = self.min_confidence
        self.min_margin = min_margin
        self.min_motion_energy = min_motion_energy
        self.cooldown_seconds = cooldown_seconds

        self.labels_path = labels_path
        self.dynamic_model_path = model_path or dynamic_model_path
        self.static_model_path = static_model_path or STATIC_MODEL_PATH
        self.feature_dim = FEATURE_DIM

        self.classes: List[str] = []
        self.id_to_label: Dict[int, str] = {}
        self.label_to_id: Dict[str, int] = {}
        self._load_labels()

        self.dynamic_model = None
        self.static_model = None
        self._load_models()

        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._recent_predictions: deque = deque(maxlen=5)

        self._last_predicted_word: Optional[str] = None
        self._last_prediction_time: float = 0.0
        self._last_emitted_word: Optional[str] = None

    def _load_labels(self):
        """Load 40 ISL classes from labels.json or word_labels.json."""
        candidates = [self.labels_path, DEFAULT_WORD_LABELS, CAT_WORD_LABELS, HAND_POSE_LABELS]
        for p in candidates:
            if p and p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self.classes = data
                    else:
                        self.classes = data.get("classes") or data.get("labels") or []
                    self.id_to_label = {int(k): v for k, v in data.get("id_to_label", {}).items()} if isinstance(data, dict) and "id_to_label" in data else {i: c for i, c in enumerate(self.classes)}
                    self.label_to_id = data.get("label_to_id", {c: i for i, c in enumerate(self.classes)}) if isinstance(data, dict) else {c: i for i, c in enumerate(self.classes)}
                    self._labels = self.classes
                    logger.info("ISLWordRecognizer: Loaded %d classes from %s", len(self.classes), p)
                    return
                except Exception as e:
                    logger.error("ISLWordRecognizer: Failed loading labels from %s: %s", p, e)

        # Fallback to standard 40 vocabulary classes
        self.classes = [
            "brother", "come", "drink", "eat", "father", "food", "friend", "go",
            "goodbye", "he", "hello", "help", "home", "hospital", "market", "me",
            "mother", "no", "okay", "please", "read", "school", "she", "sister",
            "sit", "sorry", "stand", "stop", "student", "tea", "teacher", "thank_you",
            "today", "water", "what", "when", "where", "write", "yes", "you"
        ]
        self.id_to_label = {i: c for i, c in enumerate(self.classes)}
        self.label_to_id = {c: i for i, c in enumerate(self.classes)}
        self._labels = self.classes

    def reset(self):
        """Reset buffer and prediction state."""
        self.reset_buffer()

    def _load_models(self):
        """Load trained dynamic BiLSTM and static MLP models."""
        import tensorflow as tf
        candidates = [self.dynamic_model_path, DEFAULT_WORD_MODEL, CAT_WORD_MODEL, HAND_POSE_DYNAMIC_MODEL]
        for p in candidates:
            if p and p.exists():
                try:
                    self.dynamic_model = tf.keras.models.load_model(str(p), compile=False)
                    if hasattr(self.dynamic_model, "input_shape") and self.dynamic_model.input_shape:
                        self.feature_dim = self.dynamic_model.input_shape[-1]
                    logger.info("ISLWordRecognizer: Loaded dynamic model from %s (feature_dim=%d)", p, self.feature_dim)
                    break
                except Exception as e:
                    logger.warning("ISLWordRecognizer: Failed to load dynamic model from %s: %s", p, e)

        if self.static_model_path and self.static_model_path.exists():
            try:
                self.static_model = tf.keras.models.load_model(str(self.static_model_path), compile=False)
                logger.info("ISLWordRecognizer: Loaded static model from %s", self.static_model_path)
            except Exception as e:
                logger.warning("ISLWordRecognizer: Failed to load static model: %s", e)

    @property
    def is_loaded(self) -> bool:
        return (self.dynamic_model is not None) or (self.static_model is not None)

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def reset_buffer(self):
        """Clear sequence buffer and recognition state."""
        self._buffer.clear()
        self._recent_predictions.clear()
        self._last_predicted_word = None
        self._last_prediction_time = 0.0
        self._last_emitted_word = None

    def add_frame(self, feature_vector: Optional[np.ndarray], has_hands: bool):
        """Feed a hand pose feature vector into sliding window."""
        if not has_hands or feature_vector is None:
            self._buffer.clear()
            return

        if len(feature_vector) != self.feature_dim:
            vec = np.zeros(self.feature_dim, dtype=np.float32)
            n = min(len(feature_vector), self.feature_dim)
            vec[:n] = feature_vector[:n]
            feature_vector = vec

        self._buffer.append(feature_vector)

    def calculate_motion_energy(self) -> float:
        """Calculate movement velocity of hand landmarks over buffer."""
        if len(self._buffer) < 5:
            return 0.0
        recent = np.array(list(self._buffer)[-15:])  # (T, D)
        overall_diff = np.diff(recent, axis=0)
        return float(np.mean(np.linalg.norm(overall_diff, axis=-1)))

    def predict(self) -> Dict[str, Any]:
        """
        Evaluate current buffer with motion gating, confidence thresholding,
        and margin verification.
        """
        if not self.is_loaded:
            return {
                "word": None,
                "confidence": 0.0,
                "is_new_sign": False,
                "mode": "none",
                "status": "Model Not Available",
                "status_detail": "no_model",
                "is_model_available": False,
            }

        now = time.time()

        if len(self._buffer) < self.sequence_length:
            return {
                "word": None,
                "confidence": 0.0,
                "is_new_sign": False,
                "mode": "none",
                "status": f"Collecting sign frames... ({len(self._buffer)}/{self.sequence_length})",
                "status_detail": "buffering",
            }

        seq_arr = np.array(self._buffer, dtype=np.float32)
        motion = self.calculate_motion_energy()

        # GUARD 1: Motion energy check
        # Dynamic gestures MUST have active motion. If hand is stationary (like holding 2 fingers),
        # reject dynamic word prediction!
        if motion < self.min_motion_energy:
            return {
                "word": None,
                "confidence": 0.0,
                "is_new_sign": False,
                "mode": "stationary",
                "status": "Unknown sign",
                "status_detail": "low_confidence",
                "reason": "Stationary hand — perform ISL sign motion",
            }

        if self.dynamic_model is None:
            return {
                "word": None,
                "confidence": 0.0,
                "is_new_sign": False,
                "mode": "none",
                "status": "Model Not Available",
                "status_detail": "no_model",
                "is_model_available": False,
            }

        # Adapt input feature dimension to match model's expected input dimension (e.g. 192 or 216)
        expected_dim = getattr(self.dynamic_model, "input_shape", (None, 30, FEATURE_DIM))[-1] or FEATURE_DIM
        if seq_arr.shape[-1] > expected_dim:
            adapted_seq = seq_arr[:, :expected_dim]
        elif seq_arr.shape[-1] < expected_dim:
            pad_width = expected_dim - seq_arr.shape[-1]
            adapted_seq = np.pad(seq_arr, ((0, 0), (0, pad_width)), mode="constant")
        else:
            adapted_seq = seq_arr

        input_batch = np.expand_dims(adapted_seq, axis=0)
        probs = self.dynamic_model.predict(input_batch, verbose=0)[0]

        top_indices = np.argsort(probs)[::-1]
        top1_idx = int(top_indices[0])
        top2_idx = int(top_indices[1])

        top1_conf = float(probs[top1_idx])
        top2_conf = float(probs[top2_idx])
        margin = top1_conf - top2_conf

        pred_word = self.id_to_label.get(top1_idx, "Unknown")

        # GUARD 2: Confidence threshold & Top-1 vs Top-2 margin gating
        if top1_conf < self.min_confidence or margin < self.min_margin:
            return {
                "word": None,
                "confidence": top1_conf,
                "is_new_sign": False,
                "mode": "dynamic",
                "status": "Unknown sign",
                "status_detail": "low_confidence",
                "reason": f"Low confidence: {top1_conf:.1%}, margin: {margin:.2f}",
            }

        # GUARD 3: Temporal smoothing (require consistency over 2 consecutive frames)
        self._recent_predictions.append(pred_word)
        counts = {}
        for w in self._recent_predictions:
            counts[w] = counts.get(w, 0) + 1
        most_common_word, freq = max(counts.items(), key=lambda x: x[1])

        if freq < 2:
            return {
                "word": pred_word,
                "confidence": top1_conf,
                "is_new_sign": False,
                "mode": "dynamic",
                "status": f"Stabilizing gesture: {pred_word}",
            }

        # GUARD 4: Duplicate sign cooldown
        is_repeat = (most_common_word == self._last_predicted_word)
        within_cooldown = (now - self._last_prediction_time) < self.cooldown_seconds

        if is_repeat and within_cooldown:
            return {
                "word": most_common_word,
                "confidence": top1_conf,
                "is_new_sign": False,
                "mode": "dynamic",
                "status": f"Sign held: {most_common_word} (Cooldown active)",
            }

        # Confirmed new sign!
        self._last_predicted_word = most_common_word
        self._last_prediction_time = now
        self._last_emitted_word = most_common_word

        return {
            "word": most_common_word,
            "confidence": top1_conf,
            "is_new_sign": True,
            "mode": "dynamic",
            "status": f"Recognized Word: {most_common_word.upper()}",
        }
