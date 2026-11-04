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
    Enforces top-1 vs top-2 confidence margin gating to prevent flickering between close signs.
    """

    EXPECTED_CLASSES: List[str] = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        min_confidence: float = 0.50,
        margin_threshold: float = 0.10,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
    ):
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.margin_threshold = margin_threshold
        self.model_path = model_path or (DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else CAT_MODEL_PATH)
        self.labels_path = labels_path or (DEFAULT_LABELS_PATH if DEFAULT_LABELS_PATH.exists() else CAT_LABELS_PATH)

        self.classes: List[str] = list(self.EXPECTED_CLASSES)
        self.id_to_label: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}
        self.label_to_id: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.is_valid_mapping: bool = True
        self._load_labels()

        self.model = None
        self._load_model()

        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._last_letter: Optional[str] = None
        self._last_conf: float = 0.0

    def _load_labels(self):
        """Load alphabet class mappings and strictly validate 26-class A–Z ordered coverage."""
        for p in [self.labels_path, DEFAULT_LABELS_PATH, CAT_LABELS_PATH]:
            if p and Path(p).exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    loaded_classes = data.get("classes") or data.get("labels")
                    if loaded_classes is not None:
                        loaded_str = [str(c).upper() for c in loaded_classes]
                        if loaded_str != self.EXPECTED_CLASSES:
                            logger.error(
                                "ISLFingerspellingRecognizer: Incompatible class order %s in %s; expected exact A–Z order %s",
                                loaded_str, p, self.EXPECTED_CLASSES
                            )
                            self.is_valid_mapping = False
                            return
                        self.classes = loaded_str

                    if "id_to_label" in data:
                        loaded_id_to_label = {int(k): str(v).upper() for k, v in data["id_to_label"].items()}
                        if len(loaded_id_to_label) != 26:
                            logger.error("ISLFingerspellingRecognizer: id_to_label in %s does not contain 26 classes", p)
                            self.is_valid_mapping = False
                            return
                        for expected_idx, expected_char in enumerate(self.EXPECTED_CLASSES):
                            if loaded_id_to_label.get(expected_idx) != expected_char:
                                logger.error(
                                    "ISLFingerspellingRecognizer: id_to_label mapping mismatch in %s at index %d: expected %s, got %s",
                                    p, expected_idx, expected_char, loaded_id_to_label.get(expected_idx)
                                )
                                self.is_valid_mapping = False
                                return
                        self.id_to_label = loaded_id_to_label
                    else:
                        self.id_to_label = {i: c for i, c in enumerate(self.classes)}

                    if "label_to_id" in data:
                        loaded_label_to_id = {str(k).upper(): int(v) for k, v in data["label_to_id"].items()}
                        for expected_idx, expected_char in enumerate(self.EXPECTED_CLASSES):
                            if loaded_label_to_id.get(expected_char) != expected_idx:
                                logger.error(
                                    "ISLFingerspellingRecognizer: label_to_id mapping mismatch in %s for %s: expected %d, got %s",
                                    p, expected_char, expected_idx, loaded_label_to_id.get(expected_char)
                                )
                                self.is_valid_mapping = False
                                return
                        self.label_to_id = loaded_label_to_id
                    else:
                        self.label_to_id = {v: k for k, v in self.id_to_label.items()}

                    self.is_valid_mapping = True
                    logger.info("ISLFingerspellingRecognizer: Loaded 26 strictly verified classes (A=0..Z=25) from %s", p)
                    return
                except Exception as e:
                    logger.warning("ISLFingerspellingRecognizer: Failed loading labels from %s: %s", p, e)
                    self.is_valid_mapping = False

    def _load_model(self):
        """Load trained Keras fingerspelling model and verify schema compatibility."""
        import tensorflow as tf
        for p in [self.model_path, DEFAULT_MODEL_PATH, CAT_MODEL_PATH]:
            if p and Path(p).exists():
                try:
                    loaded = tf.keras.models.load_model(str(p), compile=False)
                    verify_model_compatibility(loaded, expected_dim=FEATURE_DIM, expected_seq_len=self.sequence_length)
                    if hasattr(loaded, "output_shape") and loaded.output_shape is not None:
                        out_dim = loaded.output_shape[-1]
                        if out_dim != 26:
                            raise IncompatibleModelError(f"Alphabet model output shape {loaded.output_shape} != 26 classes.")
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
        if not self.is_loaded or not self.is_valid_mapping:
            return {
                "letter": None,
                "label": "Model Not Available" if not self.is_loaded else "Invalid Mapping",
                "confidence": 0.0,
                "status": "Model Not Available" if not self.is_loaded else "Inconsistent Class Mapping",
                "is_model_available": False,
            }

        if len(self._buffer) < 10:
            return {
                "letter": None,
                "label": "Collecting",
                "confidence": 0.0,
                "status": f"Collecting letter frames ({len(self._buffer)}/{self.sequence_length})",
                "is_model_available": True,
            }

        # Interpolate / resample buffer to target SEQUENCE_LENGTH using shared schema
        seq = resample_sequence(np.asarray(self._buffer, dtype=np.float32), target_length=self.sequence_length)
        batch = np.expand_dims(seq, axis=0)  # Shape (1, 30, 282)

        try:
            probs = self.model.predict(batch, verbose=0)[0]
            if len(probs) != len(self.classes):
                logger.error("Output dimension mismatch: %d != %d", len(probs), len(self.classes))
                return {
                    "letter": None,
                    "word": None,
                    "label": "Error",
                    "confidence": 0.0,
                    "status": "Model Dimension Mismatch",
                    "is_new_sign": False,
                    "is_model_available": True,
                }

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            top_conf = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = top_conf - second_conf

            top_5 = [
                {
                    "index": int(i),
                    "letter": self.id_to_label.get(int(i), "?"),
                    "probability": float(probs[int(i)]),
                }
                for i in sorted_indices[:5]
            ]
            pred_letter = self.id_to_label.get(top_idx)

            logger.info(
                "ISLFingerspellingRecognizer predict: winning_index=%d, resolved_letter=%s, "
                "confidence=%.4f (min_conf=%.4f), runner_up_conf=%.4f, margin=%.4f (min_margin=%.4f), "
                "top_5=%s",
                top_idx,
                pred_letter,
                top_conf,
                self.min_confidence,
                second_conf,
                margin,
                self.margin_threshold,
                [(item["letter"], f"{item['probability']:.3f}") for item in top_5],
            )

            # Gating: must exceed minimum confidence AND separate from runner-up by margin
            if top_conf < self.min_confidence or margin < self.margin_threshold:
                return {
                    "letter": None,
                    "label": "Unknown",
                    "confidence": top_conf,
                    "status": "Unknown Sign",
                    "is_model_available": True,
                    "top_5": top_5,
                    "winning_index": top_idx,
                    "resolved_letter": pred_letter,
                    "margin": margin,
                    "confidence_threshold": self.min_confidence,
                    "margin_threshold": self.margin_threshold,
                }

            if pred_letter is None or pred_letter not in self.label_to_id:
                logger.error("Top index %d not found in valid id_to_label mapping", top_idx)
                return {
                    "letter": None,
                    "label": "Invalid Mapping",
                    "confidence": 0.0,
                    "status": "Inconsistent Class Mapping",
                    "is_model_available": True,
                    "top_5": top_5,
                    "winning_index": top_idx,
                    "resolved_letter": pred_letter,
                    "margin": margin,
                    "confidence_threshold": self.min_confidence,
                    "margin_threshold": self.margin_threshold,
                }
            self._last_letter = pred_letter
            self._last_conf = top_conf

            return {
                "letter": pred_letter,
                "label": pred_letter,
                "confidence": top_conf,
                "status": f"Letter {pred_letter} ({top_conf:.1%})",
                "is_model_available": True,
                "top_5": top_5,
                "winning_index": top_idx,
                "resolved_letter": pred_letter,
                "margin": margin,
                "confidence_threshold": self.min_confidence,
                "margin_threshold": self.margin_threshold,
            }
        except Exception as e:
            logger.error("ISLFingerspellingRecognizer: Inference error: %s", e)
            return {
                "letter": None,
                "label": "Error",
                "confidence": 0.0,
                "status": f"Inference Error: {e}",
                "is_model_available": True,
            }
