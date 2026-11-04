"""
SignBridge AI — Comprehensive 15-Area Verification Suite.
Validates the complete Alphabet (A–Z) & Numbers (0–9) real-time recognition architecture.

Test Matrix:
1. Alphabet label mapping (26 classes A–Z).
2. Number label mapping (10 classes 0–9).
3. Shared feature extraction (282-dim, 140 per hand, 2 presence flags).
4. Training/inference feature compatibility (identical numerical values).
5. Missing hand detection (graceful zero-state, presence 0.0).
6. Invalid sample rejection (NaN, Inf, wrong dimensions).
7. Missing model files (graceful "Model Not Available" status).
8. Incompatible feature schema (rejection via IncompatibleModelError).
9. Low-confidence predictions (genuine Unknown rejection).
10. Temporal smoothing (30-frame interpolation).
11. Character insertion (assembling "HELLO123").
12. Backspace and clear operations.
13. Dataset class validation (strict label boundaries).
14. Model loading (Keras BiLSTM models loaded and verified).
15. Camera pipeline integration (single stream, green dots, blue connections).
"""

import json
from pathlib import Path
import sys
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision.feature_schema import (
    FEATURE_DIM,
    SINGLE_HAND_DIM,
    SEQUENCE_LENGTH,
    SCHEMA_VERSION,
    IncompatibleModelError,
    extract_frame_features,
    extract_features_from_arrays,
    extract_single_hand_features,
    resample_sequence,
    validate_features,
    validate_sequence,
    verify_model_compatibility,
)
from backend.recognition.isl_fingerspelling_recognizer import ISLFingerspellingRecognizer
from backend.recognition.isl_number_recognizer import ISLNumberRecognizer
from backend.recognition.character_builder import CharacterTextBuilder
from backend.vision.live_pipeline import LivePipeline
from backend.vision.hand_landmarks import HandLandmarkResult
from isl_data_collection.storage import validate_label


# ── Area 1: Alphabet label mapping ───────────────────────────────────────────
def test_1_alphabet_label_mapping():
    rec = ISLFingerspellingRecognizer()
    assert len(rec.classes) == 26
    expected_letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    assert rec.classes == expected_letters
    for i, char in enumerate(expected_letters):
        assert rec.id_to_label[i] == char
        assert rec.label_to_id[char] == i


# ── Area 2: Number label mapping ─────────────────────────────────────────────
def test_2_number_label_mapping():
    rec = ISLNumberRecognizer()
    assert len(rec.classes) == 10
    expected_digits = [str(i) for i in range(10)]
    assert rec.classes == expected_digits
    for i, digit in enumerate(expected_digits):
        assert str(rec.id_to_label[i]) == digit
        assert rec.label_to_id[digit] == i


# ── Area 3: Shared feature extraction ────────────────────────────────────────
def test_3_shared_feature_extraction():
    assert FEATURE_DIM == 282
    assert SINGLE_HAND_DIM == 140
    mock_pts = np.random.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    single_feat = extract_single_hand_features(mock_pts)
    assert single_feat.shape == (140,)
    assert np.all(np.isfinite(single_feat))

    # Two hands
    full_feat, count = extract_features_from_arrays(mock_pts, mock_pts)
    assert full_feat.shape == (282,)
    assert count == 2
    assert full_feat[-2] == 1.0
    assert full_feat[-1] == 1.0


# ── Area 4: Training/inference feature compatibility ────────────────────────
def test_4_training_inference_feature_compatibility():
    rng = np.random.RandomState(123)
    left = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    right = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)

    feat_training, _ = extract_features_from_arrays(left, right)

    class DummyLandmarkResult:
        left_hand_detected = True
        right_hand_detected = True
        left_landmarks = left
        right_landmarks = right
        raw_results = None

    feat_inference, _ = extract_frame_features(DummyLandmarkResult())
    assert np.allclose(feat_training, feat_inference, atol=1e-6)


# ── Area 5: Missing hand detection ───────────────────────────────────────────
def test_5_missing_hand_detection():
    # Neither hand present
    feat_empty, count_empty = extract_features_from_arrays(None, None)
    assert feat_empty.shape == (282,)
    assert count_empty == 0
    assert np.all(feat_empty == 0.0)

    # Frame extraction with None
    feat_none, count_none = extract_frame_features(None)
    assert feat_none.shape == (282,)
    assert count_none == 0
    assert np.all(feat_none == 0.0)


# ── Area 6: Invalid sample rejection ─────────────────────────────────────────
def test_6_invalid_sample_rejection():
    # Wrong dimension
    assert validate_features(np.zeros(192, dtype=np.float32)) is False
    assert validate_features(np.zeros(216, dtype=np.float32)) is False

    # NaN / Inf
    arr = np.zeros(282, dtype=np.float32)
    arr[5] = np.nan
    assert validate_features(arr) is False

    arr[5] = np.inf
    assert validate_features(arr) is False


# ── Area 7: Missing model files ──────────────────────────────────────────────
def test_7_missing_model_files(tmp_path):
    missing_path = tmp_path / "nonexistent_model.keras"
    rec_fs = ISLFingerspellingRecognizer(model_path=missing_path)
    # If fallback found, test predict behavior when model is not available
    rec_fs.model = None
    pred = rec_fs.predict()
    assert pred["letter"] is None
    assert pred["is_model_available"] is False
    assert "Model Not Available" in pred["status"]

    rec_num = ISLNumberRecognizer(model_path=missing_path)
    rec_num.trained_model = None
    pred_num = rec_num.predict()
    assert pred_num["number"] is None
    assert pred_num["is_model_available"] is False
    assert "Model Not Available" in pred_num["status"]


# ── Area 8: Incompatible feature schema rejection ────────────────────────────
def test_8_incompatible_feature_schema():
    class MockIncompatibleModel:
        input_shape = (None, 30, 192)  # legacy 192 dim

    with pytest.raises(IncompatibleModelError):
        verify_model_compatibility(MockIncompatibleModel(), expected_dim=282, expected_seq_len=30)


# ── Area 9: Low-confidence predictions ───────────────────────────────────────
def test_9_low_confidence_predictions():
    rec = ISLFingerspellingRecognizer(min_confidence=0.80)
    # Populate buffer with 15 frames
    for _ in range(15):
        rec.add_frame(np.zeros(282, dtype=np.float32), has_hands=True)

    # Mock model to return low confidence (0.30 max probability)
    class LowConfModel:
        def predict(self, batch, verbose=0):
            probs = np.full((1, 26), 1.0 / 26.0, dtype=np.float32)
            probs[0, 5] = 0.30
            return probs

    rec.model = LowConfModel()
    res = rec.predict()
    assert res["letter"] is None
    assert res["label"] == "Unknown"
    assert "Unknown Sign" in res["status"]


# ── Area 10: Temporal smoothing ──────────────────────────────────────────────
def test_10_temporal_smoothing():
    short_seq = np.ones((12, 282), dtype=np.float32)
    resampled = resample_sequence(short_seq, target_length=30)
    assert resampled.shape == (30, 282)
    assert np.allclose(resampled, 1.0)


# ── Area 11: Character insertion ─────────────────────────────────────────────
def test_11_character_insertion():
    builder = CharacterTextBuilder()
    builder.add_character("H")
    builder.add_character("E")
    builder.add_character("L")
    builder.add_character("L")
    builder.add_character("O")
    builder.add_character("1")
    builder.add_character("2")
    builder.add_character("3")
    assert builder.text == "HELLO123"


# ── Area 12: Backspace and clear operations ──────────────────────────────────
def test_12_backspace_and_clear_operations():
    builder = CharacterTextBuilder()
    builder.set_text("TEST1")
    assert builder.backspace() == "TEST"
    assert builder.backspace() == "TES"
    assert builder.clear() == ""
    assert builder.text == ""


# ── Area 13: Dataset class validation ────────────────────────────────────────
def test_13_dataset_class_validation():
    # Valid alphabet
    assert validate_label("alphabet", "a") == ("alphabet", "A")
    assert validate_label("alphabet", "Z") == ("alphabet", "Z")
    with pytest.raises(ValueError):
        validate_label("alphabet", "AB")
    with pytest.raises(ValueError):
        validate_label("alphabet", "1")

    # Valid numbers (all 10 classes)
    for i in range(10):
        assert validate_label("number", str(i)) == ("number", str(i))
    assert validate_label("number", "TWO") == ("number", "2")
    with pytest.raises(ValueError):
        validate_label("number", "10")
    with pytest.raises(ValueError):
        validate_label("number", "INVALID")


# ── Area 14: Model loading ───────────────────────────────────────────────────
@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "isl_fingerspelling_model.keras").exists() or
    not (PROJECT_ROOT / "models" / "isl_number_model.keras").exists(),
    reason="Real model weights are gitignored; run setup_assets.py or training scripts first",
)
def test_14_model_loading():
    fs_rec = ISLFingerspellingRecognizer()
    assert fs_rec.is_loaded is True
    assert fs_rec.model.input_shape == (None, 30, 282)
    assert fs_rec.model.output_shape == (None, 26)

    num_rec = ISLNumberRecognizer()
    assert num_rec.is_loaded is True
    assert num_rec.trained_model.input_shape == (None, 30, 282)
    assert num_rec.trained_model.output_shape == (None, 10)


# ── Area 15: Camera pipeline integration ─────────────────────────────────────
def test_15_camera_pipeline_integration():
    pipeline = LivePipeline(mode="Alphabet (A–Z)")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_lms = np.linspace(0.2, 0.8, 63).reshape(21, 3).astype(np.float32)

    # Mock hand detection result
    h_res = HandLandmarkResult(
        detected=True,
        num_hands=1,
        right_hand_detected=True,
        right_landmarks=dummy_lms,
        all_hands=[("Right", dummy_lms, 0.95)],
    )
    pipeline.hand_detector.detect = lambda f, is_bgr=False: h_res

    res = pipeline.process_frame(frame, mode="Alphabet (A–Z)")
    assert res["annotated_frame"] is not None
    assert res["annotated_frame"].shape == (480, 640, 3)

    # Check green dots (0, 255, 0) and blue lines (0, 0, 255 in RGB) are drawn
    ann = res["annotated_frame"]
    has_green = np.any((ann[:, :, 0] == 0) & (ann[:, :, 1] == 255) & (ann[:, :, 2] == 0))
    has_blue = np.any((ann[:, :, 0] == 0) & (ann[:, :, 1] == 0) & (ann[:, :, 2] == 255))
    assert has_green, "Green landmark dots not rendered"
    assert has_blue, "Blue finger connection lines not rendered"

    # Number mode
    res_num = pipeline.process_frame(frame, mode="Numbers (0–9)")
    assert res_num["annotated_frame"] is not None

