"""
Tests for backend/vision/feature_schema.py (Schema v2.0.0).
Verifies:
- Canonical 282 feature dimension
- Single hand 140 feature dimension
- Left and right presence flags
- Missing hands handling
- Input validation (NaN, Inf, wrong shapes)
- Temporal resampling (to 30 frames)
- Rejection of incompatible models via IncompatibleModelError
- Training and inference feature parity
"""

import sys
from pathlib import Path
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
    compute_joint_angle,
    extract_single_hand_features,
    extract_features_from_arrays,
    extract_frame_features,
    resample_sequence,
    validate_features,
    validate_sequence,
    verify_model_compatibility,
)


def test_schema_constants():
    assert SCHEMA_VERSION == "2.0.0"
    assert FEATURE_DIM == 282
    assert SINGLE_HAND_DIM == 140
    assert SEQUENCE_LENGTH == 30


def test_joint_angle_computation():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    angle_90 = compute_joint_angle(a, b, c)
    # 90 degrees / 180 degrees = 0.5
    assert np.isclose(angle_90, 0.5, atol=1e-3)

    # Collinear (180 deg)
    c_180 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert np.isclose(compute_joint_angle(a, b, c_180), 1.0, atol=1e-3)

    # Degenerate zero distance
    assert compute_joint_angle(b, b, c) == 0.0


def test_single_hand_extraction():
    dummy_hand = np.zeros((21, 3), dtype=np.float32)
    # Set middle MCP (index 9) to non-zero distance from wrist (index 0)
    dummy_hand[9] = [0.0, 0.2, 0.0]
    dummy_hand[4] = [0.1, 0.0, 0.0]  # thumb tip
    feat = extract_single_hand_features(dummy_hand)

    assert feat.shape == (SINGLE_HAND_DIM,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))


def test_two_hand_extraction_and_presence():
    left_hand = np.zeros((21, 3), dtype=np.float32)
    left_hand[9] = [0.0, 0.2, 0.0]
    right_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand[9] = [0.0, 0.2, 0.0]

    # Both hands present
    feat, count = extract_features_from_arrays(left_hand, right_hand)
    assert feat.shape == (FEATURE_DIM,)
    assert count == 2
    assert feat[-2] == 1.0  # left presence
    assert feat[-1] == 1.0  # right presence

    # Only right hand present
    feat_r, count_r = extract_features_from_arrays(None, right_hand)
    assert feat_r.shape == (FEATURE_DIM,)
    assert count_r == 1
    assert feat_r[-2] == 0.0  # left absent
    assert feat_r[-1] == 1.0  # right present
    assert np.all(feat_r[:140] == 0.0)  # left slot all zeros

    # No hands present
    feat_none, count_none = extract_features_from_arrays(None, None)
    assert feat_none.shape == (FEATURE_DIM,)
    assert count_none == 0
    assert np.all(feat_none == 0.0)


def test_feature_validation():
    valid = np.zeros(FEATURE_DIM, dtype=np.float32)
    assert validate_features(valid) is True

    # Wrong shape
    wrong_shape = np.zeros(192, dtype=np.float32)
    assert validate_features(wrong_shape) is False

    # NaN or Inf
    with_nan = valid.copy()
    with_nan[10] = np.nan
    assert validate_features(with_nan) is False

    with_inf = valid.copy()
    with_inf[50] = np.inf
    assert validate_features(with_inf) is False


def test_resample_sequence():
    # 10 frames sequence -> resampled to 30 frames
    short_seq = np.ones((10, FEATURE_DIM), dtype=np.float32) * 2.0
    resampled = resample_sequence(short_seq, target_length=30)
    assert resampled.shape == (30, FEATURE_DIM)
    assert np.allclose(resampled, 2.0)

    # 45 frames sequence -> resampled to 30 frames
    long_seq = np.zeros((45, FEATURE_DIM), dtype=np.float32)
    resampled_long = resample_sequence(long_seq, target_length=30)
    assert resampled_long.shape == (30, FEATURE_DIM)

    # Invalid sequence shapes
    with pytest.raises(ValueError):
        resample_sequence(np.zeros((10, 192), dtype=np.float32))

    with pytest.raises(ValueError):
        resample_sequence(np.empty((0, FEATURE_DIM), dtype=np.float32))


def test_verify_model_compatibility():
    class DummyModel:
        def __init__(self, shape):
            self.input_shape = shape

    # Compatible model (None, 30, 282)
    compat_model = DummyModel((None, 30, 282))
    verify_model_compatibility(compat_model, expected_dim=282, expected_seq_len=30)

    # Incompatible feature dimension (e.g. legacy 192 or 216)
    incompat_dim_model = DummyModel((None, 30, 192))
    with pytest.raises(IncompatibleModelError, match="feature dimension mismatch"):
        verify_model_compatibility(incompat_dim_model, expected_dim=282, expected_seq_len=30)

    # Incompatible sequence length
    incompat_seq_model = DummyModel((None, 60, 282))
    with pytest.raises(IncompatibleModelError, match="sequence length mismatch"):
        verify_model_compatibility(incompat_seq_model, expected_dim=282, expected_seq_len=30)


def test_training_and_inference_feature_parity():
    """Verify that simulated training extraction and inference extraction yield identical vectors."""
    rng = np.random.RandomState(42)
    mock_left = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    mock_right = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)

    # Feature extraction path A (array interface used in data collection & training)
    features_a, _ = extract_features_from_arrays(mock_left, mock_right)

    # Feature extraction path B (simulated live HandLandmarkResult dataclass)
    class MockResult:
        left_hand_detected = True
        right_hand_detected = True
        left_landmarks = mock_left
        right_landmarks = mock_right
        raw_results = None

    features_b, _ = extract_frame_features(MockResult())

    assert np.allclose(features_a, features_b, atol=1e-7)
