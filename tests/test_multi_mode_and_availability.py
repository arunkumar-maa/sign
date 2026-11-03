"""
SignBridge AI — Multi-Mode Recognition & 'Model Not Available' Validation Tests.
Verifies:
1. Mode selector support: "ISL Words", "ISL Alphabet/Fingerspelling", "ISL Numbers", "Finger Count"
2. Guard against missing models: displays "Model Not Available"
3. 282-dimensional feature vector extraction and temporal alignment
4. Sequence buffer management across mode switches
"""

import numpy as np
import pytest

from backend.vision.hand_landmarks import HandLandmarkResult
from backend.vision.live_pipeline import LivePipeline
from backend.recognition.isl_fingerspelling_recognizer import ISLFingerspellingRecognizer
from backend.recognition.isl_number_recognizer import ISLNumberRecognizer
from backend.recognition.isl_word_recognizer import ISLWordRecognizer


def test_282_feature_vector_dimension():
    """Verify that _extract_282_features produces exactly 282 floats."""
    pipeline = LivePipeline(mode="ISL Words")
    dummy_lms = np.zeros((21, 3), dtype=np.float32)
    dummy_lms[9] = [0.0, 0.2, 0.0]  # Middle MCP to establish palm scale

    res = HandLandmarkResult(
        detected=True,
        num_hands=2,
        left_hand_detected=True,
        right_hand_detected=True,
        left_landmarks=dummy_lms,
        right_landmarks=dummy_lms,
        fps=30.0,
    )

    feat = pipeline._extract_282_features(res)
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (282,), f"Expected shape (282,), got {feat.shape}"
    assert feat[-2] == 1.0, "Left presence flag must be 1.0"
    assert feat[-1] == 1.0, "Right presence flag must be 1.0"


def test_model_not_available_when_unloaded():
    """Verify that when a model is missing / unloaded, the recognizers and pipeline return 'Model Not Available'."""
    # 1. Word Recognizer
    word_rec = ISLWordRecognizer()
    word_rec.dynamic_model = None
    word_rec.static_model = None
    assert not word_rec.is_loaded
    pred = word_rec.predict()
    assert pred["status"] == "Model Not Available"
    assert not pred["is_model_available"]

    # 2. Fingerspelling Recognizer
    alph_rec = ISLFingerspellingRecognizer()
    alph_rec.model = None
    assert not alph_rec.is_loaded
    alph_pred = alph_rec.predict()
    assert alph_pred["status"] == "Model Not Available"
    assert not alph_pred["is_model_available"]

    # 3. Number Recognizer
    num_rec = ISLNumberRecognizer()
    num_rec.trained_model = None
    assert not num_rec.is_loaded
    num_pred = num_rec.predict()
    assert num_pred["status"] == "Model Not Available"
    assert not num_pred["is_model_available"]


def test_live_pipeline_mode_switching_and_availability():
    """Verify live pipeline switches modes and gates on model availability."""
    pipeline = LivePipeline(mode="ISL Words")
    dummy_lms = np.zeros((21, 3), dtype=np.float32)
    dummy_lms[9] = [0.0, 0.2, 0.0]
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Mock unequipped fingerspelling model
    pipeline.fingerspelling_recognizer.model = None
    pipeline.set_mode("ISL Alphabet/Fingerspelling")

    # Mock detection result with hand
    mock_res = HandLandmarkResult(
        detected=True,
        num_hands=1,
        right_hand_detected=True,
        right_landmarks=dummy_lms,
        fps=30.0,
    )
    pipeline.hand_detector.detect = lambda f, is_bgr=False: mock_res

    res = pipeline.process_frame(frame, mode="ISL Alphabet/Fingerspelling")
    assert res["current_sign"] == "Model Not Available"
    assert "not available" in res["hud_status"].lower()


def test_live_pipeline_buffer_reset_on_mode_switch():
    """Verify sequence buffers are reset when switching modes."""
    pipeline = LivePipeline(mode="ISL Words")
    pipeline.word_recognizer._buffer.append(np.zeros(282))
    assert pipeline.word_recognizer.buffer_count == 1

    pipeline.set_mode("ISL Numbers")
    assert pipeline.word_recognizer.buffer_count == 0
    assert pipeline.number_recognizer.buffer_count == 0
    assert pipeline.fingerspelling_recognizer.buffer_count == 0

