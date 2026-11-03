"""
SignBridge AI — Phase 13: End-to-End ISL Integration Test Suite.
Tests:
1. Camera input processing (RGB / BGR / None)
2. MediaPipe HandLandmarker Tasks API & green/blue skeleton drawing
3. 21 landmark extraction (x, y, z) & handedness
4. One-hand recognition input representation
5. Two-hand recognition input representation
6. Missing hand handling & zero-padding
7. Landmark normalization (wrist origin & scale invariance)
8. ISL Word model loading & vocabulary validation
9. Prediction confidence & sequence buffering
10. Unknown sign rejection
11. Sentence generation (Add, Undo, Clear, ordered sequence)
12. Duplicate prediction prevention
13. Text-to-speech synthesis
"""

import sys
import time
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 1. Camera Input Handling
def test_camera_input_handling():
    from backend.vision.hand_landmarks import HandLandmarker

    with HandLandmarker() as hl:
        # None frame
        res_none = hl.detect(None)
        assert res_none.detected is False
        assert res_none.num_hands == 0

        # Empty frame
        res_empty = hl.detect(np.array([], dtype=np.uint8))
        assert res_empty.detected is False

        # Valid RGB frame
        rgb_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        res_rgb = hl.detect(rgb_frame, is_bgr=False)
        assert res_rgb is not None
        assert res_rgb.inference_latency_ms >= 0


# 2. Hand Detection Tasks API & Green/Blue Skeleton Drawing
def test_hand_detection_tasks_and_drawing():
    from backend.vision.hand_landmarks import HandLandmarker, HandLandmarkResult

    with HandLandmarker() as hl:
        assert hl._using_tasks_api is True, "Must use MediaPipe Tasks HandLandmarker"

        # Create dummy detection result to verify drawing
        dummy_landmarks = np.zeros((21, 3), dtype=np.float32)
        dummy_landmarks[0] = [0.5, 0.7, 0.0]   # Wrist
        dummy_landmarks[9] = [0.5, 0.4, 0.0]   # Middle MCP

        res = HandLandmarkResult(
            detected=True,
            num_hands=1,
            right_hand_detected=True,
            right_landmarks=dummy_landmarks,
            right_confidence=0.92,
        )

        test_img = np.full((480, 640, 3), 200, dtype=np.uint8)
        annotated = hl.draw_landmarks(test_img, res, is_bgr=False)
        assert annotated.shape == (480, 640, 3)
        # Check that drawing actually modified the canvas
        assert not np.array_equal(test_img, annotated)


# 3. 21 Landmark Extraction (X, Y, Z) and Handedness
def test_landmark_extraction_coordinates_and_handedness():
    from backend.vision.hand_landmarks import HandLandmarkResult

    l_lms = np.random.rand(21, 3).astype(np.float32)
    r_lms = np.random.rand(21, 3).astype(np.float32)

    res = HandLandmarkResult(
        detected=True,
        num_hands=2,
        left_hand_detected=True,
        left_landmarks=l_lms,
        left_confidence=0.88,
        right_hand_detected=True,
        right_landmarks=r_lms,
        right_confidence=0.94,
    )

    assert res.left_landmarks.shape == (21, 3)
    assert res.right_landmarks.shape == (21, 3)
    assert res.num_hands == 2
    assert res.left_hand_detected is True
    assert res.right_hand_detected is True


# 4 & 5. One-Hand and Two-Hand Recognition Input
def test_one_hand_and_two_hand_representations():
    from backend.vision.landmark_preprocessing import (
        extract_normalized_hand_features,
        construct_multimodal_vector,
    )

    one_hand = np.random.rand(21, 3).astype(np.float32)

    # One hand (Right only)
    feat_1, left_p1, right_p1 = extract_normalized_hand_features(None, one_hand)
    assert feat_1.shape == (126,)
    assert left_p1 is False
    assert right_p1 is True
    assert np.all(feat_1[:63] == 0.0)  # Left hand zero-padded

    # Two hands
    second_hand = np.random.rand(21, 3).astype(np.float32)
    feat_2, left_p2, right_p2 = extract_normalized_hand_features(one_hand, second_hand)
    assert feat_2.shape == (126,)
    assert left_p2 is True
    assert right_p2 is True

    # 192-dim unified vector
    multi = construct_multimodal_vector(one_hand, second_hand)
    assert multi.shape == (192,)


# 6. Missing Hand Handling
def test_missing_hand_handling():
    from backend.vision.landmark_preprocessing import normalize_hand_landmarks

    # None input
    norm_none = normalize_hand_landmarks(None)
    assert norm_none.shape == (21, 3)
    assert np.all(norm_none == 0.0)

    # Empty array
    norm_empty = normalize_hand_landmarks(np.array([]))
    assert norm_empty.shape == (21, 3)
    assert np.all(norm_empty == 0.0)


# 7. Landmark Normalization (Wrist Origin & Scale Invariance)
def test_landmark_normalization_invariance():
    from backend.vision.landmark_preprocessing import normalize_hand_landmarks

    # Create synthetic hand with offset wrist (e.g. at 100, 200)
    raw = np.zeros((21, 3), dtype=np.float32)
    raw[0] = [100.0, 200.0, 10.0]  # Wrist
    raw[9] = [100.0, 250.0, 10.0]  # Middle MCP (distance = 50.0)
    raw[8] = [100.0, 300.0, 10.0]  # Index Tip

    normalized = normalize_hand_landmarks(raw)

    # Wrist must be at origin (0, 0, 0)
    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-5)
    # Joint 9 distance must be exactly 1.0 (unit scale)
    dist_mcp9 = np.linalg.norm(normalized[9])
    assert abs(dist_mcp9 - 1.0) < 1e-4


# 8. ISL Word Model Loading & Vocabulary Validation
def test_isl_word_model_loading():
    from backend.recognition.isl_word_recognizer import ISLWordRecognizer
    import config as cfg

    model_path = cfg.MODELS_DIR / "isl_word_model.keras"
    labels_path = cfg.MODELS_DIR / "word_labels.json"

    assert model_path.exists(), "models/isl_word_model.keras must exist"
    assert labels_path.exists(), "models/word_labels.json must exist"

    recognizer = ISLWordRecognizer(model_path=model_path, labels_path=labels_path)
    assert recognizer.is_loaded is True
    assert len(recognizer._labels) == 40
    assert "hello" in [lbl.lower() for lbl in recognizer._labels]


# 9. Prediction Confidence & Sequence Buffering
def test_prediction_confidence_and_buffering():
    from backend.recognition.isl_word_recognizer import ISLWordRecognizer
    import config as cfg

    rec = ISLWordRecognizer(
        model_path=cfg.MODELS_DIR / "isl_word_model.keras",
        labels_path=cfg.MODELS_DIR / "word_labels.json",
        sequence_length=30,
        confidence_threshold=0.35,
    )

    rec.reset()
    assert rec._buffer.maxlen == 30

    # Feeding fewer than 30 frames reports buffering
    for _ in range(15):
        rec.add_frame(np.zeros(192, dtype=np.float32), has_hands=True)

    pred = rec.predict()
    assert pred["status_detail"] == "buffering"
    assert "Collecting sign frames" in pred["status"]

    # Fill remaining 15 frames
    for _ in range(15):
        rec.add_frame(np.zeros(192, dtype=np.float32), has_hands=True)

    assert len(rec._buffer) == 30
    pred_full = rec.predict()
    assert "confidence" in pred_full
    assert pred_full["status_detail"] in ("recognized", "low_confidence", "smoothing")


# 10. Unknown Sign Rejection
def test_unknown_sign_rejection():
    from backend.recognition.isl_word_recognizer import ISLWordRecognizer
    import config as cfg

    # Enforce ultra-high threshold (0.999) to verify rejection
    rec = ISLWordRecognizer(
        model_path=cfg.MODELS_DIR / "isl_word_model.keras",
        labels_path=cfg.MODELS_DIR / "word_labels.json",
        confidence_threshold=0.999,
    )
    rec.reset()
    for _ in range(30):
        rec.add_frame(np.random.rand(192).astype(np.float32), has_hands=True)

    pred = rec.predict()
    assert pred["status_detail"] == "low_confidence"
    assert pred["word"] is None
    assert pred["status"] == "Unknown sign"


# 11. Sentence Generation (Add, Undo, Clear, Ordered Sequence)
def test_sentence_builder_crud():
    from backend.recognition.sentence_builder import SentenceBuilder

    sb = SentenceBuilder(repeat_cooldown_seconds=0.1)

    # 1. Add words
    assert sb.add_word("hello", 0.95) is True
    assert sb.add_word("friend", 0.88) is True
    assert sb.words == ["Hello", "Friend"]
    assert sb.current_sentence == "Hello Friend."

    # 2. Undo
    undone = sb.undo()
    assert undone == "Friend"
    assert sb.words == ["Hello"]
    assert sb.current_sentence == "Hello."

    # 3. User manual edit
    sb.set_user_text("Hello there!")
    assert sb.current_sentence == "Hello there!"

    # 4. Clear
    sb.clear()
    assert sb.words == []
    assert sb.current_sentence == ""


# 12. Duplicate Prediction Prevention
def test_duplicate_prediction_prevention():
    from backend.recognition.sentence_builder import SentenceBuilder

    sb = SentenceBuilder(repeat_cooldown_seconds=1.0)
    sb.add_word("water", 0.90)

    # Immediate duplicate within cooldown is blocked
    added_again = sb.add_word("water", 0.90)
    assert added_again is False
    assert len(sb.words) == 1

    # Different word is accepted immediately
    added_diff = sb.add_word("please", 0.85)
    assert added_diff is True
    assert len(sb.words) == 2


# 13. Text-to-Speech Synthesis
def test_text_to_speech_audio():
    from backend.audio.text_to_speech import TTSService
    import config as cfg

    tts = TTSService(
        provider=cfg.TTS_PROVIDER,
        voice_map=cfg.EDGE_TTS_VOICES,
        gtts_lang_map=cfg.GTTS_LANG_MAP,
    )
    path, err = tts.synthesize("SignBridge AI sentence audio test.", "en")
    assert path is not None
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
    assert err is None

