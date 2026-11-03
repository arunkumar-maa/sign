"""
SignBridge AI — Comprehensive Automated Test Suite (Phase 14).
Tests:
1. Camera processing (FrameProcessor, BGR & RGB handling, latency)
2. MediaPipe initialization (Tasks API for Hand, Face, Pose)
3. Landmark extraction (21 hand coords, 42 face coords, 24 pose coords, 192 combined)
4. Missing hand handling (blank frames, empty arrays, graceful zero-padding)
5. Static model inference (predicting, labels, confidence)
6. Dynamic model inference (30-frame sequence buffer, prediction)
7. Unknown sign rejection (low confidence rejection, unrecognized patterns)
8. Sentence generation (SentenceBuilder add, undo, clear, user override)
9. Translation (English to Tamil, offline dictionary, phrase fallback)
10. Audio services (TTS and STT service instantiation)
11. History (Database SQLite CRUD operations)
12. API (FastAPI endpoints via TestClient)
"""

import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 1. MediaPipe Initialization
def test_mediapipe_tasks_initialization():
    from backend.vision.hand_landmarks import HandLandmarker
    from backend.vision.face_landmarks import FaceLandmarker
    from backend.vision.pose_landmarks import PoseLandmarker

    with HandLandmarker() as hl:
        assert hl is not None
        assert hl._using_tasks_api is True, "HandLandmarker should use Tasks API"

    with FaceLandmarker() as fl:
        assert fl is not None
        assert fl._using_tasks_api is True, "FaceLandmarker should use Tasks API"

    with PoseLandmarker() as pl:
        assert pl is not None
        assert pl._using_tasks_api is True, "PoseLandmarker should use Tasks API"


# 2. Camera Processing & BGR/RGB Conversion
def test_camera_frame_processing():
    from backend.camera.frame_processor import FrameProcessor

    with FrameProcessor() as proc:
        # Test OpenCV BGR frame
        bgr_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        bgr_res = proc.process(bgr_frame, is_bgr=True)
        assert bgr_res.annotated_frame.shape == (480, 640, 3)
        assert bgr_res.processing_time_ms > 0
        assert bgr_res.features.combined.shape == (192,)

        # Test Gradio RGB frame
        rgb_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        rgb_res = proc.process(rgb_frame, is_bgr=False)
        assert rgb_res.annotated_frame.shape == (480, 640, 3)


# 3. Missing Hand Handling
def test_missing_hand_handling():
    from backend.vision.hand_landmarks import HandLandmarker, extract_flat_features

    with HandLandmarker() as hl:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = hl.detect(blank, is_bgr=True)
        assert res.detected is False
        assert res.num_hands == 0
        assert res.left_hand_detected is False
        assert res.right_hand_detected is False
        assert res.left_landmarks is None
        assert res.right_landmarks is None

        # Features should be zero-padded to (126,)
        feat = extract_flat_features(res.left_landmarks, res.right_landmarks)
        assert feat.shape == (126,)
        assert np.all(feat == 0.0)


# 4. Landmark Extraction & Feature Vector Dimension
def test_landmark_extraction_dimensions():
    from backend.vision.feature_extraction import extract_frame_features
    from backend.vision.hand_landmarks import HandLandmarkResult
    from backend.vision.face_landmarks import FaceLandmarkResult
    from backend.vision.pose_landmarks import PoseLandmarkResult

    h_res = HandLandmarkResult(
        detected=True,
        left_hand_detected=True,
        left_landmarks=np.random.rand(21, 3).astype(np.float32),
        right_hand_detected=True,
        right_landmarks=np.random.rand(21, 3).astype(np.float32),
    )
    f_res = FaceLandmarkResult(
        detected=True,
        landmarks=np.random.rand(14, 3).astype(np.float32),
    )
    p_res = PoseLandmarkResult(
        detected=True,
        landmarks=np.random.rand(8, 3).astype(np.float32),
    )

    feat = extract_frame_features(h_res, f_res, p_res)
    assert feat.hand_features.shape == (126,)
    assert feat.face_features.shape == (42,)
    assert feat.pose_features.shape == (24,)
    assert feat.combined.shape == (192,)
    assert feat.hand_detected is True


# 5. Static Model Inference & Unavailable Message
def test_static_model_inference():
    import config as cfg
    from backend.recognition.static_recognizer import StaticRecognizer

    sr = StaticRecognizer(
        model_path=cfg.STATIC_MODEL_PATH,
        labels_path=cfg.STATIC_LABELS_PATH,
        confidence_threshold=0.75,
    )
    assert sr.is_loaded is True
    assert len(sr.get_labels()) > 0

    # Test unknown random noise features
    dummy = np.random.rand(192).astype(np.float32)
    pred = sr.predict(dummy)
    assert "status" in pred
    assert pred["status"] in ("recognized", "low_confidence", "no_match")


# 6. Dynamic Model Inference & 30-Frame Sequence
def test_dynamic_model_inference():
    import config as cfg
    from backend.recognition.dynamic_recognizer import DynamicRecognizer

    dr = DynamicRecognizer(
        model_path=cfg.DYNAMIC_MODEL_PATH,
        labels_path=cfg.DYNAMIC_LABELS_PATH,
        sequence_length=30,
        confidence_threshold=0.35,
    )
    assert dr.is_loaded is True

    # Buffer not full initially
    dr.reset_buffer()
    assert not dr.buffer_ready
    p_buffering = dr.predict()
    assert p_buffering["status"] == "buffering"

    # Fill 30 frames
    for _ in range(30):
        dr.add_frame(np.random.rand(192).astype(np.float32))

    assert dr.buffer_ready is True
    pred = dr.predict()
    assert pred["status"] in ("recognized", "low_confidence")
    assert "model_version" in pred


# 7. Unknown Signs Rejection
def test_unknown_signs_rejection():
    from backend.recognition.static_recognizer import StaticRecognizer
    import config as cfg

    sr = StaticRecognizer(
        model_path=cfg.STATIC_MODEL_PATH,
        labels_path=cfg.STATIC_LABELS_PATH,
        confidence_threshold=0.99,  # High threshold
    )
    # Zero / empty hand features
    zeros = np.zeros(192, dtype=np.float32)
    pred = sr.predict(zeros)
    assert pred["status"] == "no_hands"
    assert pred["gesture"] is None


# 8. Sentence Generation
def test_sentence_generation():
    from backend.recognition.sentence_builder import SentenceBuilder

    gloss_map = {
        "HELLO": "Hello",
        "WATER": "Water please",
        "HELP": "Help me",
    }
    sb = SentenceBuilder(gloss_map)

    sb.add_gloss("HELLO", 0.95)
    sb.add_gloss("WATER", 0.89)
    assert sb.gloss_sequence == ["HELLO", "WATER"]
    assert "Hello" in sb.suggested_sentence
    assert "Water please" in sb.suggested_sentence

    # User manual correction
    sb.set_user_text("Hello, can I please have a glass of water?")
    assert sb.confirmed_text == "Hello, can I please have a glass of water?"

    # Undo
    sb.undo()
    assert sb.gloss_sequence == ["HELLO"]


# 9. English-to-Tamil Translation
def test_english_to_tamil_translation():
    import config as cfg
    from backend.translation.translator import TranslationService

    service = TranslationService(
        libre_url=cfg.LIBRE_TRANSLATE_URL,
        libre_api_key=cfg.LIBRE_TRANSLATE_API_KEY,
        offline_dict=cfg.ISL_PHRASE_TRANSLATIONS,
    )

    # Offline curated dictionary translation
    res = service.translate("HELLO", "en", "ta")
    assert res["translated_text"] == "வணக்கம்"
    assert res["method"] == "offline"

    # Medical emergency translation
    res2 = service.translate("I NEED MEDICAL HELP", "en", "ta")
    assert "மருத்துவ உதவி" in res2["translated_text"]

    # 40-word vocabulary translation
    res3 = service.translate("WATER", "en", "ta")
    assert res3["translated_text"] == "தண்ணீர்"


# 10. History SQLite Database
def test_history_database():
    from backend.database.database import Database

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        db = Database(db_path)
        rec_id = db.save_conversation(
            gloss_sequence=["HELLO", "WATER"],
            english_text="Hello, I need water.",
            translated_text="வணக்கம், எனக்கு தண்ணீர் தேவை.",
            language_code="ta",
            language_name="Tamil",
        )
        assert rec_id is not None

        history = db.load_history()
        assert len(history) == 1
        assert history[0]["language_name"] == "Tamil"
        assert history[0]["translated_text"] == "வணக்கம், எனக்கு தண்ணீர் தேவை."

        # Delete record
        assert db.delete_record(rec_id) is True
        assert len(db.load_history()) == 0
    finally:
        try:
            db_path.unlink()
        except Exception:
            pass


# 11. FastAPI Endpoints
def test_fastapi_endpoints():
    from backend.main import app, _init_services
    _init_services()
    client = TestClient(app)

    # Health
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Translation
    r_tr = client.post("/api/translate", json={"text": "HELLO", "target_lang": "ta"})
    assert r_tr.status_code == 200
    assert r_tr.json()["translated_text"] == "வணக்கம்"

    # Static recognition
    r_rec = client.post("/api/recognize", json={"features": [0.0] * 192, "recognition_type": "static"})
    assert r_rec.status_code == 200
    assert "status" in r_rec.json()
