"""
SignBridge AI — Core Unit Test Suite.
Verifies essential modules without requiring external hardware or internet.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_config_loads():
    import config as cfg
    assert cfg.STATIC_CONFIDENCE_THRESHOLD > 0
    assert cfg.DYNAMIC_CONFIDENCE_THRESHOLD > 0
    assert "HELLO" in cfg.ISL_PHRASE_TRANSLATIONS["en"]
    assert "HELLO" in cfg.ISL_PHRASE_TRANSLATIONS["ta"]


def test_database_crud(tmp_path):
    from backend.database.database import Database
    db_file = tmp_path / "test_history.db"
    db = Database(db_file)

    # Save
    rec_id = db.save_conversation(
        gloss_sequence=["HELLO", "HELP"],
        english_text="Hello, I need help.",
        translated_text="வணக்கம், எனக்கு உதவி தேவை.",
        language_code="ta",
        language_name="Tamil",
    )
    assert rec_id is not None
    assert len(rec_id) > 0

    # Load
    records = db.load_history()
    assert len(records) == 1
    assert records[0]["english_text"] == "Hello, I need help."
    assert records[0]["gloss_sequence"] == ["HELLO", "HELP"]

    # Delete
    deleted = db.delete_record(rec_id)
    assert deleted is True
    assert len(db.load_history()) == 0


def test_sentence_builder():
    from backend.recognition.sentence_builder import SentenceBuilder
    gloss_map = {
        "HELLO": "Hello!",
        "HELP": "Help!",
        "WATER": "I need water.",
    }
    sb = SentenceBuilder(gloss_map)
    assert sb.suggested_sentence == ""

    sb.add_gloss("HELLO", 0.95)
    assert "Hello!" in sb.suggested_sentence

    sb.add_gloss("HELP", 0.88)
    assert "Help!" in sb.suggested_sentence
    assert sb.gloss_sequence == ["HELLO", "HELP"]

    # Undo
    sb.undo()
    assert sb.gloss_sequence == ["HELLO"]

    # Clear
    sb.clear()
    assert len(sb.gloss_sequence) == 0


def test_temporal_filter():
    from backend.recognition.temporal_filter import TemporalFilter
    tf = TemporalFilter(stability_frames=3, repeat_block_frames=5, neutral_frames=2)

    # 1st and 2nd frame: not yet stable
    assert tf.update({"gesture": "HELLO", "status": "recognized"}) is None
    assert tf.update({"gesture": "HELLO", "status": "recognized"}) is None

    # 3rd frame: stable -> accepted
    assert tf.update({"gesture": "HELLO", "status": "recognized"}) == "HELLO"

    # Held gesture: repeat blocked
    assert tf.update({"gesture": "HELLO", "status": "recognized"}) is None

    # Neutral pose resets block
    tf.update({"gesture": None, "status": "none"})
    tf.update({"gesture": None, "status": "none"})

    # New sign after neutral
    tf.update({"gesture": "HELP", "status": "recognized"})
    tf.update({"gesture": "HELP", "status": "recognized"})
    assert tf.update({"gesture": "HELP", "status": "recognized"}) == "HELP"


def test_emergency_service():
    from backend.services.emergency_service import EmergencyService
    svc = EmergencyService()
    msgs = svc.get_all_messages()
    assert len(msgs) >= 5

    msg = svc.get_message("medical_help")
    assert "medical help" in msg["english"].lower()
    assert "மருத்துவ உதவி" in msg["tamil"]

    formatted = svc.format_display("medical_help", lang="both")
    assert "medical help" in formatted.lower()
    assert "மருத்துவ உதவி" in formatted
    assert "DISCLAIMER" in formatted or "IMPORTANT" in formatted


def test_avatar_unverified_guard(tmp_path):
    """Ensure that signs with unverified status or missing files CANNOT play."""
    import json
    from backend.avatar.animation_mapping import AnimationMapper

    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "animations": [
            {
                "gesture": "HELLO",
                "animation_path": "nonexistent.mp4",
                "status": "pending",  # not verified
            }
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    mapper = AnimationMapper(manifest_file)
    clip = mapper.get_clip("HELLO")
    assert clip["available"] is False
    assert "not yet supported" in clip["message"].lower()

    # Unknown sign
    unknown = mapper.get_clip("UNKNOWN_SIGN")
    assert unknown["available"] is False


def test_static_recognizer_unavailable_message():
    """Ensure static recognizer reports model unavailable rather than failing or faking."""
    from backend.recognition.static_recognizer import StaticRecognizer
    sr = StaticRecognizer(
        model_path=Path("nonexistent_model.keras"),
        labels_path=Path("nonexistent_labels.json"),
    )
    assert not sr.is_loaded
    pred = sr.predict(np.zeros(192, dtype=np.float32))
    assert pred["status"] == "model_unavailable"
    assert pred["gesture"] is None
    assert "unavailable" in pred["message"].lower()


def test_dynamic_recognizer_unavailable_message():
    """Ensure dynamic recognizer reports model unavailable when model is missing."""
    from backend.recognition.dynamic_recognizer import DynamicRecognizer
    dr = DynamicRecognizer(
        model_path=Path("nonexistent_dynamic.keras"),
        labels_path=Path("nonexistent_labels.json"),
    )
    assert not dr.is_loaded
    pred = dr.predict()
    assert pred["status"] == "model_unavailable"
    assert pred["gesture"] is None

