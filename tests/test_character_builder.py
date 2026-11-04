"""
Tests for backend/recognition/character_builder.py.
Verifies:
- Character accumulation (e.g. 'HELLO123')
- Character validation (A-Z, 0-9)
- Rejection of invalid/empty tokens
- Backspace and clear operations
- Auto-add stabilization and neutral release gating
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.recognition.character_builder import CharacterTextBuilder


def test_character_builder_manual_add():
    builder = CharacterTextBuilder()
    assert builder.text == ""

    # Add characters one by one
    assert builder.add_character("H") is True
    assert builder.add_character("E") is True
    assert builder.add_character("L") is True
    assert builder.add_character("L") is True
    assert builder.add_character("O") is True
    assert builder.text == "HELLO"

    # Add numbers
    assert builder.add_character("1") is True
    assert builder.add_character("2") is True
    assert builder.add_character("3") is True
    assert builder.text == "HELLO123"


def test_character_builder_reject_invalid():
    builder = CharacterTextBuilder()
    assert builder.add_character("—") is False
    assert builder.add_character("UNKNOWN") is False
    assert builder.add_character("") is False
    assert builder.add_character(None) is False
    assert builder.add_character("HELLO_LONG_WORD") is False
    assert builder.text == ""


def test_character_builder_backspace_and_clear():
    builder = CharacterTextBuilder()
    builder.set_text("ABC12")
    assert builder.text == "ABC12"

    # Backspace
    assert builder.backspace() == "ABC1"
    assert builder.backspace() == "ABC"
    assert builder.text == "ABC"

    # Clear
    assert builder.clear() == ""
    assert builder.text == ""

    # Backspace on empty
    assert builder.backspace() == ""


def test_character_builder_auto_add_and_neutral_release():
    builder = CharacterTextBuilder(required_stable_frames=4, min_auto_confidence=0.7)
    builder.auto_add_enabled = True

    # 3 frames of 'A' (not enough yet)
    for _ in range(3):
        res = builder.process_auto_frame("A", confidence=0.85, hand_detected=True)
        assert res is None
    assert builder.text == ""

    # 4th frame of 'A' -> triggered!
    res = builder.process_auto_frame("A", confidence=0.85, hand_detected=True)
    assert res == "A"
    assert builder.text == "A"

    # Continuous holding of 'A' -> must NOT duplicate!
    for _ in range(10):
        res = builder.process_auto_frame("A", confidence=0.85, hand_detected=True)
        assert res is None
    assert builder.text == "A"

    # User drops hand / neutral pose for 3 frames
    for _ in range(3):
        builder.process_auto_frame(None, confidence=0.0, hand_detected=False)

    # Now user signs 'A' again for 4 frames -> can be added again!
    for _ in range(3):
        res = builder.process_auto_frame("A", confidence=0.85, hand_detected=True)
        assert res is None
    res = builder.process_auto_frame("A", confidence=0.85, hand_detected=True)
    assert res == "A"
    assert builder.text == "AA"
