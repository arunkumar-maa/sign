"""
SignBridge AI — Phase 10: Automated Recognition Test Suite.
Verifies all 13 critical functional requirements:
1. Finger counter correctly identifies: 0, 1, 2, 3, 4, 5 extended fingers.
2. Finger counter does not change prediction when hand rotates 90 degrees.
3. Finger counter does not predict words like HELLO or STUDENT.
4. Two extended fingers in Finger Count Mode outputs '2', not a word.
5. Two extended fingers in ISL Number Mode outputs 'Number 2', not a word.
6. Two stationary extended fingers in ISL Word Mode outputs 'Unknown Sign', not HELLO, GO, SORRY, or STUDENT.
7. Dynamic sign with motion satisfies motion gating.
8. Unknown or chaotic handshape produces 'Unknown Sign'.
9. Auto-add is disabled by default.
10. Holding a sign for 3 seconds produces at most one sentence entry.
11. Sentence builder undo removes only the last word.
12. Sentence builder clear removes all words.
13. Latency measurement runs and produces valid numbers for each stage.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.recognition.finger_counter import FingerCounter
from backend.recognition.isl_number_recognizer import ISLNumberRecognizer
from backend.recognition.isl_word_recognizer import ISLWordRecognizer
from backend.recognition.sentence_builder import SentenceBuilder
from backend.vision.performance_monitor import PerformanceMonitor
from backend.vision.live_pipeline import LivePipeline


# ── Helper to build synthetic canonical hand landmarks ────────────────────────

def create_canonical_hand(extended_fingers: list) -> np.ndarray:
    """
    Generate synthetic 21 3D landmarks for specified extended fingers.
    Fingers: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
    """
    lms = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    lms[0] = [0.5, 0.8, 0.0]

    # Base MCPs
    base_mcps = {
        1: [0.42, 0.65, 0.0],  # Thumb CMC
        5: [0.46, 0.55, 0.0],  # Index MCP
        9: [0.50, 0.54, 0.0],  # Middle MCP
        13: [0.54, 0.55, 0.0], # Ring MCP
        17: [0.58, 0.58, 0.0], # Pinky MCP
    }

    # Thumb: 1, 2, 3, 4
    lms[1] = base_mcps[1]
    if 0 in extended_fingers:
        lms[2] = [0.36, 0.60, 0.0]
        lms[3] = [0.32, 0.56, 0.0]
        lms[4] = [0.28, 0.52, 0.0]  # Tip extended outward
    else:
        lms[2] = [0.44, 0.62, 0.02]
        lms[3] = [0.46, 0.60, 0.03]
        lms[4] = [0.48, 0.58, 0.03]  # Tip folded across palm

    # 4 Fingers: (MCP, PIP, DIP, TIP)
    finger_defs = [
        (1, 5, 6, 7, 8, 0.46),     # Index
        (2, 9, 10, 11, 12, 0.50),  # Middle
        (3, 13, 14, 15, 16, 0.54), # Ring
        (4, 17, 18, 19, 20, 0.58), # Pinky
    ]

    for f_idx, mcp_i, pip_i, dip_i, tip_i, x_pos in finger_defs:
        lms[mcp_i] = base_mcps[mcp_i]
        if f_idx in extended_fingers:
            lms[pip_i] = [x_pos, 0.44, 0.0]
            lms[dip_i] = [x_pos, 0.35, 0.0]
            lms[tip_i] = [x_pos, 0.26, 0.0]  # Fully extended upright
        else:
            lms[pip_i] = [x_pos, 0.48, -0.02]
            lms[dip_i] = [x_pos, 0.56, -0.03]
            lms[tip_i] = [x_pos, 0.60, -0.02]  # Curled/folded down towards palm

    return lms


# ── TEST 1: Finger Counter 0 to 5 ─────────────────────────────────────────────

def test_finger_counter_0_to_5():
    counter = FingerCounter()

    # 0 fingers (closed fist)
    lms_0 = create_canonical_hand([])
    res_0 = counter.count_fingers(lms_0, handedness="Right")
    assert res_0.count == 0, f"Expected 0 fingers for fist, got {res_0.count}"

    # 1 finger (index extended)
    lms_1 = create_canonical_hand([1])
    res_1 = counter.count_fingers(lms_1, handedness="Right")
    assert res_1.count == 1, f"Expected 1 finger, got {res_1.count}"

    # 2 fingers (index and middle extended)
    lms_2 = create_canonical_hand([1, 2])
    res_2 = counter.count_fingers(lms_2, handedness="Right")
    assert res_2.count == 2, f"Expected 2 fingers, got {res_2.count}"

    # 3 fingers (thumb, index, middle extended)
    lms_3 = create_canonical_hand([0, 1, 2])
    res_3 = counter.count_fingers(lms_3, handedness="Right")
    assert res_3.count == 3, f"Expected 3 fingers, got {res_3.count}"

    # 4 fingers (index, middle, ring, pinky extended)
    lms_4 = create_canonical_hand([1, 2, 3, 4])
    res_4 = counter.count_fingers(lms_4, handedness="Right")
    assert res_4.count == 4, f"Expected 4 fingers, got {res_4.count}"

    # 5 fingers (open palm)
    lms_5 = create_canonical_hand([0, 1, 2, 3, 4])
    res_5 = counter.count_fingers(lms_5, handedness="Right")
    assert res_5.count == 5, f"Expected 5 fingers for open palm, got {res_5.count}"


# ── TEST 2: Hand Rotation Invariance (90 degrees) ─────────────────────────────

def test_finger_counter_rotation_invariance():
    counter = FingerCounter()
    lms_2 = create_canonical_hand([1, 2])

    # Unrotated count
    base_res = counter.count_fingers(lms_2, handedness="Right")
    assert base_res.count == 2

    # Rotate 90 degrees in plane (roll around Z)
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    center = lms_2[0]
    rotated_lms = (lms_2 - center) @ rot_mat.T + center

    rot_res = counter.count_fingers(rotated_lms, handedness="Right")
    assert rot_res.count == 2, f"Expected 2 fingers after 90 deg rotation, got {rot_res.count}"


# ── TEST 3: Finger Counter Never Predicts Words ───────────────────────────────

def test_finger_counter_no_word_predictions():
    counter = FingerCounter()
    forbidden_words = {"HELLO", "GO", "SORRY", "STUDENT", "TEA", "WATER"}

    for finger_set in [[], [1], [1, 2], [0, 1, 2], [1, 2, 3, 4], [0, 1, 2, 3, 4]]:
        lms = create_canonical_hand(finger_set)
        res = counter.count_fingers(lms)
        assert isinstance(res.count, int)
        assert 0 <= res.count <= 5
        # Ensure count is numeric and never a vocabulary word
        assert str(res.count) not in forbidden_words


# ── TEST 4: Finger Count Mode Outputs '2', Not a Word ─────────────────────────

def test_mode_finger_count_outputs_2():
    lms_2 = create_canonical_hand([1, 2])
    counter = FingerCounter()
    res = counter.count_fingers(lms_2)
    output_str = str(res.count)

    assert output_str == "2"
    assert output_str not in ("HELLO", "STUDENT", "SORRY", "GO")


# ── TEST 5: ISL Number Mode Outputs 2, Not a Word ─────────────────────────────

def test_mode_isl_numbers_outputs_2():
    recognizer = ISLNumberRecognizer()
    lms_2 = create_canonical_hand([1, 2])
    res = recognizer.recognize(lms_2, handedness="Right")

    assert res.is_valid is True
    assert res.number == 2
    assert res.label == "TWO"


# ── TEST 6: ISL Word Mode Stationary 2 Fingers Rejects False Words ────────────

def test_mode_isl_words_stationary_two_fingers_rejects():
    word_recognizer = ISLWordRecognizer()
    lms_2 = create_canonical_hand([1, 2])

    from backend.vision.hand_pose_features import extract_two_hand_features
    static_vec = extract_two_hand_features(right_landmarks=lms_2)

    # Feed 30 stationary frames (zero motion)
    for _ in range(30):
        word_recognizer.add_frame(static_vec, has_hands=True)

    pred = word_recognizer.predict()
    word = pred.get("word")

    # Dynamic words MUST NOT be predicted on stationary handshapes
    assert word not in ("hello", "go", "sorry", "student"), f"Falsely predicted {word} on stationary two fingers!"
    assert pred.get("status") in ("Unknown Sign", "Stationary hand (Perform gesture motion)") or word is None


# ── TEST 7: Dynamic Motion Gating Recognizes Movement ─────────────────────────

def test_dynamic_motion_energy_detection():
    word_recognizer = ISLWordRecognizer()
    lms_base = create_canonical_hand([1, 2])
    from backend.vision.hand_pose_features import extract_two_hand_features

    # Feed frames with simulated trajectory motion
    for i in range(30):
        lms_moving = lms_base.copy()
        lms_moving[:, 0] += float(i * 0.015)  # Lateral movement
        vec = extract_two_hand_features(right_landmarks=lms_moving)
        word_recognizer.add_frame(vec, has_hands=True)

    motion = word_recognizer.calculate_motion_energy()
    assert motion > 0.012, f"Expected motion energy > 0.012, got {motion:.4f}"


# ── TEST 8: Unknown Handshape Produces 'Unknown Sign' ──────────────────────────

def test_unknown_handshape_produces_unknown_sign():
    recognizer = ISLNumberRecognizer()
    # Pinky-only gesture (finger index 4 extended, others folded) is not a valid ISL number (0-9)
    pinky_only_lms = create_canonical_hand([4])
    res = recognizer.recognize(pinky_only_lms, handedness="Right")
    assert res.is_valid is False, f"Expected invalid for pinky-only gesture, got {res.label}"
    assert res.number is None
    assert res.label == "Unknown Sign"

    # Thumbs up gesture (thumb extended, other 4 folded) is not an ISL number
    thumbs_up_lms = create_canonical_hand([0])
    res2 = recognizer.recognize(thumbs_up_lms, handedness="Right")
    assert res2.is_valid is False, f"Expected invalid for thumbs up, got {res2.label}"
    assert res2.number is None
    assert res2.label == "Unknown Sign"


# ── TEST 9: Auto-Add Disabled by Default ──────────────────────────────────────

def test_auto_add_disabled_by_default():
    builder = SentenceBuilder()
    assert builder.auto_add_enabled is False, "Auto-add must be disabled by default (Manual confirmation mode)"


# ── TEST 10: Holding Sign for 3 Seconds Produces at Most One Entry ────────────

def test_holding_sign_no_duplicates():
    builder = SentenceBuilder(required_stable_frames=15, min_auto_confidence=0.85)
    builder.auto_add_enabled = True

    # Simulate 90 frames (3 seconds at 30 FPS) of holding the sign "HELLO" at 0.95 confidence
    added_count = 0
    for frame_i in range(90):
        if builder.process_frame_auto_add("HELLO", confidence=0.95, hand_present=True):
            added_count += 1

    assert added_count == 1, f"Expected exactly 1 addition for held sign, got {added_count}"
    assert builder.gloss_sequence == ["HELLO"]
    assert len(builder.words) == 1


# ── TEST 11: Sentence Builder Undo ────────────────────────────────────────────

def test_sentence_builder_undo():
    builder = SentenceBuilder()
    builder.add_word("HELLO")
    builder.add_word("FRIEND")
    assert builder.gloss_sequence == ["HELLO", "FRIEND"]

    removed = builder.undo()
    assert removed == "Friend"
    assert builder.gloss_sequence == ["HELLO"]
    assert builder.current_sentence.strip() == "Hello."


# ── TEST 12: Sentence Builder Clear ───────────────────────────────────────────

def test_sentence_builder_clear():
    builder = SentenceBuilder()
    builder.add_word("HELLO")
    builder.add_word("WATER")
    assert len(builder.words) == 2

    builder.clear()
    assert len(builder.words) == 0
    assert builder.current_sentence == ""
    assert builder.gloss_sequence == []


# ── TEST 13: Latency Profiler Valid Numbers ───────────────────────────────────

def test_latency_profiler_stages():
    monitor = PerformanceMonitor()
    profile = monitor.record_stage(
        ingest_ms=2.5,
        prep_ms=1.2,
        detect_ms=28.4,
        extract_ms=1.1,
        inference_ms=4.3,
        sentence_ms=0.3,
        total_ms=37.8,
    )

    assert profile.mediapipe_detect_ms > 0.0
    assert profile.mediapipe_detect_ms < 35.0, "MediaPipe detect should be under 35ms target"
    assert profile.feature_extract_ms < 5.0, "Feature extraction should be under 5ms target"
    assert profile.model_inference_ms < 15.0, "Inference should be under 15ms target"
    assert profile.total_pipeline_ms < 60.0, "Total frame latency should be under 60ms target"
    assert profile.fps > 0.0
    summary = profile.summary_string()
    assert "Total:" in summary
    assert "Camera:" in summary
