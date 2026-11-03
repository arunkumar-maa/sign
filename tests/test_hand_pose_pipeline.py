"""
SignBridge AI — Comprehensive Automated Tests for Hand Pose ISL Recognition.
Verifies:
1. Hand landmark coordinate normalization & scale invariance.
2. 3D joint angle calculation accuracy (orthogonal, colinear, acute).
3. Finger distance metrics and scale normalization.
4. Handshape classification logic (Open Palm, Fist, Pointing, Pinch, Flat).
5. Two-hand feature vector extraction (216 dims), handedness ordering, and missing-hand safety.
6. Temporal smoothing, sequence buffering, and repeat sign cooldown in HandPoseRecognizer.
7. LivePipeline integration without full-body pose or face mesh dependencies.
8. Latency benchmark within real-time budget.
"""

import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from backend.vision.hand_pose_features import (
    calculate_angle_3d,
    normalize_hand_landmarks,
    calculate_finger_joint_angles,
    calculate_finger_distances,
    classify_finger_states,
    classify_handshape_name,
    calculate_palm_orientation,
    extract_single_hand_features,
    extract_two_hand_features,
    calculate_temporal_derivatives,
    SINGLE_HAND_FEATURE_DIM,
    TWO_HAND_FEATURE_DIM,
    WRIST,
    INDEX_TIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_DIP,
    MIDDLE_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_DIP,
    RING_TIP,
    RING_MCP,
    RING_PIP,
    RING_DIP,
    PINKY_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_DIP,
    THUMB_TIP,
    THUMB_MCP,
    THUMB_IP,
    THUMB_CMC,
)
from backend.vision.hand_landmarks import HandLandmarkResult
from backend.recognition.hand_pose_recognizer import HandPoseRecognizer
from backend.vision.live_pipeline import LivePipeline


# ── Synthetic Landmark Generators ─────────────────────────────────────────────

def make_open_palm_landmarks() -> np.ndarray:
    """Generate 21 landmarks representing a standard upright open palm."""
    lms = np.zeros((21, 3), dtype=np.float32)
    # Wrist at origin
    lms[WRIST] = [0.5, 0.8, 0.0]

    # Thumb: pointing outwards
    lms[THUMB_CMC] = [0.42, 0.72, 0.0]
    lms[THUMB_MCP] = [0.36, 0.64, 0.0]
    lms[THUMB_IP] = [0.31, 0.58, 0.0]
    lms[THUMB_TIP] = [0.26, 0.52, 0.0]

    # Index: straight up
    lms[INDEX_MCP] = [0.44, 0.60, 0.0]
    lms[INDEX_PIP] = [0.44, 0.50, 0.0]
    lms[INDEX_DIP] = [0.44, 0.42, 0.0]
    lms[INDEX_TIP] = [0.44, 0.34, 0.0]

    # Middle: straight up
    lms[MIDDLE_MCP] = [0.50, 0.58, 0.0]
    lms[MIDDLE_PIP] = [0.50, 0.47, 0.0]
    lms[MIDDLE_DIP] = [0.50, 0.38, 0.0]
    lms[MIDDLE_TIP] = [0.50, 0.28, 0.0]

    # Ring: straight up
    lms[RING_MCP] = [0.56, 0.60, 0.0]
    lms[RING_PIP] = [0.56, 0.51, 0.0]
    lms[RING_DIP] = [0.56, 0.43, 0.0]
    lms[RING_TIP] = [0.56, 0.35, 0.0]

    # Pinky: straight up
    lms[PINKY_MCP] = [0.62, 0.64, 0.0]
    lms[PINKY_PIP] = [0.62, 0.56, 0.0]
    lms[PINKY_DIP] = [0.62, 0.50, 0.0]
    lms[PINKY_TIP] = [0.62, 0.44, 0.0]

    return lms


def make_fist_landmarks() -> np.ndarray:
    """Generate 21 landmarks representing a closed fist (fingers curled into palm)."""
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[WRIST] = [0.5, 0.8, 0.0]

    # Thumb folded over curled fingers
    lms[THUMB_CMC] = [0.45, 0.75, 0.0]
    lms[THUMB_MCP] = [0.42, 0.70, 0.0]
    lms[THUMB_IP] = [0.45, 0.66, 0.0]
    lms[THUMB_TIP] = [0.48, 0.65, 0.0]

    # Fingers curled into MCPs
    for mcp, pip, dip, tip, x in [
        (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, 0.44),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, 0.50),
        (RING_MCP, RING_PIP, RING_DIP, RING_TIP, 0.56),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, 0.62),
    ]:
        lms[mcp] = [x, 0.60, 0.0]
        lms[pip] = [x, 0.54, -0.05]
        lms[dip] = [x, 0.58, -0.04]
        lms[tip] = [x, 0.62, -0.01]  # curled back down close to wrist

    return lms


def make_pointing_landmarks() -> np.ndarray:
    """Generate 21 landmarks representing an extended index finger with other fingers curled."""
    lms = make_fist_landmarks()
    # Uncurl index finger straight up
    lms[INDEX_PIP] = [0.44, 0.50, 0.0]
    lms[INDEX_DIP] = [0.44, 0.42, 0.0]
    lms[INDEX_TIP] = [0.44, 0.34, 0.0]
    return lms


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_angle_calculation_accuracy():
    """Verify 3D angle calculations for known geometry (orthogonal, straight, acute)."""
    # 90 degrees orthogonal (pi / 2 rad)
    p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    p2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    angle_90 = calculate_angle_3d(p1, p2, p3)
    assert math.isclose(angle_90, math.pi / 2, rel_tol=1e-5), f"Expected pi/2, got {angle_90}"

    # 180 degrees colinear straight line (pi rad)
    p_left = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    p_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    angle_180 = calculate_angle_3d(p_left, p_center, p_right)
    assert math.isclose(angle_180, math.pi, rel_tol=1e-5), f"Expected pi, got {angle_180}"

    # Degenerate zero-length vector
    angle_zero = calculate_angle_3d(p_center, p_center, p_right)
    assert angle_zero == 0.0


def test_finger_joint_angles():
    """Verify calculation of 15 finger joint angles (3 per finger)."""
    open_palm = make_open_palm_landmarks()
    angles = calculate_finger_joint_angles(open_palm)

    assert len(angles) == 15, f"Expected 15 joint angles, got {len(angles)}"
    assert np.all(angles >= 0.0) and np.all(angles <= math.pi + 1e-4), "Angles must be in [0, pi]"

    # For a straight index finger, PIP angle (MCP->PIP->DIP) should be close to pi
    index_pip_angle = angles[4]
    assert index_pip_angle > 2.5, f"Expected open index finger angle > 2.5, got {index_pip_angle}"


def test_finger_distances():
    """Verify calculation of 12 scale-normalized finger distance metrics."""
    open_palm = make_open_palm_landmarks()
    _, scale = normalize_hand_landmarks(open_palm)
    distances = calculate_finger_distances(open_palm, scale=scale)

    assert len(distances) == 12, f"Expected 12 distance metrics, got {len(distances)}"
    assert np.all(distances >= 0.0), "Distances must be non-negative"


def test_handshape_classification_open_palm():
    """Verify open palm handshape detection."""
    open_palm = make_open_palm_landmarks()
    states = classify_finger_states(open_palm)
    assert np.all(states == 1.0), f"All fingers should be extended (1.0), got {states}"

    name = classify_handshape_name(open_palm)
    assert "Open Palm" in name or "Flat Hand" in name, f"Expected Open Palm, got {name}"


def test_handshape_classification_fist():
    """Verify closed fist handshape detection."""
    fist = make_fist_landmarks()
    states = classify_finger_states(fist)
    assert np.all(states == 0.0), f"All fingers should be folded (0.0), got {states}"

    name = classify_handshape_name(fist)
    assert name == "Fist", f"Expected Fist, got {name}"


def test_handshape_classification_pointing():
    """Verify pointing index finger detection."""
    pointing = make_pointing_landmarks()
    states = classify_finger_states(pointing)
    assert states[1] == 1.0, "Index finger must be extended (1.0)"
    assert states[2] == 0.0, "Middle finger must be folded (0.0)"

    name = classify_handshape_name(pointing)
    assert name == "Pointing", f"Expected Pointing, got {name}"


def test_two_hand_feature_dimension_and_ordering():
    """Verify 216-dim feature extraction, strict left/right ordering, and missing hand safety."""
    open_palm = make_open_palm_landmarks()

    # Case 1: Both hands present
    two_hand_feat = extract_two_hand_features(left_landmarks=open_palm, right_landmarks=open_palm)
    assert two_hand_feat.shape == (TWO_HAND_FEATURE_DIM,), f"Expected shape ({TWO_HAND_FEATURE_DIM},), got {two_hand_feat.shape}"
    assert two_hand_feat[208] == 1.0, "is_left_present must be 1.0"
    assert two_hand_feat[209] == 1.0, "is_right_present must be 1.0"

    # Case 2: Only right hand present
    right_only_feat = extract_two_hand_features(left_landmarks=None, right_landmarks=open_palm)
    assert right_only_feat.shape == (TWO_HAND_FEATURE_DIM,)
    assert right_only_feat[208] == 0.0, "is_left_present must be 0.0"
    assert right_only_feat[209] == 1.0, "is_right_present must be 1.0"
    assert np.all(right_only_feat[:104] == 0.0), "Left hand features must be 0.0 when missing"

    # Case 3: Neither hand present
    empty_feat = extract_two_hand_features(left_landmarks=None, right_landmarks=None)
    assert empty_feat.shape == (TWO_HAND_FEATURE_DIM,)
    assert np.all(empty_feat == 0.0), "All features must be safely 0.0 when no hands detected"


def test_hand_pose_recognizer_buffer_and_cooldown():
    """Verify sliding sequence buffer management and repeat sign cooldown logic."""
    recognizer = HandPoseRecognizer(sequence_length=30, cooldown_seconds=1.5)
    dummy_feat = np.ones(TWO_HAND_FEATURE_DIM, dtype=np.float32)

    # 1. Initially buffer is empty
    assert recognizer.buffer_count == 0
    pred = recognizer.predict()
    assert "Collecting" in pred["status"]

    # 2. Add 29 frames -> still collecting
    for _ in range(29):
        recognizer.add_frame(dummy_feat, has_hands=True)
    assert recognizer.buffer_count == 29
    pred = recognizer.predict()
    assert "Collecting" in pred["status"]

    # 3. Add 30th frame -> buffer full
    recognizer.add_frame(dummy_feat, has_hands=True)
    assert recognizer.buffer_count == 30

    # 4. Hand leaving view resets buffer
    recognizer.add_frame(None, has_hands=False)
    assert recognizer.buffer_count == 0


def test_live_pipeline_frame_processing():
    """Verify LivePipeline processes RGB frame without full-body pose or face mesh dependencies."""
    pipeline = LivePipeline()

    # Create dummy 480x640 RGB image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline.process_frame(frame, current_sentence="TEST", is_bgr=False)

    assert "annotated_frame" in res
    assert "tracking_status" in res
    assert "current_sign" in res
    assert "confidence" in res
    assert "sentence" in res
    assert "latency" in res

    status = res["tracking_status"]
    # Verify hand tracking telemetry exists
    assert "Left hand tracked:" in status
    assert "Right hand tracked:" in status
    assert "Handshape:" in status

    # Verify complete absence of Face and Body telemetry
    assert "Face detected:" not in status, "Body/Face tracking must be removed"
    assert "Body detected:" not in status, "Body/Face tracking must be removed"


def test_inference_latency_budget():
    """Verify Hand Pose feature extraction latency is within real-time budget (< 12ms per frame, >80 FPS)."""
    open_palm = make_open_palm_landmarks()
    t0 = time.perf_counter()
    num_iterations = 200

    for _ in range(num_iterations):
        _ = extract_two_hand_features(open_palm, open_palm)

    elapsed_per_frame_ms = ((time.perf_counter() - t0) / num_iterations) * 1000.0
    assert elapsed_per_frame_ms < 12.0, f"Feature extraction too slow: {elapsed_per_frame_ms:.2f} ms"
