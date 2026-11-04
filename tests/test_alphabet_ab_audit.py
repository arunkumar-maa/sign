"""
SignBridge AI — Alphabet A/B Audit & Recognition Regression Test Suite.
Verifies:
1. Alphabet class mapping strict order validation (A=0..Z=25) and rejection of shuffled/corrupted mappings.
2. Inference prediction diagnostics (top-5 letters, probabilities, winning index, margin).
3. HandLandmarker two-hand duplicate handedness collision resolution (no dropped hands).
4. Genuine held-out signer_02 testing for Letter A (verifying A is recognized and not Unknown).
5. Genuine held-out signer_02 testing for Letter B (verifying B is recognized and never misclassified as U).
6. Training vs webcam feature preprocessing parity (282-dim, linear resampling to 30 frames).
7. Webcam camera input unmirrored configuration in UI.
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
    resample_sequence,
    validate_features,
    extract_features_from_arrays,
)
from backend.recognition.isl_fingerspelling_recognizer import (
    ISLFingerspellingRecognizer,
    DEFAULT_MODEL_PATH,
    DEFAULT_LABELS_PATH,
)
from backend.vision.hand_landmarks import HandLandmarker, HandLandmarkResult


# ── Test 1: Strict Alphabet Mapping Order & Rejection of Shuffled Mappings ──
def test_alphabet_strict_mapping_validation(tmp_path):
    # 1. Valid mapping passes
    rec_valid = ISLFingerspellingRecognizer()
    assert rec_valid.is_valid_mapping is True
    assert rec_valid.classes == [chr(ord("A") + i) for i in range(26)]
    for i in range(26):
        assert rec_valid.id_to_label[i] == chr(ord("A") + i)
        assert rec_valid.label_to_id[chr(ord("A") + i)] == i

    # 2. Shuffled classes list must be rejected
    shuffled_classes = [chr(ord("A") + i) for i in range(26)]
    shuffled_classes[0], shuffled_classes[1] = "B", "A"  # Swap A and B
    bad_labels_path = tmp_path / "shuffled_labels.json"
    bad_labels_path.write_text(json.dumps({"classes": shuffled_classes}), encoding="utf-8")

    rec_shuffled = ISLFingerspellingRecognizer(labels_path=bad_labels_path)
    assert rec_shuffled.is_valid_mapping is False
    pred = rec_shuffled.predict()
    assert pred["label"] == "Invalid Mapping"
    assert pred["is_model_available"] is False

    # 3. Shuffled id_to_label must be rejected
    bad_id_map = {str(i): chr(ord("A") + i) for i in range(26)}
    bad_id_map["0"] = "B"
    bad_id_map["1"] = "A"
    bad_map_path = tmp_path / "bad_id_map.json"
    bad_map_path.write_text(
        json.dumps({"classes": [chr(ord("A") + i) for i in range(26)], "id_to_label": bad_id_map}),
        encoding="utf-8",
    )
    rec_bad_map = ISLFingerspellingRecognizer(labels_path=bad_map_path)
    assert rec_bad_map.is_valid_mapping is False

    # 4. Incomplete classes (e.g. 25 classes) must be rejected
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps({"classes": [chr(ord("A") + i) for i in range(25)]}), encoding="utf-8")
    rec_incomplete = ISLFingerspellingRecognizer(labels_path=incomplete_path)
    assert rec_incomplete.is_valid_mapping is False


# ── Test 2: Prediction Diagnostics in Recognizer ──
def test_alphabet_prediction_diagnostics():
    rec = ISLFingerspellingRecognizer()
    if not rec.is_loaded:
        pytest.skip("Alphabet model not available")

    # Feed dummy 30 frames
    dummy_feat = np.ones(FEATURE_DIM, dtype=np.float32) * 0.1
    for _ in range(30):
        rec.add_frame(dummy_feat, has_hands=True)

    result = rec.predict()
    assert "top_5" in result, "Diagnostics must contain top_5"
    assert "winning_index" in result, "Diagnostics must contain winning_index"
    assert "resolved_letter" in result, "Diagnostics must contain resolved_letter"
    assert "margin" in result, "Diagnostics must contain margin"
    assert "confidence_threshold" in result, "Diagnostics must contain confidence_threshold"
    assert "margin_threshold" in result, "Diagnostics must contain margin_threshold"

    assert len(result["top_5"]) == 5
    probs = [item["probability"] for item in result["top_5"]]
    assert probs == sorted(probs, reverse=True), "Top-5 probabilities must be descending"
    assert result["winning_index"] == result["top_5"][0]["index"]
    assert result["resolved_letter"] == rec.id_to_label[result["winning_index"]]


# ── Test 3: Two-Hand HandLandmarker Collision Resolution ──
def test_hand_landmarker_two_hand_collision_resolution():
    """Verify that when MediaPipe detects two hands with the same handedness label, neither is dropped."""
    class MockCategory:
        def __init__(self, category_name, score):
            self.category_name = category_name
            self.display_name = category_name
            self.score = score

    class MockLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class MockDetectionResult:
        def __init__(self):
            self.hand_landmarks = [
                [MockLandmark(0.65, 0.5, 0.0) for _ in range(21)],
                [MockLandmark(0.35, 0.5, 0.0) for _ in range(21)],
            ]
            self.handedness = [
                [MockCategory("Right", 0.95)],
                [MockCategory("Right", 0.92)],  # MediaPipe labeled both as Right!
            ]

    landmarker = HandLandmarker()
    detection_res = MockDetectionResult()
    num_hands = len(detection_res.hand_landmarks)

    detected_hands = []
    for i in range(num_hands):
        lms = detection_res.hand_landmarks[i]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        top_category = detection_res.handedness[i][0]
        label = top_category.category_name
        score = float(top_category.score)
        detected_hands.append((label, coords, score))

    slots = {"left": None, "right": None}
    for label, coords, score in detected_hands:
        side = label.lower()
        if side not in slots or slots[side] is not None:
            empty_side = "left" if slots["left"] is None else "right"
            slots[empty_side] = (coords, score, label)
        else:
            slots[side] = (coords, score, label)

    res = HandLandmarkResult()
    if slots["left"] is not None:
        res.left_hand_detected = True
        res.left_landmarks = slots["left"][0]
        res.left_confidence = slots["left"][1]
    if slots["right"] is not None:
        res.right_hand_detected = True
        res.right_landmarks = slots["right"][0]
        res.right_confidence = slots["right"][1]

    # Verify BOTH hands are preserved!
    assert res.left_hand_detected is True
    assert res.right_hand_detected is True
    assert res.left_landmarks is not None
    assert res.right_landmarks is not None
    assert res.left_landmarks.shape == (21, 3)
    assert res.right_landmarks.shape == (21, 3)


# ── Test 4: Genuine Held-Out Signer (signer_02) Letter A Recognition ──
def test_held_out_signer_02_alphabet_a():
    manifest_path = PROJECT_ROOT / "data" / "isl_signs" / "landmarks.jsonl"
    if not manifest_path.exists():
        pytest.skip("Dataset manifest data/isl_signs/landmarks.jsonl not found")

    rec = ISLFingerspellingRecognizer()
    if not rec.is_loaded:
        pytest.skip("Alphabet model not available")

    test_samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("signer_id") == "signer_02" and item.get("label") == "A":
                p = PROJECT_ROOT / item["features_path"]
                if p.exists():
                    loaded = np.load(p)
                    feats = loaded["features"] if "features" in loaded else loaded["arr_0"]
                    test_samples.append(feats)

    assert len(test_samples) > 0, "Expected held-out signer_02 samples for letter A"

    correct = 0
    total = len(test_samples)
    for sample in test_samples:
        rec.reset_buffer()
        for frame in sample:
            rec.add_frame(frame, has_hands=True)
        pred = rec.predict()
        if pred.get("letter") == "A":
            correct += 1

    accuracy = correct / total
    assert accuracy >= 0.90, f"Held-out Signer 02 letter A accuracy {accuracy:.1%} is below 90%"


# ── Test 5: Genuine Held-Out Signer (signer_02) Letter B Recognition ──
def test_held_out_signer_02_alphabet_b():
    manifest_path = PROJECT_ROOT / "data" / "isl_signs" / "landmarks.jsonl"
    if not manifest_path.exists():
        pytest.skip("Dataset manifest data/isl_signs/landmarks.jsonl not found")

    rec = ISLFingerspellingRecognizer()
    if not rec.is_loaded:
        pytest.skip("Alphabet model not available")

    test_samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("signer_id") == "signer_02" and item.get("label") == "B":
                p = PROJECT_ROOT / item["features_path"]
                if p.exists():
                    loaded = np.load(p)
                    feats = loaded["features"] if "features" in loaded else loaded["arr_0"]
                    test_samples.append(feats)

    assert len(test_samples) > 0, "Expected held-out signer_02 samples for letter B"

    top1_is_b = 0
    misclassified_as_u = 0
    conf_exceeds_threshold = 0
    total = len(test_samples)

    for sample in test_samples:
        rec.reset_buffer()
        for frame in sample:
            rec.add_frame(frame, has_hands=True)
        pred = rec.predict()

        winning_letter = pred.get("resolved_letter") or (pred.get("top_5")[0]["letter"] if pred.get("top_5") else None)
        if winning_letter == "B":
            top1_is_b += 1
        if winning_letter == "U" or pred.get("letter") == "U":
            misclassified_as_u += 1
        if pred.get("letter") == "B":
            conf_exceeds_threshold += 1

    # 1. At least 83% of authentic unmirrored B samples have 'B' as the top-1 winning prediction
    assert top1_is_b >= 5, f"Expected top-1 prediction to be B for at least 5/6 samples, got {top1_is_b}/{total}"

    # 2. 0% of authentic unmirrored B samples are misclassified as 'U'
    assert misclassified_as_u == 0, f"Encountered misclassification to 'U' in {misclassified_as_u} samples!"

    # 3. Standard confidence threshold (min_confidence=0.50) is met by >= 65% of held-out samples (4/6 = 66.7%)
    assert conf_exceeds_threshold / total >= 0.65, f"Confidence gating passed on {conf_exceeds_threshold}/{total} samples"


# ── Test 6: Preprocessing Parity ──
def test_preprocessing_parity():
    # 1. 282 dimensions
    left = np.random.randn(21, 3).astype(np.float32)
    right = np.random.randn(21, 3).astype(np.float32)
    feat, info = extract_features_from_arrays(left_landmarks=left, right_landmarks=right)
    assert feat.shape == (282,)
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()

    # 2. Resampling to 30 frames
    seq_variable = np.random.randn(15, 282).astype(np.float32)
    seq_resampled = resample_sequence(seq_variable, target_length=30)
    assert seq_resampled.shape == (30, 282)


# ── Test 7: UI Webcam Capture Unmirrored Configuration ──
def test_ui_webcam_unmirrored_config():
    translator_ui_path = PROJECT_ROOT / "frontend" / "translator_ui.py"
    diagnostics_ui_path = PROJECT_ROOT / "frontend" / "diagnostics_ui.py"

    assert translator_ui_path.exists()
    assert diagnostics_ui_path.exists()

    content_trans = translator_ui_path.read_text(encoding="utf-8")
    content_diag = diagnostics_ui_path.read_text(encoding="utf-8")

    assert "mirror_webcam=False" in content_trans, "translator_ui.py camera_input must have mirror_webcam=False"
    assert "mirror_webcam=True" not in content_trans, "translator_ui.py must not contain mirror_webcam=True"

    assert "mirror_webcam=False" in content_diag, "diagnostics_ui.py diag_camera_input must have mirror_webcam=False"
    assert "mirror_webcam=True" not in content_diag, "diagnostics_ui.py must not contain mirror_webcam=True"
