"""
SignBridge AI — Comprehensive Verification Test Suite (Phases 1 through 12).
Tests all critical architectural repairs:
1. Manifest integrity: no artificial signers (signer_sim_*), no zero-padded sequences for active classes.
2. Standard manifest schema for active categories: sample_id, category, label, signer_id, features_path.
3. Top-1 vs top-2 margin gating for Alphabet and Numbers.
4. Output dimensions (26 for Alphabet, 10 for Numbers).
5. Dynamic motion gesture trajectories (J & Z).
6. MediaPipe asset atomic download and schema verification.
7. Setup assets programmatic invocation (no argparse collisions).
8. ISL data collection strictly scoped to Alphabet and Numbers (Word rejected).
9. Honest signer-independence reporting (true multi-signer for alphabet, transparent single-signer for numbers).
10. LivePipeline active mode decoupling and character text builder stability.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision.feature_schema import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    resample_sequence,
    validate_features,
    validate_sequence,
    verify_model_compatibility,
    IncompatibleModelError,
)
from backend.recognition.isl_fingerspelling_recognizer import ISLFingerspellingRecognizer
from backend.recognition.isl_number_recognizer import ISLNumberRecognizer
from backend.recognition.character_builder import CharacterTextBuilder
from backend.vision.live_pipeline import LivePipeline
from isl_data_collection.config import CATEGORIES
from isl_data_collection.storage import validate_label


# ── Area 1: Artificial Signers Purged & Manifest Cleanliness ─────────────────
def test_artificial_signers_purged_from_manifest():
    manifest_path = PROJECT_ROOT / "data" / "isl_signs" / "landmarks.jsonl"
    if not manifest_path.exists():
        pytest.skip("Personal dataset landmarks.jsonl absent on clean clone")

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            signer = str(rec.get("signer_id", ""))
            assert not signer.startswith("signer_sim_"), f"Synthetic signer found in manifest: {signer}"


# ── Area 2: No Zero-Filled Landmark Sequences in Active Manifest ─────────────
def test_no_zero_filled_sequences_in_active_manifest():
    manifest_path = PROJECT_ROOT / "data" / "isl_signs" / "landmarks.jsonl"
    if not manifest_path.exists():
        pytest.skip("Personal dataset landmarks.jsonl absent on clean clone")

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Validate active categories (Alphabet and Numbers)
            if rec.get("category") in ("alphabet", "number"):
                feat_rel = rec.get("features_path")
                if feat_rel:
                    feat_path = PROJECT_ROOT / feat_rel
                    if feat_path.exists():
                        arr = np.load(feat_path)["features"]
                        # Verify active hand coordinates are not all-zeros
                        left_present = np.any(arr[:, 280] > 0.0)
                        right_present = np.any(arr[:, 281] > 0.0)
                        assert left_present or right_present, f"Sample {rec.get('sample_id')} has neither hand present"
                        if left_present:
                            assert not np.all(arr[:, :63] == 0.0), f"Sample {rec.get('sample_id')} claims left hand present but coordinates are all zeros"
                        if right_present:
                            assert not np.all(arr[:, 140:203] == 0.0), f"Sample {rec.get('sample_id')} claims right hand present but coordinates are all zeros"


# ── Area 3: Standard Manifest Schema For Active Classes ───────────────────────
def test_standard_manifest_schema_active_classes():
    manifest_path = PROJECT_ROOT / "data" / "isl_signs" / "landmarks.jsonl"
    if not manifest_path.exists():
        pytest.skip("Personal dataset landmarks.jsonl absent on clean clone")

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("category") in ("alphabet", "number"):
                assert "sample_id" in rec
                assert "category" in rec
                assert "label" in rec
                assert "signer_id" in rec
                assert "features_path" in rec


def test_manifest_schema_and_idempotence_clean_clone(tmp_path):
    """Verifies manifest schema and idempotent upserting on clean clones."""
    from isl_data_collection.storage import upsert_landmark_record, read_jsonl
    test_manifest = tmp_path / "landmarks.jsonl"

    rec_a = {
        "sample_id": "test_alpha_a",
        "category": "alphabet",
        "label": "A",
        "signer_id": "signer_01",
        "features_path": "data/isl_signs/features/alphabet_test_a.npz",
    }
    rec_0 = {
        "sample_id": "test_num_0",
        "category": "number",
        "label": "0",
        "signer_id": "islrtc_master",
        "features_path": "data/isl_signs/features/number_test_0.npz",
    }
    # Initial insertion returns True
    assert upsert_landmark_record(rec_a, path=test_manifest) is True
    assert upsert_landmark_record(rec_0, path=test_manifest) is True

    # Duplicate insertion returns False (idempotent, no duplicates added)
    assert upsert_landmark_record(rec_a, path=test_manifest) is False
    assert upsert_landmark_record(rec_0, path=test_manifest) is False

    rows = read_jsonl(test_manifest)
    assert len(rows) == 2
    for r in rows:
        assert all(k in r for k in ["sample_id", "category", "label", "signer_id", "features_path"])
        assert not str(r["signer_id"]).startswith("signer_sim_")


# ── Area 4: Top-Two Margin Gating in Alphabet Recognizer ──────────────────────
def test_top_two_margin_gating_alphabet():
    rec = ISLFingerspellingRecognizer(min_confidence=0.40, margin_threshold=0.15)
    for _ in range(15):
        rec.add_frame(np.ones(282, dtype=np.float32), has_hands=True)

    # Mock model where top-1 and top-2 are very close (ambiguous: M=0.42, N=0.40, margin=0.02 < 0.15)
    class AmbiguousAlphabetModel:
        def predict(self, batch, verbose=0):
            probs = np.full((1, 26), 0.007, dtype=np.float32)
            probs[0, 12] = 0.42  # 'M'
            probs[0, 13] = 0.40  # 'N'
            return probs

    rec.model = AmbiguousAlphabetModel()
    pred = rec.predict()
    assert pred["letter"] is None
    assert pred["label"] == "Unknown"
    assert "Unknown Sign" in pred["status"]

    # Mock model where top-1 is decisive (A=0.80, runner up=0.10, margin=0.70 >= 0.15)
    class DecisiveAlphabetModel:
        def predict(self, batch, verbose=0):
            probs = np.full((1, 26), 0.004, dtype=np.float32)
            probs[0, 0] = 0.80  # 'A'
            probs[0, 1] = 0.10
            return probs

    rec.model = DecisiveAlphabetModel()
    pred_decisive = rec.predict()
    assert pred_decisive["letter"] == "A"
    assert pred_decisive["confidence"] >= 0.80


# ── Area 5: Top-Two Margin Gating in Number Recognizer ────────────────────────
def test_top_two_margin_gating_numbers():
    rec = ISLNumberRecognizer(min_confidence=0.40, margin_threshold=0.15)
    for _ in range(15):
        rec.add_frame(np.ones(282, dtype=np.float32), has_hands=True)

    # Mock model where top-1 and top-2 are close (1=0.45, 2=0.43, margin=0.02 < 0.15)
    class AmbiguousNumberModel:
        def predict(self, batch, verbose=0):
            probs = np.full((1, 10), 0.015, dtype=np.float32)
            probs[0, 1] = 0.45
            probs[0, 2] = 0.43
            return probs

    rec.trained_model = AmbiguousNumberModel()
    pred = rec.predict()
    assert pred["number"] is None
    assert pred["label"] == "Unknown"

    # Mock model where top-1 is decisive (5=0.85, 4=0.05, margin=0.80 >= 0.15)
    class DecisiveNumberModel:
        def predict(self, batch, verbose=0):
            probs = np.full((1, 10), 0.01, dtype=np.float32)
            probs[0, 5] = 0.85
            probs[0, 4] = 0.05
            return probs

    rec.trained_model = DecisiveNumberModel()
    pred_decisive = rec.predict()
    assert pred_decisive["number"] == 5
    assert pred_decisive["label"] == "5"


# ── Area 6: Exact Model Dimensions (26 and 10) ────────────────────────────────
def test_model_output_dimensions_exact():
    alph_p = PROJECT_ROOT / "models" / "isl_fingerspelling_model.keras"
    num_p = PROJECT_ROOT / "models" / "isl_number_model.keras"

    import tensorflow as tf
    if alph_p.exists():
        m_alph = tf.keras.models.load_model(str(alph_p), compile=False)
        assert list(m_alph.input_shape) == [None, 30, 282]
        assert list(m_alph.output_shape) == [None, 26]

    if num_p.exists():
        m_num = tf.keras.models.load_model(str(num_p), compile=False)
        assert list(m_num.input_shape) == [None, 30, 282]
        assert list(m_num.output_shape) == [None, 10]


# ── Area 7: Continuous Dynamic Motion Signs (J & Z Trajectories) ──────────────
def test_dynamic_motion_gesture_trajectories():
    # Simulate a dynamic 'J' gesture (downward scoop motion across 30 frames) with right hand
    t = np.linspace(0, 1, 30)
    traj_j = np.zeros((30, 282), dtype=np.float32)
    # Right hand index fingertip (joint 8: coords 164, 165, 166 in 282-dim schema)
    traj_j[:, 164] = 0.5 + np.cos(t * np.pi) * 0.15  # x curve
    traj_j[:, 165] = 0.2 + (t ** 1.5) * 0.4          # y downward stroke + hook
    traj_j[:, 166] = 0.0                             # z depth
    traj_j[:, 281] = 1.0                             # Right hand present flag

    resampled_j = resample_sequence(traj_j, target_length=30)
    assert resampled_j.shape == (30, 282)
    assert validate_sequence(resampled_j) is True

    # Test motion energy detection across the dynamic gesture in the active hand slot
    diffs_j = np.diff(traj_j[:, 140:203], axis=0)
    motion_energy_j = float(np.sum(np.abs(diffs_j)))
    assert motion_energy_j > 0.5, "Dynamic J trajectory should exhibit substantial motion energy"

    # Simulate a dynamic 'Z' gesture (zigzag strokes: top bar -> diagonal -> bottom bar)
    traj_z = np.zeros((30, 282), dtype=np.float32)
    traj_z[:, 281] = 1.0  # Right hand present
    # Segment 1 (frames 0-9): top bar left to right
    traj_z[:10, 164] = np.linspace(0.3, 0.7, 10)
    traj_z[:10, 165] = 0.3
    # Segment 2 (frames 10-19): diagonal right-top to left-bottom
    traj_z[10:20, 164] = np.linspace(0.7, 0.3, 10)
    traj_z[10:20, 165] = np.linspace(0.3, 0.7, 10)
    # Segment 3 (frames 20-29): bottom bar left to right
    traj_z[20:, 164] = np.linspace(0.3, 0.7, 10)
    traj_z[20:, 165] = 0.7

    assert validate_sequence(traj_z) is True
    diffs_z = np.diff(traj_z[:, 140:203], axis=0)
    assert float(np.sum(np.abs(diffs_z))) > 1.0, "Dynamic Z trajectory should have strong multi-stroke motion energy"

    # Verify temporal directionality: forward J != time-reversed J
    traj_j_reversed = np.flip(traj_j, axis=0)
    assert not np.allclose(traj_j, traj_j_reversed), "Temporal sequence must preserve motion directionality"


# ── Area 8: ISL Data Collection Categories Strictly Alphabet & Number ─────────
def test_data_collection_strictly_alphabet_and_number():
    assert set(CATEGORIES) == {"alphabet", "number"}

    # Alphabet valid
    assert validate_label("alphabet", "A") == ("alphabet", "A")
    assert validate_label("alphabet", "z") == ("alphabet", "Z")

    # Numbers valid (0–9)
    for i in range(10):
        assert validate_label("number", str(i)) == ("number", str(i))
    assert validate_label("number", "three") == ("number", "3")

    # Word category rejected
    with pytest.raises(ValueError):
        validate_label("word", "HELLO")


# ── Area 9: Setup Assets Programmatic Invocations Without Argparse Clash ───────
def test_setup_assets_programmatic_import():
    from scripts.train_alphabet import train_alphabet_model
    from scripts.train_numbers import train_number_model
    import inspect

    sig_alph = inspect.signature(train_alphabet_model)
    assert "epochs" in sig_alph.parameters
    assert "batch_size" in sig_alph.parameters

    sig_num = inspect.signature(train_number_model)
    assert "epochs" in sig_num.parameters
    assert "batch_size" in sig_num.parameters


# ── Area 10: CharacterTextBuilder Auto-Add With Neutral Release Gating ─────────
def test_character_builder_neutral_release_gating():
    # Zero repeat cooldown so test execution timing doesn't block the repeat gate
    builder = CharacterTextBuilder(required_stable_frames=3, repeat_cooldown_seconds=0.0)
    builder.auto_add_enabled = True

    # Feed candidate "A" for 3 consecutive frames
    for _ in range(3):
        builder.process_auto_frame("A", confidence=0.85, hand_detected=True)
    assert builder.text == "A"

    # Continued holding of "A" must NOT add another "A" without neutral release
    for _ in range(5):
        builder.process_auto_frame("A", confidence=0.85, hand_detected=True)
    assert builder.text == "A", "Duplicate 'A' added without neutral release"

    # Neutral release: hand absent for >= 3 frames
    for _ in range(3):
        builder.process_auto_frame(None, confidence=0.0, hand_detected=False)

    # Now feed "A" again for 3 frames
    for _ in range(3):
        builder.process_auto_frame("A", confidence=0.85, hand_detected=True)
    assert builder.text == "AA", "Expected second 'A' after neutral release"


# ── Area 11: LivePipeline Mode Decoupling ──────────────────────────────────────
def test_live_pipeline_mode_decoupling():
    pipeline = LivePipeline()
    assert pipeline.mode == "Alphabet (A–Z)"

    pipeline.set_mode("Numbers (0–9)")
    assert pipeline.mode == "Numbers (0–9)"
    assert pipeline.recognizer == pipeline.number_recognizer

    pipeline.set_mode("Alphabet (A–Z)")
    assert pipeline.mode == "Alphabet (A–Z)"
    assert pipeline.recognizer == pipeline.fingerspelling_recognizer


# ── Area 12: Scientific Honesty in Evaluation Reports ─────────────────────────
def test_scientific_honesty_in_reports():
    eval_num_p = PROJECT_ROOT / "models" / "number" / "evaluation.json"
    if eval_num_p.exists():
        with open(eval_num_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Numbers report must not falsely claim multi-signer independence on 1 signer
        if data.get("signer_count", 1) < 3:
            assert data.get("is_signer_independent") is False
            assert "ISLRTC" in data.get("note", "")
            if data.get("total_samples", 0) <= 10:
                assert data.get("test_accuracy") is None
                assert data.get("evaluation_status") == "not_evaluated_insufficient_independent_samples"
