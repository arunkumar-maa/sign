"""
SignBridge AI — Synthetic landmark generator & model trainer.
Generates biomechanically grounded landmark features and trains
both static MLP and dynamic LSTM models for complete end-to-end functionality.
"""

import json
import logging
import os
import sys
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Root setup
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STATIC_CLASSES = [
    # Alphabets A-Z
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # Numbers 1-5
    "1", "2", "3", "4", "5",
    # Common ISL core signs
    "HELLO", "THANK YOU", "HELP", "YES", "NO", "PLEASE", "WATER", "EMERGENCY"
]

DYNAMIC_CLASSES = [
    "HELLO", "THANK YOU", "HELP", "YES", "NO", "PLEASE", "WATER", "EMERGENCY"
]


def create_hand_pattern(sign: str) -> np.ndarray:
    """
    Generate normalized 21 hand landmarks (21 * 3 = 63 floats)
    based on the finger flexion/extension state for each sign.
    """
    lms = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    lms[0] = [0.5, 0.7, 0.0]

    # Finger MCP positions (bases)
    lms[1] = [0.42, 0.65, 0.0] # Thumb CMC
    lms[2] = [0.38, 0.58, 0.0] # Thumb MCP
    lms[5] = [0.44, 0.50, 0.0] # Index MCP
    lms[9] = [0.49, 0.48, 0.0] # Middle MCP
    lms[13] = [0.54, 0.50, 0.0] # Ring MCP
    lms[17] = [0.59, 0.54, 0.0] # Pinky MCP

    # Finger state rules (extended=1, folded=0)
    # Default open hand
    ext = [True, True, True, True, True]

    if sign == "A":
        # Fist with thumb up/beside
        ext = [True, False, False, False, False]
    elif sign == "B":
        # 4 fingers up, thumb folded
        ext = [False, True, True, True, True]
    elif sign == "C":
        # Curved hand
        ext = [False, False, False, False, False]
    elif sign == "D" or sign == "1":
        # Index pointing up
        ext = [False, True, False, False, False]
    elif sign == "V" or sign == "2" or sign == "NO":
        # Index and middle up
        ext = [False, True, True, False, False]
    elif sign == "W" or sign == "3":
        # Index, middle, ring up
        ext = [False, True, True, True, False]
    elif sign == "4":
        # Four fingers up
        ext = [False, True, True, True, True]
    elif sign == "5" or sign == "HELLO":
        # Open 5 fingers
        ext = [True, True, True, True, True]
    elif sign == "I" or sign == "Y":
        # Pinky extended (and thumb for Y)
        ext = [sign == "Y", False, False, False, True]
    elif sign == "YES":
        # Fist nod
        ext = [False, False, False, False, False]
    elif sign == "HELP":
        # Flat hand with thumb up
        ext = [True, True, True, True, True]
    elif sign in ("PLEASE", "THANK YOU", "WATER", "EMERGENCY"):
        # Distinct handshapes
        ext = [True, True, True, False, False]
    else:
        # Default distribution for other letters
        h = hash(sign) % 16
        ext = [bool(h & 1), bool(h & 2), bool(h & 4), bool(h & 8), False]

    # Generate joints for Thumb (1..4)
    if ext[0]:
        lms[3] = [0.34, 0.52, -0.02]
        lms[4] = [0.30, 0.46, -0.04]
    else:
        lms[3] = [0.42, 0.58, 0.02]
        lms[4] = [0.46, 0.58, 0.03]

    # Index (5..8)
    if ext[1]:
        lms[6] = [0.44, 0.38, -0.02]
        lms[7] = [0.44, 0.28, -0.03]
        lms[8] = [0.44, 0.18, -0.04]
    else:
        lms[6] = [0.44, 0.55, 0.03]
        lms[7] = [0.44, 0.60, 0.04]
        lms[8] = [0.44, 0.62, 0.05]

    # Middle (9..12)
    if ext[2]:
        lms[10] = [0.49, 0.35, -0.02]
        lms[11] = [0.49, 0.24, -0.03]
        lms[12] = [0.49, 0.14, -0.04]
    else:
        lms[10] = [0.49, 0.55, 0.03]
        lms[11] = [0.49, 0.60, 0.04]
        lms[12] = [0.49, 0.62, 0.05]

    # Ring (13..16)
    if ext[3]:
        lms[14] = [0.54, 0.38, -0.02]
        lms[15] = [0.54, 0.28, -0.03]
        lms[16] = [0.54, 0.19, -0.04]
    else:
        lms[14] = [0.54, 0.56, 0.03]
        lms[15] = [0.54, 0.61, 0.04]
        lms[16] = [0.54, 0.63, 0.05]

    # Pinky (17..20)
    if ext[4]:
        lms[18] = [0.59, 0.44, -0.02]
        lms[19] = [0.59, 0.36, -0.03]
        lms[20] = [0.59, 0.28, -0.04]
    else:
        lms[18] = [0.58, 0.58, 0.03]
        lms[19] = [0.58, 0.62, 0.04]
        lms[20] = [0.58, 0.64, 0.05]

    return lms.flatten() # 63 floats


def generate_static_samples(samples_per_class=120):
    """Generate 192-dimensional samples for static signs."""
    X = []
    y = []

    for class_idx, label in enumerate(STATIC_CLASSES):
        base_hand = create_hand_pattern(label)
        for _ in range(samples_per_class):
            # Apply slight translation, scale and jitter
            noise = np.random.normal(0, 0.012, size=base_hand.shape)
            scale = np.random.uniform(0.92, 1.08)
            right_hand = (base_hand + noise) * scale

            # Left hand (empty or present for two-handed signs like HELP/PLEASE)
            if label in ("HELP", "PLEASE", "EMERGENCY"):
                left_hand = right_hand * 0.9 + np.random.normal(0, 0.02, size=base_hand.shape)
            else:
                left_hand = np.zeros(63, dtype=np.float32)

            # Face (42) + Pose (24)
            face = np.random.normal(0.5, 0.02, size=42).astype(np.float32)
            pose = np.random.normal(0.5, 0.03, size=24).astype(np.float32)

            # Total = 63 + 63 + 42 + 24 = 192
            combined = np.concatenate([right_hand, left_hand, face, pose])
            X.append(combined)
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def generate_dynamic_samples(samples_per_class=80, seq_len=30):
    """Generate (seq_len, 192) sequences for dynamic signs."""
    X = []
    y = []

    for class_idx, label in enumerate(DYNAMIC_CLASSES):
        base_hand = create_hand_pattern(label)
        for _ in range(samples_per_class):
            seq = []
            # Trajectory modulation over 30 frames
            freq = np.random.uniform(1.0, 2.5)
            phase = np.random.uniform(0, np.pi)
            for t in range(seq_len):
                shift = 0.05 * np.sin(2 * np.pi * freq * (t / seq_len) + phase)
                noise = np.random.normal(0, 0.015, size=base_hand.shape)
                right_hand = base_hand + noise
                right_hand[0] += shift # wrist moves

                if label in ("HELP", "PLEASE", "EMERGENCY"):
                    left_hand = right_hand * 0.9 + np.random.normal(0, 0.02, size=base_hand.shape)
                else:
                    left_hand = np.zeros(63, dtype=np.float32)

                face = np.random.normal(0.5, 0.02, size=42).astype(np.float32)
                pose = np.random.normal(0.5, 0.03, size=24).astype(np.float32)
                pose[0] += shift * 0.3

                frame = np.concatenate([right_hand, left_hand, face, pose])
                seq.append(frame)
            X.append(seq)
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    import tensorflow as tf
    from ml.static_cnn import build_landmark_mlp
    from ml.dynamic_lstm import build_lstm_classifier
    from ml.evaluate import evaluate_model

    # 1. Paths
    labels_dir = ROOT / "models" / "labels"
    static_dir = ROOT / "models" / "static"
    dynamic_dir = ROOT / "models" / "dynamic"
    meta_dir = ROOT / "models" / "metadata"
    data_dir = ROOT / "data" / "processed" / "landmarks"

    for d in [labels_dir, static_dir, dynamic_dir, meta_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Save labels
    static_labels_path = labels_dir / "static_labels.json"
    with open(static_labels_path, "w", encoding="utf-8") as f:
        json.dump({"version": "1.0.0", "labels": STATIC_CLASSES, "num_classes": len(STATIC_CLASSES)}, f, indent=2)
    logger.info("Saved static labels to %s (%d classes)", static_labels_path, len(STATIC_CLASSES))

    dynamic_labels_path = labels_dir / "dynamic_labels.json"
    with open(dynamic_labels_path, "w", encoding="utf-8") as f:
        json.dump({"version": "1.0.0", "labels": DYNAMIC_CLASSES, "num_classes": len(DYNAMIC_CLASSES)}, f, indent=2)
    logger.info("Saved dynamic labels to %s (%d classes)", dynamic_labels_path, len(DYNAMIC_CLASSES))

    # 3. Generate Static Data
    logger.info("Generating static landmark dataset...")
    X_static, y_static = generate_static_samples(samples_per_class=100)
    indices = np.random.permutation(len(X_static))
    X_static, y_static = X_static[indices], y_static[indices]

    n_train = int(len(X_static) * 0.70)
    n_val = int(len(X_static) * 0.15)
    X_train, y_train = X_static[:n_train], y_static[:n_train]
    X_val, y_val = X_static[n_train:n_train+n_val], y_static[n_train:n_train+n_val]
    X_test, y_test = X_static[n_train+n_val:], y_static[n_train+n_val:]

    np.save(data_dir / "X_train.npy", X_train)
    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "X_val.npy", X_val)
    np.save(data_dir / "y_val.npy", y_val)
    np.save(data_dir / "X_test.npy", X_test)
    np.save(data_dir / "y_test.npy", y_test)
    logger.info("Saved static dataset splits to %s", data_dir)

    # 4. Train Static MLP
    logger.info("Training Static Landmark MLP (input_dim=192, classes=%d)...", len(STATIC_CLASSES))
    static_model = build_landmark_mlp(input_dim=192, num_classes=len(STATIC_CLASSES))
    static_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        verbose=1
    )

    static_model_path = static_dir / "isl_static_v1.keras"
    static_model.save(str(static_model_path))
    logger.info("Static model saved to %s", static_model_path)

    # Evaluate static
    static_eval = evaluate_model(static_model, X_test, y_test, STATIC_CLASSES, meta_dir / "static_evaluation_report.json")
    logger.info("Static Test Accuracy: %.2f%%, Mean F1: %.4f", static_eval["test_accuracy"] * 100, static_eval["weighted_f1"])

    # 5. Generate Dynamic Data & Train LSTM
    logger.info("Generating dynamic sequence dataset...")
    X_dyn, y_dyn = generate_dynamic_samples(samples_per_class=60, seq_len=30)
    d_indices = np.random.permutation(len(X_dyn))
    X_dyn, y_dyn = X_dyn[d_indices], y_dyn[d_indices]

    dn_train = int(len(X_dyn) * 0.70)
    dn_val = int(len(X_dyn) * 0.15)
    dX_train, dy_train = X_dyn[:dn_train], y_dyn[:dn_train]
    dX_val, dy_val = X_dyn[dn_train:dn_train+dn_val], y_dyn[dn_train:dn_train+dn_val]
    dX_test, dy_test = X_dyn[dn_train+dn_val:], y_dyn[dn_train+dn_val:]

    logger.info("Training Dynamic LSTM (seq_len=30, input_dim=192, classes=%d)...", len(DYNAMIC_CLASSES))
    dynamic_model = build_lstm_classifier(sequence_length=30, feature_dim=192, num_classes=len(DYNAMIC_CLASSES))
    dynamic_model.fit(
        dX_train, dy_train,
        validation_data=(dX_val, dy_val),
        epochs=25,
        batch_size=16,
        verbose=1
    )

    dynamic_model_path = dynamic_dir / "isl_dynamic_v1.keras"
    dynamic_model.save(str(dynamic_model_path))
    logger.info("Dynamic model saved to %s", dynamic_model_path)

    dyn_eval = evaluate_model(dynamic_model, dX_test, dy_test, DYNAMIC_CLASSES, meta_dir / "dynamic_evaluation_report.json")
    logger.info("Dynamic Test Accuracy: %.2f%%, Mean F1: %.4f", dyn_eval["test_accuracy"] * 100, dyn_eval["weighted_f1"])

    logger.info("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
