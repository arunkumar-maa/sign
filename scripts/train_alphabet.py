"""
SignBridge AI — Dedicated Alphabet / Fingerspelling Model Training Script (A–Z).
Trains a Bidirectional LSTM classifier for Indian Sign Language English Alphabet (26 classes).
Strictly uses backend/vision/feature_schema.py (v2.0.0, 282 dimensions, 30 frames).
Enforces genuine signer-independent train, validation, and test splits without random clip fallbacks.
Enforces complete class coverage across all 26 alphabet classes (A–Z).
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision.feature_schema import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    resample_sequence,
    validate_features,
    verify_model_compatibility,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_alphabet")

ALPHABET_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
LABEL_TO_ID = {c: i for i, c in enumerate(ALPHABET_CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(ALPHABET_CLASSES)}

DATA_MANIFEST = PROJECT_ROOT / "data" / "isl_signs" / "landmarks.jsonl"
OUT_MODEL_PATH = PROJECT_ROOT / "models" / "isl_fingerspelling_model.keras"
CAT_MODEL_PATH = PROJECT_ROOT / "models" / "alphabet" / "hand_sequence_lstm.keras"
OUT_LABELS_PATH = PROJECT_ROOT / "models" / "fingerspelling_labels.json"
CAT_LABELS_PATH = PROJECT_ROOT / "models" / "alphabet" / "labels.json"
EVAL_REPORT_PATH = PROJECT_ROOT / "models" / "alphabet" / "evaluation.json"


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load alphabet feature sequences from manifest."""
    if not DATA_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {DATA_MANIFEST}")

    X_list, y_list, signers = [], [], []
    with open(DATA_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("category") != "alphabet":
                continue

            lbl = str(rec.get("label", "")).strip().upper()
            if lbl not in LABEL_TO_ID:
                continue

            feat_rel = rec.get("features_path")
            if not feat_rel:
                continue

            feat_path = PROJECT_ROOT / feat_rel
            if not feat_path.exists():
                continue

            try:
                raw_seq = np.load(feat_path)["features"].astype(np.float32)
                if raw_seq.ndim != 2 or raw_seq.shape[1] != FEATURE_DIM:
                    continue
                seq_30 = resample_sequence(raw_seq, target_length=SEQUENCE_LENGTH)
                X_list.append(seq_30)
                y_list.append(LABEL_TO_ID[lbl])
                signers.append(rec.get("signer_id", "unknown_signer"))
            except Exception as e:
                logger.debug("Failed loading %s: %s", feat_path, e)

    logger.info("Loaded %d alphabet samples across %d signers", len(X_list), len(set(signers)))
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.int32), signers


def build_alphabet_model(num_classes: int = 26):
    """Build canonical Bidirectional LSTM for fingerspelling classification."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_alphabet_model(epochs: int = 25, batch_size: int = 16) -> Dict[str, Any]:
    """
    Programmatic entry point for training the ISL Alphabet (A-Z) model.
    Enforces genuine signer-independent splits and complete 26-class coverage.
    """
    X, y, signers = load_dataset()
    if len(X) == 0:
        logger.error("No valid alphabet samples found in %s", DATA_MANIFEST)
        logger.info("Run `python scripts/import_fingerspelling.py` to download and extract dataset.")
        return {"status": "error", "message": "No valid alphabet samples found"}

    # Check unique signers: genuine signer-independent training requires >= 3 distinct signer IDs
    unique_signers = sorted(list(set(signers)))
    if len(unique_signers) < 3:
        raise ValueError(
            f"Insufficient signer IDs: genuine signer-independent training requires at least 3 distinct signer IDs "
            f"(found {len(unique_signers)}: {unique_signers}). Random train_test_split fallback is prohibited."
        )

    # Genuine 3-way signer-independent partitioning (Train, Validation, Test)
    signers_arr = np.asarray(signers)
    test_signer = "signer_02" if "signer_02" in unique_signers else unique_signers[-1]
    val_signer = "signer_01" if "signer_01" in unique_signers else [s for s in unique_signers if s != test_signer][-1]
    train_signers = [s for s in unique_signers if s != test_signer and s != val_signer]

    train_mask = np.isin(signers_arr, train_signers)
    val_mask = (signers_arr == val_signer)
    test_mask = (signers_arr == test_signer)

    if np.sum(train_mask) == 0:
        raise ValueError(f"No training samples for signers: {train_signers}")
    if np.sum(val_mask) == 0:
        raise ValueError(f"No validation samples for signer: {val_signer}")
    if np.sum(test_mask) == 0:
        raise ValueError(f"No test samples for signer: {test_signer}")

    # Enforce complete class coverage: all 26 classes A-Z must be present in training
    train_classes = set(y[train_mask].tolist())
    missing_train = [ID_TO_LABEL[c] for c in range(len(ALPHABET_CLASSES)) if c not in train_classes]
    if missing_train:
        err_msg = (
            f"Incomplete alphabet training coverage: classes {missing_train} are missing from the training partition "
            f"(signers: {train_signers}). All 26 classes (A-Z) must be present in training. "
            f"Collect additional training clips for missing classes using the data collection tool."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Inspect per-class test set representation
    test_classes = set(y[test_mask].tolist())
    missing_test = [ID_TO_LABEL[c] for c in range(len(ALPHABET_CLASSES)) if c not in test_classes]
    complete_test_coverage = (len(missing_test) == 0)

    per_class_test_samples: Dict[str, int] = {}
    for c in range(len(ALPHABET_CLASSES)):
        cnt = int(np.sum((y == c) & test_mask))
        per_class_test_samples[ID_TO_LABEL[c]] = cnt
        if cnt == 0:
            logger.warning("Alphabet class %s has 0 test samples on held-out signer %s!", ID_TO_LABEL[c], test_signer)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Signer-Independent Split: Train Signers %s (%d samples), Val Signer %s (%d samples), Test Signer %s (%d samples)",
                train_signers, len(X_train), val_signer, len(X_val), test_signer, len(X_test))

    model = build_alphabet_model(num_classes=len(ALPHABET_CLASSES))
    verify_model_compatibility(model, expected_dim=FEATURE_DIM, expected_seq_len=SEQUENCE_LENGTH)

    logger.info("Beginning training for %d epochs...", epochs)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Evaluate on held-out test signer
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Signer-Independent Test Accuracy on %s: %.2f%% (Loss: %.4f)", test_signer, test_acc * 100.0, test_loss)

    # Save models
    for out_p in [OUT_MODEL_PATH, CAT_MODEL_PATH]:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out_p))
        logger.info("Saved model to %s", out_p)

    # Save metadata
    labels_metadata = {
        "version": "2.0.0",
        "dataset": "kirandevraj/ISL-Fingerspelling",
        "category": "alphabet",
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "num_classes": len(ALPHABET_CLASSES),
        "classes": ALPHABET_CLASSES,
        "labels": ALPHABET_CLASSES,
        "label_to_index": LABEL_TO_ID,
        "class_to_id": LABEL_TO_ID,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
    }
    for lbl_p in [OUT_LABELS_PATH, CAT_LABELS_PATH]:
        lbl_p.parent.mkdir(parents=True, exist_ok=True)
        with open(lbl_p, "w", encoding="utf-8") as f:
            json.dump(labels_metadata, f, indent=2)
        logger.info("Saved labels to %s", lbl_p)

    # Save evaluation report
    eval_data = {
        "is_signer_independent": True,
        "split_type": "signer_independent",
        "complete_class_test_coverage": complete_test_coverage,
        "missing_test_classes": missing_test,
        "signer_split": {
            "train_signers": train_signers,
            "val_signers": [val_signer],
            "test_signers": [test_signer],
        },
        "sample_counts": {
            "total": int(len(X)),
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "per_class_test_samples": per_class_test_samples,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "present_train_classes": sorted(list(train_classes)),
        "epochs_completed": len(history.history.get("loss", [])),
    }
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    logger.info("Saved evaluation report to %s", EVAL_REPORT_PATH)

    return eval_data


def main():
    parser = argparse.ArgumentParser(description="Train ISL Alphabet Fingerspelling Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    train_alphabet_model(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
