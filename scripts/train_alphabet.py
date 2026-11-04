"""
SignBridge AI — Dedicated Alphabet / Fingerspelling Model Training Script (A–Z).
Trains a Bidirectional LSTM classifier for Indian Sign Language English Alphabet (26 classes).
Strictly uses backend/vision/feature_schema.py (v2.0.0, 282 dimensions, 30 frames).
Partitions train/val/test splits strictly by signer ID for genuine out-of-sample evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

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


def main():
    parser = argparse.ArgumentParser(description="Train ISL Alphabet Fingerspelling Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    X, y, signers = load_dataset()
    if len(X) == 0:
        logger.error("No valid alphabet samples found in %s", DATA_MANIFEST)
        logger.info("Run `python scripts/import_fingerspelling.py` to download and extract dataset.")
        return

    # Check class coverage
    present_classes = sorted(set(y.tolist()))
    missing = [c for i, c in enumerate(ALPHABET_CLASSES) if i not in present_classes]
    logger.info("Present classes (%d/26): %s", len(present_classes), [ID_TO_LABEL[i] for i in present_classes])
    if missing:
        logger.warning("Missing classes (%d): %s", len(missing), missing)

    # Signer-independent split
    signers_arr = np.asarray(signers)
    test_signer = "signer_02" if "signer_02" in signers_arr else signers_arr[-1]
    test_mask = (signers_arr == test_signer)
    train_mask = ~test_mask

    if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
        # Fallback to random stratified split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Train set: %d samples, Test set (%s): %d samples", len(X_train), test_signer, len(X_test))

    model = build_alphabet_model(num_classes=len(ALPHABET_CLASSES))
    verify_model_compatibility(model, expected_dim=FEATURE_DIM, expected_seq_len=SEQUENCE_LENGTH)

    logger.info("Beginning training for %d epochs...", args.epochs)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test Accuracy on %s: %.2f%% (Loss: %.4f)", test_signer, test_acc * 100.0, test_loss)

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
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "test_samples": int(len(X_test)),
            "train_samples": int(len(X_train)),
            "test_signer": str(test_signer),
            "present_classes": [ID_TO_LABEL[i] for i in present_classes],
            "missing_classes": missing,
        }, f, indent=2)
    logger.info("Saved evaluation report to %s", EVAL_REPORT_PATH)


if __name__ == "__main__":
    main()
