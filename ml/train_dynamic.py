"""
Training script for the dynamic gesture LSTM/GRU model.
Run this script directly to train on prepared landmark sequences.

Usage:
  .venv/Scripts/python.exe ml/train_dynamic.py \
    --data_dir data/processed/sequences \
    --output_dir models/dynamic \
    --epochs 80 --use_gru
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Train ISL Dynamic Gesture LSTM/GRU")
    parser.add_argument("--data_dir", type=str, default="data/processed/sequences")
    parser.add_argument("--output_dir", type=str, default="models/dynamic")
    parser.add_argument("--labels_path", type=str, default="models/labels/dynamic_labels.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="isl_dynamic_v1")
    parser.add_argument("--use_gru", action="store_true", help="Use GRU instead of LSTM")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    with open(args.labels_path, "r") as f:
        label_data = json.load(f)
    labels = label_data.get("labels", label_data) if isinstance(label_data, dict) else label_data

    sequence_length = X_train.shape[1]
    feature_dim = X_train.shape[2]
    num_classes = len(labels)
    logger.info("Data: %d train, %d val, %d test | seq=%d, feat=%d, classes=%d",
                len(X_train), len(X_val), len(X_test), sequence_length, feature_dim, num_classes)

    import tensorflow as tf
    from ml.dynamic_lstm import build_lstm_classifier
    from ml.evaluate import evaluate_model

    model = build_lstm_classifier(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        num_classes=num_classes,
        use_gru=args.use_gru,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, verbose=1),
    ]

    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    eval_report = evaluate_model(
        model, X_test, y_test, labels,
        output_path=Path("models/metadata/evaluation_report_dynamic.json"),
    )

    model_path = output_dir / f"{args.model_name}.keras"
    model.save(str(model_path))
    logger.info("Model saved to %s", model_path)

    import time
    metadata = {
        "model_name": args.model_name,
        "model_type": "GRU_sequence" if args.use_gru else "LSTM_sequence",
        "version": "1.0",
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sequence_length": sequence_length,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "labels": labels,
        "confidence_threshold": 0.70,
        "epochs_trained": len(history.history["loss"]),
        "test_metrics": eval_report,
    }
    meta_path = Path("models/metadata") / f"{args.model_name}_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Training complete. Test accuracy: %.4f", eval_report["test_accuracy"])


if __name__ == "__main__":
    main()

