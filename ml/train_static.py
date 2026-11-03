"""
Training script for the static gesture MLP model.
Run this script directly to train on prepared landmark features.

Usage:
  .venv/Scripts/python.exe ml/train_static.py \
    --data_dir data/processed/landmarks \
    --output_dir models/static \
    --epochs 50 --batch_size 32
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Train ISL Static Gesture MLP")
    parser.add_argument("--data_dir", type=str, default="data/processed/landmarks",
                        help="Directory containing X_train.npy, y_train.npy, etc.")
    parser.add_argument("--output_dir", type=str, default="models/static")
    parser.add_argument("--labels_path", type=str, default="models/labels/static_labels.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="isl_static_v1")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    with open(args.labels_path, "r") as f:
        label_data = json.load(f)
    labels = label_data.get("labels", label_data) if isinstance(label_data, dict) else label_data

    num_classes = len(labels)
    feature_dim = X_train.shape[1]
    logger.info("Data loaded: %d train, %d val, %d test, %d features, %d classes",
                len(X_train), len(X_val), len(X_test), feature_dim, num_classes)

    import tensorflow as tf
    from ml.static_cnn import build_landmark_mlp
    from ml.evaluate import evaluate_model

    model = build_landmark_mlp(input_dim=feature_dim, num_classes=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
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

    # Evaluate on test set
    eval_report = evaluate_model(
        model, X_test, y_test, labels,
        output_path=Path("models/metadata/evaluation_report_static.json"),
    )

    # Save model
    model_path = output_dir / f"{args.model_name}.keras"
    model.save(str(model_path))
    logger.info("Model saved to %s", model_path)

    # Save metadata
    import time
    metadata = {
        "model_name": args.model_name,
        "model_type": "MLP_landmark",
        "version": "1.0",
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "labels": labels,
        "confidence_threshold": 0.75,
        "epochs_trained": len(history.history["loss"]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_metrics": eval_report,
    }
    meta_path = Path("models/metadata") / f"{args.model_name}_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)
    logger.info("Training complete. Test accuracy: %.4f", eval_report["test_accuracy"])


if __name__ == "__main__":
    main()

