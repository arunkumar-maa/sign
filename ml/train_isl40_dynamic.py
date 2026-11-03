"""
SignBridge AI - Train & Fine-Tune Dynamic LSTM Model on ISL 40 Words Dataset
Preserves existing architecture and pretrained backbone weights.
Adapts output layer to 40 classes.
Evaluates thoroughly with accuracy, precision, recall, F1, confusion matrix, and per-class metrics.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_adapted_dynamic_model(
    sequence_length: int = 30,
    feature_dim: int = 192,
    num_classes: int = 40,
    existing_model_path: Path = None,
) -> "tf.keras.Model":
    """
    Builds the existing LSTM architecture adapted for num_classes,
    preserving backbone weights from existing_model_path if compatible.
    """
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(sequence_length, feature_dim), name="sequence")
    x = tf.keras.layers.LSTM(128, return_sequences=True, name="LSTM_1")(inputs)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False, name="LSTM_2")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="Dense_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ISL_Dynamic_LSTM_40Words")

    # Transfer compatible layer weights if existing model is available
    if existing_model_path and existing_model_path.exists():
        try:
            logger.info("Attempting backbone weight transfer from %s...", existing_model_path)
            old_model = tf.keras.models.load_model(str(existing_model_path))
            transferred = []
            for target_layer in model.layers:
                layer_name = target_layer.name
                if layer_name == "output" or not target_layer.get_weights():
                    continue  # skip output head since class dimension changed from 8 to 40
                
                target_shapes = [w.shape for w in target_layer.get_weights()]
                matched_layer = None

                # First try matching by exact name
                try:
                    candidate = old_model.get_layer(layer_name)
                    if [w.shape for w in candidate.get_weights()] == target_shapes:
                        matched_layer = candidate
                except Exception:
                    pass

                # If no name match, find first layer in old_model with matching shapes
                if matched_layer is None:
                    for old_l in old_model.layers:
                        if old_l.name != "output" and [w.shape for w in old_l.get_weights()] == target_shapes:
                            matched_layer = old_l
                            break

                if matched_layer is not None:
                    target_layer.set_weights(matched_layer.get_weights())
                    transferred.append(f"{target_layer.name} (from {matched_layer.name})")

            logger.info("Successfully transferred weights for layers: %s", transferred)
        except Exception as exc:
            logger.warning("Could not load weights from existing model: %s. Training from scratch.", exc)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ISL Dynamic Gesture Model on 40 Words")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    import tensorflow as tf
    from ml.evaluate import evaluate_model

    data_dir = PROJECT_ROOT / "data" / "processed" / "isl40"
    labels_path = PROJECT_ROOT / "models" / "labels" / "isl40_labels.json"
    output_dir = PROJECT_ROOT / "models" / "dynamic"
    metadata_dir = PROJECT_ROOT / "models" / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading processed dataset from %s...", data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_info = json.load(f)
    labels = labels_info["labels"]
    num_classes = len(labels)

    seq_len = X_train.shape[1]
    feat_dim = X_train.shape[2]
    logger.info("Dataset loaded: Train=%d, Val=%d, Test=%d | seq=%d, feat=%d, classes=%d",
                len(X_train), len(X_val), len(X_test), seq_len, feat_dim, num_classes)

    # 2. Compute balanced class weights
    unique_classes = np.unique(y_train)
    class_weights_arr = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
    class_weights = {int(cls): float(w) for cls, w in zip(unique_classes, class_weights_arr)}

    # 3. Build model with transferred backbone weights
    existing_ckpt = output_dir / "isl_dynamic_v1.keras"
    model = build_adapted_dynamic_model(
        sequence_length=seq_len,
        feature_dim=feat_dim,
        num_classes=num_classes,
        existing_model_path=existing_ckpt,
    )
    model.summary(print_fn=logger.info)

    # 4. Callbacks
    best_weights_path = output_dir / "temp_best_weights.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=16,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=6,
            factor=0.5,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    # 5. Train model
    logger.info("Beginning model training for %d epochs...", args.epochs)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # If best weights file was saved, load it
    if best_weights_path.exists():
        try:
            model = tf.keras.models.load_model(str(best_weights_path))
            logger.info("Loaded best checkpoint model.")
            best_weights_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not reload best checkpoint: %s", exc)

    # 6. Evaluation on Train, Val, and Test
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    logger.info("--- EVALUATION SUMMARY ---")
    logger.info("Train Accuracy: %.4f (Loss: %.4f)", train_acc, train_loss)
    logger.info("Val Accuracy:   %.4f (Loss: %.4f)", val_acc, val_loss)
    logger.info("Test Accuracy:  %.4f (Loss: %.4f)", test_acc, test_loss)

    report_path = metadata_dir / "isl40_evaluation_report.json"
    eval_report = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        output_path=report_path,
    )

    # Attach training and validation curves
    eval_report["train_accuracy"] = float(train_acc)
    eval_report["val_accuracy"] = float(val_acc)
    eval_report["epochs_trained"] = len(history.history["loss"])
    eval_report["history"] = {
        "loss": [float(v) for v in history.history["loss"]],
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved full evaluation report to %s", report_path)

    # 7. Save Model & Update Labels
    # Primary model path
    saved_model_path = output_dir / "isl_dynamic_40words.keras"
    model.save(str(saved_model_path))
    logger.info("Saved 40-class model to %s", saved_model_path)

    # Also update default model isl_dynamic_v1.keras for seamless backward compatibility
    compat_model_path = output_dir / "isl_dynamic_v1.keras"
    model.save(str(compat_model_path))
    logger.info("Updated backward-compatible model at %s", compat_model_path)

    # Update dynamic_labels.json
    dynamic_labels_path = PROJECT_ROOT / "models" / "labels" / "dynamic_labels.json"
    with open(dynamic_labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_info, f, indent=2)
    logger.info("Updated dynamic_labels.json with 40 classes at %s", dynamic_labels_path)

    # Also copy evaluation report to dynamic_evaluation_report.json
    compat_eval_path = metadata_dir / "dynamic_evaluation_report.json"
    with open(compat_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    logger.info("Dynamic model training and evaluation successfully finished!")


if __name__ == "__main__":
    main()
