"""
SignBridge AI — Phase 6: ISL Word Recognition Model Trainer.
Trains a Bidirectional LSTM / GRU sequence classifier on 192-dim landmark sequences.
Saves:
- models/isl_word_model.keras
- models/word_labels.json
Evaluates on the held-out test split and generates a confusion matrix and classification report.
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "isl40"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "isl_word_model.keras"
LABELS_SAVE_PATH = MODELS_DIR / "word_labels.json"
REPORT_SAVE_PATH = MODELS_DIR / "metadata" / "word_model_evaluation_report.json"


def build_word_lstm_model(
    sequence_length: int = 30,
    feature_dim: int = 192,
    num_classes: int = 40,
) -> tf.keras.Model:
    """
    Construct Bidirectional LSTM sequence classifier with SpatialDropout
    and BatchNormalization to prevent overfitting on landmark trajectories.
    """
    inputs = tf.keras.layers.Input(shape=(sequence_length, feature_dim), name="landmark_sequence")
    
    # Layer 1: BiLSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Layer 2: BiLSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.1)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Dense classification head
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="word_prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ISL_Word_BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_or_load_word_model(epochs: int = 25, batch_size: int = 32) -> Tuple[tf.keras.Model, Dict]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load arrays
    logger.info("Loading training data from %s...", DATA_DIR)
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    # Load label mapping
    labels_file = PROJECT_ROOT / "models" / "labels" / "isl40_labels.json"
    with open(labels_file, "r", encoding="utf-8") as f:
        lbl_data = json.load(f)
    labels = lbl_data.get("labels", [])
    num_classes = len(labels)

    # Save models/word_labels.json
    word_labels_manifest = {
        "version": "1.0.0",
        "dataset": "vidit031/isl-isolated-40words",
        "num_classes": num_classes,
        "labels": labels,
        "class_to_id": {lbl: i for i, lbl in enumerate(labels)},
    }
    with open(LABELS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(word_labels_manifest, f, indent=2)
    logger.info("Saved word labels to %s (%d classes)", LABELS_SAVE_PATH, num_classes)

    # Check if a trained dynamic model is already saved
    existing_model = PROJECT_ROOT / "models" / "dynamic" / "isl_dynamic_40words.keras"
    if existing_model.exists() and not MODEL_SAVE_PATH.exists():
        logger.info("Found existing verified dynamic model at %s. Copying to %s...", existing_model, MODEL_SAVE_PATH)
        shutil.copy2(existing_model, MODEL_SAVE_PATH)

    if MODEL_SAVE_PATH.exists():
        logger.info("Loading word model from %s...", MODEL_SAVE_PATH)
        model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))
    else:
        logger.info("Training new BiLSTM model for %d epochs...", epochs)
        model = build_word_lstm_model(
            sequence_length=X_train.shape[1],
            feature_dim=X_train.shape[2],
            num_classes=num_classes,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        model.save(str(MODEL_SAVE_PATH))
        logger.info("Trained model saved to %s", MODEL_SAVE_PATH)

    # Run rigorous evaluation on held-out test split
    logger.info("Running evaluation on test split (%d samples)...", len(X_test))
    report = evaluate_model(model, X_test, y_test, labels, output_path=REPORT_SAVE_PATH)

    logger.info(
        "Evaluation results: Accuracy=%.2f%%, F1=%.2f%%, Latency=%.2f ms/sample",
        report["test_accuracy"] * 100.0,
        report["weighted_f1"] * 100.0,
        report["avg_inference_latency_ms"],
    )
    return model, report


def main():
    train_or_load_word_model()


if __name__ == "__main__":
    main()

