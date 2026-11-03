"""
SignBridge AI — Train Dynamic ISL Hand Movement Recognizer.
Trains a Bidirectional LSTM network on 30-frame hand pose sequences (shape (N, 30, 216)).
Recognizes 40 Indian Sign Language isolated signs without pose or face models.
Saves model to: models/hand_pose/isl_dynamic_handpose.keras
"""

import json
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "hand_pose"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hand_pose"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 216


def build_dynamic_model(num_classes: int) -> tf.keras.Model:
    """Construct Bidirectional LSTM architecture for hand movement sequence recognition."""
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name="hand_pose_sequence")

    # Layer normalization on inputs
    x = layers.LayerNormalization(name="input_norm")(inputs)

    # First Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.15),
        name="bilstm_1",
    )(x)

    # Second Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.15),
        name="bilstm_2",
    )(x)

    # Dense classification layers
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.3, name="drop_1")(x)

    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.2, name="drop_2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="isl_word_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ISL_Dynamic_HandPose_LSTM")
    return model


def train_dynamic_model(epochs: int = 50, batch_size: int = 32):
    """Load extracted sequence data, train LSTM model, and save weights."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    labels_file = MODEL_DIR / "labels.json"
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found at: {labels_file}")

    with open(labels_file, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    num_classes = labels_meta["num_classes"]

    # Load sequence datasets
    logger.info("Loading dynamic sequence datasets from %s...", PROCESSED_DATASET_DIR)
    X_train = np.load(PROCESSED_DATASET_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DATASET_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DATASET_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DATASET_DIR / "y_val.npy")
    X_test = np.load(PROCESSED_DATASET_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DATASET_DIR / "y_test.npy")

    logger.info("Train shape: %s, Val shape: %s, Test shape: %s", X_train.shape, X_val.shape, X_test.shape)

    model = build_dynamic_model(num_classes=num_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    out_model_path = MODEL_DIR / "isl_dynamic_handpose.keras"

    call_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(str(out_model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    logger.info("Starting training for %d epochs...", epochs)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=call_list,
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test Results: Loss = %.4f, Accuracy = %.2f%%", test_loss, test_acc * 100.0)

    # Ensure saved
    model.save(str(out_model_path))
    logger.info("✅ Saved dynamic hand pose model to: %s", out_model_path)
    return model, history


if __name__ == "__main__":
    train_dynamic_model()

