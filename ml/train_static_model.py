"""
SignBridge AI — Train Static ISL Hand Pose Recognizer.
Trains a Dense Neural Network on single-frame hand pose features (shape (N, 216)).
Recognizes 40 Indian Sign Language signs based on stable handshape, angles, and distances.
Saves model to: models/hand_pose/isl_static_handpose.keras
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
FEATURE_DIM = 216


def build_static_model(num_classes: int) -> tf.keras.Model:
    """Construct MLP architecture for static handshape recognition."""
    inputs = layers.Input(shape=(FEATURE_DIM,), name="hand_pose_features")

    x = layers.Dense(256, activation="relu", name="dense_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.3, name="drop_1")(x)

    x = layers.Dense(128, activation="relu", name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.25, name="drop_2")(x)

    x = layers.Dense(64, activation="relu", name="dense_3")(x)
    x = layers.Dropout(0.2, name="drop_3")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ISL_Static_HandPose_MLP")
    return model


def train_static_model(epochs: int = 50, batch_size: int = 32):
    """Load extracted static keyframe features, train model, and save weights."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    labels_file = MODEL_DIR / "labels.json"
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found at: {labels_file}")

    with open(labels_file, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    num_classes = labels_meta["num_classes"]

    # Load static datasets
    logger.info("Loading static keyframe datasets from %s...", PROCESSED_DATASET_DIR)
    X_train = np.load(PROCESSED_DATASET_DIR / "static_X_train.npy")
    y_train = np.load(PROCESSED_DATASET_DIR / "static_y_train.npy")
    X_val = np.load(PROCESSED_DATASET_DIR / "static_X_val.npy")
    y_val = np.load(PROCESSED_DATASET_DIR / "static_y_val.npy")
    X_test = np.load(PROCESSED_DATASET_DIR / "static_X_test.npy")
    y_test = np.load(PROCESSED_DATASET_DIR / "static_y_test.npy")

    logger.info("Static Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape)

    model = build_static_model(num_classes=num_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    out_model_path = MODEL_DIR / "isl_static_handpose.keras"

    call_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(str(out_model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    logger.info("Starting static training for %d epochs...", epochs)
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
    logger.info("Static Test Results: Loss = %.4f, Accuracy = %.2f%%", test_loss, test_acc * 100.0)

    # Ensure saved
    model.save(str(out_model_path))
    logger.info("✅ Saved static hand pose model to: %s", out_model_path)
    return model, history


if __name__ == "__main__":
    train_static_model()

