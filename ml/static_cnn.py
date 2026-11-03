"""
Static gesture recognition models.
MLP on landmark features (primary) + CNN on 128x128 images (baseline).
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_landmark_mlp(input_dim: int, num_classes: int) -> "tf.keras.Model":
    """
    Build an MLP for landmark-based static gesture classification.
    
    Args:
        input_dim: Feature vector length (e.g. 192 for hand+face+pose).
        num_classes: Number of gesture classes.
    
    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="landmarks"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="ISL_Static_MLP")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("MLP model built: input=%d, classes=%d, params=%d",
                input_dim, num_classes, model.count_params())
    return model


def build_image_cnn(num_classes: int, image_size: Tuple[int, int] = (128, 128)) -> "tf.keras.Model":
    """
    Build a CNN for 128x128 image classification (baseline comparison).
    
    Args:
        num_classes: Number of gesture classes.
        image_size: (H, W) of input images.
    
    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*image_size, 3), name="image_input"),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="ISL_Static_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("CNN model built: classes=%d, params=%d", num_classes, model.count_params())
    return model

