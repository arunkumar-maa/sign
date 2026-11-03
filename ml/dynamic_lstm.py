"""
Dynamic gesture recognition model (LSTM/GRU sequence classifier).
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def build_lstm_classifier(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    use_gru: bool = False,
) -> "tf.keras.Model":
    """
    Build an LSTM (or GRU) sequence classifier for dynamic gestures.
    
    Args:
        sequence_length: Number of frames per sequence.
        feature_dim: Feature vector length per frame.
        num_classes: Number of gesture classes.
        use_gru: If True, use GRU instead of LSTM.
    
    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    
    RNN = tf.keras.layers.GRU if use_gru else tf.keras.layers.LSTM
    cell_type = "GRU" if use_gru else "LSTM"

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, feature_dim), name="sequence"),
        
        RNN(128, return_sequences=True, name=f"{cell_type}_1"),
        tf.keras.layers.Dropout(0.3),
        
        RNN(64, return_sequences=False, name=f"{cell_type}_2"),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name=f"ISL_Dynamic_{cell_type}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(
        "%s model built: seq=%d, features=%d, classes=%d, params=%d",
        cell_type, sequence_length, feature_dim, num_classes, model.count_params(),
    )
    return model

