import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from .config import FEATURE_DIM, LANDMARK_MANIFEST, MODEL_DIR, SEQUENCE_LENGTH
from .storage import read_jsonl


def _resample_sequence(sequence: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [frames, {FEATURE_DIM}] feature array; received {sequence.shape}.")
    if len(sequence) == 0:
        raise ValueError("Empty feature sequence.")
    positions = np.linspace(0, len(sequence) - 1, target_length)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    weight = (positions - left)[:, None].astype(np.float32)
    return (sequence[left] * (1.0 - weight) + sequence[right] * weight).astype(np.float32)


def _load_category(category: str):
    records = [r for r in read_jsonl(LANDMARK_MANIFEST) if r.get("category") == category]
    X, labels, groups = [], [], []
    for record in records:
        feature_path = Path(record["features_path"])
        if not feature_path.is_absolute():
            feature_path = LANDMARK_MANIFEST.parent.parent / feature_path
        if not feature_path.exists():
            continue
        with np.load(feature_path) as content:
            sequence = content["features"].astype(np.float32)
        X.append(_resample_sequence(sequence))
        labels.append(str(record["label"]))
        groups.append(str(record.get("signer_id", "UNKNOWN")))
    return np.asarray(X, dtype=np.float32), np.asarray(labels), np.asarray(groups)


def _split_by_signer(X, y, groups):
    unique_signers = np.unique(groups)
    if len(unique_signers) < 3:
        raise ValueError(
            f"Need at least 3 different signer IDs for a signer-independent train/validation/test split; "
            f"found {len(unique_signers)}. Record this category with more signers."
        )
    first = GroupShuffleSplit(n_splits=30, test_size=0.2, random_state=42)
    split1 = None
    for train_val_idx, test_idx in first.split(X, y, groups):
        if len(np.unique(y[train_val_idx])) >= 2 and len(np.unique(y[test_idx])) >= 1:
            split1 = (train_val_idx, test_idx)
            break
    if split1 is None:
        raise ValueError("Could not create a signer-independent test split. Record more examples per class and signer.")
    train_val_idx, test_idx = split1
    second = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=43)
    split2 = None
    for train_local, val_local in second.split(X[train_val_idx], y[train_val_idx], groups[train_val_idx]):
        train_idx = train_val_idx[train_local]
        val_idx = train_val_idx[val_local]
        if len(np.unique(y[train_idx])) >= 2:
            split2 = (train_idx, val_idx)
            break
    if split2 is None:
        raise ValueError("Could not create a signer-independent validation split. Add more signers.")
    return split2[0], split2[1], test_idx


def train_category(category: str, epochs: int = 30, batch_size: int = 8) -> str:
    category = category.strip().lower()
    if category not in {"word", "alphabet", "number"}:
        raise ValueError("Category must be word, alphabet or number.")
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not available in this Python environment. Use Python 3.12 and install requirements.txt."
        ) from exc

    X, labels, groups = _load_category(category)
    classes = sorted(np.unique(labels).tolist())
    if len(X) < 12 or len(classes) < 2:
        raise ValueError(
            f"Not enough data for {category}: found {len(X)} clips and {len(classes)} classes. "
            "Record more labelled examples first."
        )
    for label in classes:
        count = int(np.sum(labels == label))
        if count < 3:
            raise ValueError(f"Class {label} has only {count} clips. Record at least 3 clips per class, preferably many more.")

    train_idx, val_idx, test_idx = _split_by_signer(X, labels, groups)
    train_labels = sorted(np.unique(labels[train_idx]).tolist())
    if train_labels != classes:
        raise ValueError(
            "Training split does not contain every class. Add more examples from multiple signers for each label."
        )
    label_to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        tf.keras.layers.LSTM(96, return_sequences=True),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = model.fit(
        X[train_idx], y[train_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=max(1, min(int(epochs), 200)),
        batch_size=max(1, min(int(batch_size), 64)),
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(X[test_idx], verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    test_true = y[test_idx]
    report = classification_report(
        test_true, predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_true, predictions, labels=list(range(len(classes))))

    category_dir = MODEL_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    model_path = category_dir / "hand_sequence_lstm.keras"
    model.save(model_path)
    (category_dir / "labels.json").write_text(
        json.dumps({"classes": classes, "label_to_index": label_to_index,
                    "feature_dim": FEATURE_DIM, "sequence_length": SEQUENCE_LENGTH,
                    "category": category}, indent=2), encoding="utf-8"
    )
    result = {
        "category": category,
        "samples": int(len(X)),
        "classes": classes,
        "unique_signers": sorted(np.unique(groups).tolist()),
        "train_signers": sorted(np.unique(groups[train_idx]).tolist()),
        "validation_signers": sorted(np.unique(groups[val_idx]).tolist()),
        "test_signers": sorted(np.unique(groups[test_idx]).tolist()),
        "test_accuracy": float(accuracy_score(test_true, predictions)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "epochs_completed": len(history.history.get("loss", [])),
        "model_path": str(model_path),
        "note": "Metrics are from the held-out signer test split. Small datasets may produce unstable estimates.",
    }
    (category_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return (
        f"Training complete for {category}.\n"
        f"Samples: {len(X)} | Classes: {len(classes)} | Signers: {len(np.unique(groups))}\n"
        f"Held-out signer test accuracy: {result['test_accuracy']:.3f}\n"
        f"Model: {model_path}\n"
        f"Evaluation: {category_dir / 'evaluation.json'}"
    )
