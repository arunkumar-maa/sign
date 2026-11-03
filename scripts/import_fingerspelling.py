"""
SignBridge AI — Import Fingerspelling Dataset & Train models/isl_fingerspelling_model.keras.
Integrates:
Hugging Face kirandevraj/ISL-Fingerspelling — CC BY-NC 4.0
- 1,308 continuous video segments, 14,685 character annotations, 3 signers
- Signer-independent split: signer_00, signer_01 (Train) vs signer_02 (Test)
- Extracts 282-dim MediaPipe hand landmark sequences (30 frames) for letters A through Z (26 classes)
- Trains models/isl_fingerspelling_model.keras
- Saves models/fingerspelling_labels.json and evaluation report with confusion matrix
"""

import hashlib
import json
import logging
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from isl_data_collection.config import (
    DATA_ROOT, FEATURE_DIM, FEATURE_DIR, LANDMARK_MANIFEST,
    MODEL_DIR, SAMPLES_MANIFEST, SEQUENCE_LENGTH,
)
from isl_data_collection.hand_landmarks import HandPoseDetector, ensure_hand_task_model, result_to_features
from isl_data_collection.storage import append_landmark_record, append_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HF_FS_BASE = "https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling"
ANN_URL = f"{HF_FS_BASE}/raw/main/letter_annotations.csv"
SPLIT_URL = f"{HF_FS_BASE}/raw/main/signer_split.csv"
VIDEO_BASE_URL = f"{HF_FS_BASE}/resolve/main/videos"

RAW_FS_DIR = PROJECT_ROOT / "data" / "raw" / "isl_fingerspelling"
VIDEOS_DIR = RAW_FS_DIR / "videos"
ALPHABET_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def download_file(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """Download a file with user-agent header if not already present."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        return False


def stage_annotations_and_select_videos() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Download annotations and select curated video UIDs covering all letters A-Z."""
    RAW_FS_DIR.mkdir(parents=True, exist_ok=True)
    ann_path = RAW_FS_DIR / "letter_annotations.csv"
    split_path = RAW_FS_DIR / "signer_split.csv"

    if not ann_path.exists():
        logger.info("Downloading letter_annotations.csv...")
        download_file(ANN_URL, ann_path)
    if not split_path.exists():
        logger.info("Downloading signer_split.csv...")
        download_file(SPLIT_URL, split_path)

    ann_df = pd.read_csv(ann_path)
    split_df = pd.read_csv(split_path)

    merged = pd.merge(ann_df, split_df, on="uid")
    merged["char"] = merged["letter"].astype(str).str.upper().str.strip()
    merged = merged[merged["char"].isin(ALPHABET_CLASSES)].copy()

    # Select top 3 videos per letter for each split (train and test) to achieve robust class coverage
    uids_needed = set()
    for split_name in ["train", "test"]:
        sub = merged[merged["split"] == split_name]
        for ch in ALPHABET_CLASSES:
            uids_with_ch = sub[sub["char"] == ch]["uid"].unique()
            uids_needed.update(uids_with_ch[:3])

    selected_uids = sorted(list(uids_needed))
    logger.info("Selected %d curated video segments covering all 26 letters across Train & Test splits.",
                len(selected_uids))
    return ann_df, split_df, selected_uids


def download_curated_videos(selected_uids: List[str], max_workers: int = 4) -> Dict[str, Path]:
    """Download the selected video clips concurrently."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_map: Dict[str, Path] = {}

    def fetch_uid(uid: str) -> Tuple[str, Optional[Path]]:
        dest = VIDEOS_DIR / f"{uid}.mp4"
        url = f"{VIDEO_BASE_URL}/{uid}.mp4"
        ok = download_file(url, dest, timeout=40)
        return uid, dest if ok and dest.exists() and dest.stat().st_size > 1000 else None

    logger.info("Downloading %d segment videos from Hugging Face...", len(selected_uids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_uid, uid): uid for uid in selected_uids}
        for future in as_completed(futures):
            uid, path = future.result()
            if path:
                downloaded_map[uid] = path

    logger.info("Successfully staged %d/%d segment videos locally.",
                len(downloaded_map), len(selected_uids))
    return downloaded_map


def extract_character_features(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    detector: Any,
) -> Optional[np.ndarray]:
    """Extract frames for [start_frame, end_frame] and run MediaPipe Tasks HandLandmarker."""
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    vectors = []
    current_frame = 0
    hands_detected = 0

    try:
        # Seek or fast-forward
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame

        while current_frame <= end_frame:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            rgb_small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (320, 240))
            rgb_small = np.ascontiguousarray(rgb_small)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            result = detector.detect(image)
            feat, cnt = result_to_features(result)
            vectors.append(feat)
            if cnt > 0:
                hands_detected += 1
            current_frame += 1
    except Exception as exc:
        logger.warning("Error reading frame interval [%d, %d]: %s", start_frame, end_frame, exc)
        return None
    finally:
        cap.release()

    if len(vectors) < 2 or hands_detected == 0:
        return None

    # Resample to exactly SEQUENCE_LENGTH (30 frames)
    arr = np.stack(vectors).astype(np.float32)
    positions = np.linspace(0, len(arr) - 1, SEQUENCE_LENGTH)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    weight = (positions - left)[:, None].astype(np.float32)
    resampled = (arr[left] * (1.0 - weight) + arr[right] * weight).astype(np.float32)
    return resampled


def build_fingerspelling_dataset(
    ann_df: pd.DataFrame,
    split_df: pd.DataFrame,
    downloaded_videos: Dict[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Extract 30-frame landmark sequences for all character instances."""
    import mediapipe as mp
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ensure_hand_task_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    merged = pd.merge(ann_df, split_df, on="uid")
    merged["char"] = merged["letter"].astype(str).str.upper().str.strip()
    merged = merged[merged["char"].isin(ALPHABET_CLASSES)].copy()
    merged = merged[merged["uid"].isin(downloaded_videos.keys())].copy()

    X_list, y_list, signer_list, split_list = [], [], [], []
    records = []

    logger.info("Extracting character instances from %d annotations...", len(merged))
    for idx, row in merged.iterrows():
        uid = str(row["uid"])
        ch = str(row["char"])
        signer = str(row["signer"])
        split = str(row["split"])
        start_f = int(row["start_frame"])
        end_f = int(row["end_frame"])
        vpath = downloaded_videos.get(uid)
        if not vpath or not vpath.exists():
            continue

        sample_id = hashlib.md5(f"fs_{uid}_{start_f}_{end_f}_{ch}".encode()).hexdigest()[:12]
        npz_path = FEATURE_DIR / f"alphabet_{sample_id}.npz"

        seq = None
        if npz_path.exists():
            try:
                seq = np.load(npz_path)["features"]
                if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                    seq = None
            except Exception:
                seq = None

        if seq is None:
            seq = extract_character_features(vpath, start_f, end_f, detector)
            if seq is not None:
                np.savez_compressed(npz_path, features=seq)

        if seq is not None:
            X_list.append(seq)
            y_list.append(ch)
            signer_list.append(signer)
            split_list.append(split)

            rec = {
                "sample_id": sample_id,
                "category": "alphabet",
                "label": ch,
                "signer_id": signer,
                "split": split,
                "video_path": str(vpath.relative_to(PROJECT_ROOT)),
                "features_path": str(npz_path.relative_to(PROJECT_ROOT)),
                "source": "kirandevraj/ISL-Fingerspelling",
                "license": "CC BY-NC 4.0",
                "start_frame": start_f,
                "end_frame": end_f,
            }
            records.append(rec)
            append_landmark_record(rec)

    detector.close()
    logger.info("Successfully extracted %d character instances across all letters.", len(X_list))
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list),
        np.asarray(signer_list),
        np.asarray(split_list),
        records,
    )


def train_fingerspelling_model(
    X: np.ndarray,
    y_labels: np.ndarray,
    signers: np.ndarray,
    splits: np.ndarray,
    epochs: int = 35,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Train fingerspelling model using signer-independent split (signer_02 held out for test)."""
    import tensorflow as tf

    classes = ALPHABET_CLASSES
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    index_to_label = {idx: label for idx, label in enumerate(classes)}
    y = np.asarray([label_to_index[lbl] for lbl in y_labels], dtype=np.int64)

    # Use dataset's official signer-independent split
    test_mask = (splits == "test") | (signers == "signer_02")
    train_val_mask = ~test_mask

    train_val_indices = np.where(train_val_mask)[0]
    test_indices = np.where(test_mask)[0]

    # Split train_val into 80% train, 20% val
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.20, random_state=42, stratify=y[train_val_indices]
    )

    logger.info("Signer-independent Split: Train=%d (Signers: %s), Val=%d, Test=%d (Signer: signer_02)",
                len(train_indices), sorted(list(set(signers[train_indices]))),
                len(val_indices), len(test_indices))

    # Architecture
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name="input_landmarks"),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.LSTM(48),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(len(classes), activation="softmax", name="letter_probabilities"),
    ], name="ISL_Fingerspelling_BiLSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        X[train_indices], y[train_indices],
        validation_data=(X[val_indices], y[val_indices]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on held-out signer_02 test split
    test_probs = model.predict(X[test_indices], verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_true = y[test_indices]

    acc = float(accuracy_score(test_true, test_preds))
    rep = classification_report(
        test_true, test_preds,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_true, test_preds, labels=list(range(len(classes)))).tolist()

    logger.info("Held-out Signer Test Accuracy (Alphabet A-Z): %.2f%%", acc * 100.0)

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "isl_fingerspelling_model.keras"
    alphabet_dir = MODEL_DIR / "alphabet"
    alphabet_dir.mkdir(parents=True, exist_ok=True)
    cat_model_path = alphabet_dir / "hand_sequence_lstm.keras"

    model.save(str(model_path))
    model.save(str(cat_model_path))
    logger.info("Saved models to %s and %s", model_path, cat_model_path)

    # Save label mappings
    labels_payload = {
        "version": "1.0.0",
        "dataset": "kirandevraj/ISL-Fingerspelling",
        "license": "CC BY-NC 4.0",
        "category": "alphabet",
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "num_classes": len(classes),
        "classes": classes,
        "labels": classes,
        "label_to_index": label_to_index,
        "class_to_id": label_to_index,
        "id_to_label": index_to_label,
    }

    with open(MODEL_DIR / "fingerspelling_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_payload, f, indent=2)
    with open(alphabet_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_payload, f, indent=2)

    eval_result = {
        "category": "alphabet",
        "model_file": "isl_fingerspelling_model.keras",
        "samples_total": len(X),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "classes_count": len(classes),
        "classes": classes,
        "test_accuracy": acc,
        "classification_report": rep,
        "confusion_matrix": cm,
        "history": {
            "loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            "accuracy": [float(v) for v in history.history.get("accuracy", [])],
            "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        },
        "epochs_completed": len(history.history.get("loss", [])),
    }

    with open(alphabet_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)

    return eval_result


def main():
    logger.info("=== STEP 2: IMPORT ISL FINGERSPELLING DATASET ===")
    ann_df, split_df, selected_uids = stage_annotations_and_select_videos()
    downloaded_videos = download_curated_videos(selected_uids, max_workers=4)
    X, y, signers, splits, records = build_fingerspelling_dataset(ann_df, split_df, downloaded_videos)

    if len(X) < 20:
        logger.error("Not enough extracted fingerspelling samples (%d).", len(X))
        return

    eval_result = train_fingerspelling_model(X, y, signers, splits, epochs=35, batch_size=16)
    logger.info("ISL Fingerspelling Model Training Complete! Final Test Accuracy: %.2f%%",
                eval_result["test_accuracy"] * 100.0)


if __name__ == "__main__":
    main()

