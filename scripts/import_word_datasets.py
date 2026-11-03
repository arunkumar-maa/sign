"""
SignBridge AI — Import Word Recognition Datasets & Train models/isl_word_model.keras.
Integrates:
1. Mendeley Data dtg9s785v8/1 (Amrita SLR Real-Time ISL Dataset) — CC BY 4.0
2. Hugging Face vidit031/isl-isolated-40words — CC BY 4.0 / Open Access

Extracts standardized 282-dim MediaPipe hand landmark sequences (30 frames),
performs signer-independent train/val/test split, trains models/isl_word_model.keras,
and saves confusion matrix, classification report, and label mappings.
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from isl_data_collection.config import (
    DATA_ROOT, FEATURE_DIM, FEATURE_DIR, LANDMARK_MANIFEST,
    MODEL_DIR, SAMPLES_MANIFEST, SEQUENCE_LENGTH,
)
from isl_data_collection.hand_landmarks import HandPoseDetector, result_to_features
from isl_data_collection.storage import append_landmark_record, append_sample, read_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

AMRITA_S3_URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/21487202-e9d2-4f71-940c-f882f1f72695"
AMRITA_ARCHIVE_SIZE = 1465051081

# Standard 40 vocabulary classes
STANDARD_40_WORDS = [
    "brother", "come", "drink", "eat", "father", "food", "friend", "go",
    "goodbye", "he", "hello", "help", "home", "hospital", "market", "me",
    "mother", "no", "okay", "please", "read", "school", "she", "sister",
    "sit", "sorry", "stand", "stop", "student", "tea", "teacher", "thank you",
    "today", "water", "what", "when", "where", "write", "yes", "you"
]

AMRITA_CLASS_MAP = {
    "sign project/20.brother/": "brother",
    "sign project/18.drinking/": "drink",
    "sign project/16.eat/": "eat",
    "sign project/22.father/": "father",  
    "sign project/17.friend/": "friend",
    "sign project/9.hello/": "hello",
    "sign project/24.help/": "help",
    "sign project/25.mother/": "mother",
    "sign project/31.sit/": "sit",
    "sign project/32.sorry/": "sorry",
    "sign project/33.stand/": "stand",
    "sign project/34.stop/": "stop",
    "sign project/35.student/": "student",
    "sign project/37.teacher/": "teacher",
    "sign project/38.thankyou/": "thank you",
    "sign project/39.today/": "today",
    "sign project/46.write/": "write",
}


class RemoteZipFile:
    """HTTP Range-request backed seekable reader for remote ZIP files."""
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


import struct
import zlib

def extract_file_from_remote_zip(url: str, info: zipfile.ZipInfo) -> bytes:
    """Download and decompress a ZIP member directly using a single HTTP Range request."""
    total_needed = 30 + len(info.filename.encode("utf-8")) + 2048 + info.compress_size
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={info.header_offset}-{info.header_offset + total_needed}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=25) as r:
        raw = r.read()
    sig, ver, flag, comp, mtime, mdate, crc, csize, usize, fn_len, extra_len = struct.unpack("<IHHHHHIIIHH", raw[:30])
    data_start = 30 + fn_len + extra_len
    cdata = raw[data_start:data_start + info.compress_size]
    if info.compress_type == 8:
        return zlib.decompress(cdata, -15)
    return cdata


def download_amrita_subset(dest_dir: Path, max_per_class: int = 5) -> List[Dict[str, Any]]:
    """Download matching video clips from Amrita SLR dataset via 1-shot HTTP Range requests."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_videos = list(dest_dir.glob("*/*.mp4")) + list(dest_dir.glob("*/*.mov"))
    if len(existing_videos) >= 50:
        logger.info("Amrita SLR: Found %d existing staged clips locally. Skipping remote zip index.", len(existing_videos))
        records = []
        for vpath in existing_videos:
            target_label = vpath.parent.name
            fname = vpath.name
            signer = "amrita_unknown"
            lower_fname = fname.lower()
            for s in ["krishnaraj", "manideep", "nachiketh", "0403"]:
                if s in lower_fname:
                    signer = f"amrita_{s}"
                    break
            if signer == "amrita_unknown":
                signer = f"amrita_signer_{fname[:6]}"
            records.append({
                "video_path": str(vpath),
                "label": target_label,
                "signer": signer,
                "dataset": "mendeley_amrita_slr",
                "license": "CC BY 4.0",
                "sample_id": hashlib.md5(f"amrita_{target_label}_{fname}".encode()).hexdigest()[:12],
            })
        return records

    logger.info("Connecting to Amrita SLR remote archive (%s)...", AMRITA_S3_URL)

    rz = RemoteZipFile(AMRITA_S3_URL, AMRITA_ARCHIVE_SIZE)
    zf = zipfile.ZipFile(rz)
    namelist = zf.namelist()
    logger.info("Parsed remote ZIP directory: %d total entries.", len(namelist))

    downloaded_records = []
    for prefix, target_label in AMRITA_CLASS_MAP.items():
        matching_files = [
            f for f in namelist
            if f.startswith(prefix) and (f.lower().endswith(".mp4") or f.lower().endswith(".mov"))
        ]
        target_class_dir = dest_dir / target_label
        target_class_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Amrita SLR: Found %d files for '%s'", len(matching_files), target_label)
        # Prioritize classes that have fewer samples in HF
        n_to_download = max_per_class
        if target_label in {"brother", "sorry", "stand", "stop", "write"}:
            n_to_download = min(len(matching_files), 10)

        for rel_file in matching_files[:n_to_download]:
            fname = Path(rel_file).name
            local_path = target_class_dir / fname

            signer = "amrita_unknown"
            lower_fname = fname.lower()
            for s in ["krishnaraj", "manideep", "nachiketh", "0403"]:
                if s in lower_fname:
                    signer = f"amrita_{s}"
                    break
            if signer == "amrita_unknown":
                signer = f"amrita_signer_{fname[:6]}"

            if not local_path.exists() or local_path.stat().st_size < 1000:
                try:
                    info = zf.getinfo(rel_file)
                    data = extract_file_from_remote_zip(AMRITA_S3_URL, info)
                    local_path.write_bytes(data)
                except Exception as e:
                    logger.warning("Failed to download %s: %s", rel_file, e)
                    continue

            downloaded_records.append({
                "video_path": local_path,
                "label": target_label,
                "signer": signer,
                "dataset": "mendeley_amrita_slr",
                "license": "CC BY 4.0",
                "sample_id": hashlib.md5(f"amrita_{target_label}_{fname}".encode()).hexdigest()[:12],
            })

    logger.info("Amrita SLR: Successfully staged %d clips across %d classes.",
                len(downloaded_records), len(AMRITA_CLASS_MAP))
    return downloaded_records


def load_hf_word_dataset() -> List[Dict[str, Any]]:
    """Load accepted clips from data/raw/isl_isolated_40words."""
    meta_path = PROJECT_ROOT / "data" / "raw" / "isl_isolated_40words" / "metadata.csv"
    if not meta_path.exists():
        logger.error("Metadata not found at %s", meta_path)
        return []

    df = pd.read_csv(meta_path)
    # Filter review_status == 'accepted' (Rule 5: Filter 'Needs Manual Review')
    accepted = df[df["review_status"] == "accepted"].copy()
    logger.info("Loaded HF isl-isolated-40words: %d total, %d accepted clips.", len(df), len(accepted))

    records = []
    base_dir = PROJECT_ROOT / "data" / "raw" / "isl_isolated_40words"
    for _, row in accepted.iterrows():
        rel_vpath = row["video_path"]
        vpath = PROJECT_ROOT / rel_vpath if not Path(rel_vpath).is_absolute() else Path(rel_vpath)
        if not vpath.exists():
            # Check alternative relative path
            candidate = base_dir / row["normalized_word"] / Path(rel_vpath).name
            if candidate.exists():
                vpath = candidate
            else:
                continue

        word = str(row["normalized_word"]).strip()
        if word not in STANDARD_40_WORDS:
            continue

        signer = str(row.get("signer", "unknown"))
        records.append({
            "video_path": vpath,
            "label": word,
            "signer": f"hf_{signer}",
            "dataset": "hf_isl_isolated_40words",
            "license": str(row.get("license", "CC BY 4.0")),
            "sample_id": hashlib.md5(f"hf_{word}_{vpath.name}".encode()).hexdigest()[:12],
        })

    logger.info("HF dataset: Validated %d clips across %d classes.",
                len(records), len(set(r["label"] for r in records)))
    return records


def extract_features_from_clip(
    video_path: Path,
    detector: Any,
    stride: int = 2,
) -> Optional[np.ndarray]:
    """Extract 282-dim hand landmarks from video using MediaPipe Tasks HandLandmarker in IMAGE mode."""
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    vectors = []
    frame_idx = 0
    hands_detected = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            if frame_idx % stride == 0:
                # Resize to 320x240 for rapid MediaPipe Tasks inference
                rgb_small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (320, 240))
                rgb_small = np.ascontiguousarray(rgb_small)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
                result = detector.detect(image)
                feat, cnt = result_to_features(result)
                vectors.append(feat)
                if cnt > 0:
                    hands_detected += 1
            frame_idx += 1
    except Exception as e:
        logger.warning("Error processing %s: %s", video_path.name, e)
        return None
    finally:
        cap.release()

    if len(vectors) < 4:
        return None
    # Require at least 20% hand detection rate
    if hands_detected < max(2, int(len(vectors) * 0.20)):
        return None

    return np.stack(vectors).astype(np.float32)


def resample_sequence(sequence: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linear interpolation of sequence to target_length frames."""
    positions = np.linspace(0, len(sequence) - 1, target_length)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    weight = (positions - left)[:, None].astype(np.float32)
    return (sequence[left] * (1.0 - weight) + sequence[right] * weight).astype(np.float32)


import threading

tls = threading.local()

def get_thread_detector() -> Any:
    """Thread-local MediaPipe HandLandmarker in IMAGE mode."""
    import mediapipe as mp
    from isl_data_collection.hand_landmarks import ensure_hand_task_model
    if not hasattr(tls, "detector"):
        model_path = ensure_hand_task_model()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
        )
        tls.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    return tls.detector


def process_single_clip(rec: Dict[str, Any]) -> Optional[Tuple[np.ndarray, str, str, Dict[str, Any]]]:
    """Process a single clip with thread-local detector and file-level caching."""
    sample_id = rec["sample_id"]
    npz_path = FEATURE_DIR / f"word_{sample_id}.npz"

    raw_features = None
    if npz_path.exists():
        try:
            raw_features = np.load(npz_path)["features"]
            if raw_features.ndim != 2 or raw_features.shape[1] != FEATURE_DIM:
                raw_features = None
        except Exception:
            raw_features = None

    if raw_features is None:
        detector = get_thread_detector()
        raw_features = extract_features_from_clip(rec["video_path"], detector, stride=2)
        if raw_features is not None:
            np.savez_compressed(npz_path, features=raw_features)

    if raw_features is not None:
        resampled = resample_sequence(raw_features, SEQUENCE_LENGTH)
        record_meta = dict(rec)
        record_meta["features_path"] = str(npz_path.relative_to(PROJECT_ROOT))
        record_meta["extracted_frames"] = int(raw_features.shape[0])
        record_meta["category"] = "word"
        return resampled, rec["label"], rec["signer"], record_meta
    return None


def process_all_clips(all_records: List[Dict[str, Any]], max_per_class: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Extract features for staged video clips concurrently with per-class balancing."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Balance clips per class so every word has up to max_per_class clips
    from collections import defaultdict
    class_groups = defaultdict(list)
    for r in all_records:
        class_groups[r["label"]].append(r)

    balanced_records = []
    for lbl, recs in class_groups.items():
        balanced_records.extend(recs[:max_per_class])

    logger.info("Extracting features for %d balanced clips across %d classes using 4 parallel threads...",
                len(balanced_records), len(class_groups))
    t0 = time.time()

    X_list, y_list, signer_list, processed_records = [], [], [], []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_clip, rec) for rec in balanced_records]
        for idx, f in enumerate(as_completed(futures)):
            res = f.result()
            if res is not None:
                feat, lbl, signer, meta = res
                X_list.append(feat)
                y_list.append(lbl)
                signer_list.append(signer)
                processed_records.append(meta)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(balanced_records):
                elapsed = time.time() - t0
                logger.info("Progress: %d/%d clips processed (Valid: %d) [%.1fs]",
                            idx + 1, len(balanced_records), len(X_list), elapsed)

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list),
        np.asarray(signer_list),
        processed_records,
    )


def build_and_train_word_model(
    X: np.ndarray,
    y_labels: np.ndarray,
    signers: np.ndarray,
    epochs: int = 35,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Train BiLSTM word model with signer-independent validation & test splits."""
    import tensorflow as tf

    classes = sorted(list(set(y_labels)))
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    index_to_label = {idx: label for idx, label in enumerate(classes)}
    y = np.asarray([label_to_index[lbl] for lbl in y_labels], dtype=np.int64)

    logger.info("Dataset shape: X=%s, y=%s, Classes: %d, Unique Signers: %d",
                X.shape, y.shape, len(classes), len(np.unique(signers)))

    # Signer-independent train/val/test split
    # Step 1: Split into Train+Val vs Test
    gss1 = GroupShuffleSplit(n_splits=20, test_size=0.15, random_state=42)
    split1 = None
    for train_val_idx, test_idx in gss1.split(X, y, signers):
        train_val_classes = set(y[train_val_idx])
        test_classes = set(y[test_idx])
        if len(train_val_classes) == len(classes):
            split1 = (train_val_idx, test_idx)
            break

    if split1 is None:
        logger.warning("Could not achieve 100% class coverage in signer-independent test split; falling back to stratified split.")
        from sklearn.model_selection import train_test_split
        train_val_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.15, random_state=42, stratify=y)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=43, stratify=y[train_val_idx])
    else:
        train_val_idx, test_idx = split1
        gss2 = GroupShuffleSplit(n_splits=20, test_size=0.15, random_state=43)
        split2 = None
        for train_local, val_local in gss2.split(X[train_val_idx], y[train_val_idx], signers[train_val_idx]):
            t_idx = train_val_idx[train_local]
            v_idx = train_val_idx[val_local]
            if len(set(y[t_idx])) == len(classes):
                split2 = (t_idx, v_idx)
                break
        if split2 is None:
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=43, stratify=y[train_val_idx])
        else:
            train_idx, val_idx = split2

    logger.info("Split sizes: Train=%d, Val=%d, Test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Build BiLSTM Model Architecture
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name="input_landmarks"),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(classes), activation="softmax", name="word_probabilities"),
    ], name="ISL_Word_BiLSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    logger.info("Beginning training for %d epochs...", epochs)
    history = model.fit(
        X[train_idx], y[train_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluation on held-out test split
    test_probs = model.predict(X[test_idx], verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_true = y[test_idx]

    acc = float(accuracy_score(test_true, test_preds))
    rep = classification_report(
        test_true, test_preds,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_true, test_preds, labels=list(range(len(classes)))).tolist()

    logger.info("Held-out Test Accuracy: %.2f%%", acc * 100.0)

    # Save models and metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    word_model_path = MODEL_DIR / "isl_word_model.keras"
    word_dir = MODEL_DIR / "word"
    word_dir.mkdir(parents=True, exist_ok=True)
    cat_model_path = word_dir / "hand_sequence_lstm.keras"

    model.save(str(word_model_path))
    model.save(str(cat_model_path))
    logger.info("Saved models to %s and %s", word_model_path, cat_model_path)

    # Save word_labels.json in both standard and category format
    labels_payload = {
        "version": "2.0.0",
        "dataset": "Merged: Amrita SLR (dtg9s785v8/1) + HF vidit031/isl-isolated-40words",
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "num_classes": len(classes),
        "classes": classes,
        "labels": classes,
        "label_to_index": label_to_index,
        "class_to_id": label_to_index,
        "id_to_label": index_to_label,
    }

    with open(MODEL_DIR / "word_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_payload, f, indent=2)
    with open(word_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_payload, f, indent=2)

    eval_result = {
        "category": "word",
        "model_file": "isl_word_model.keras",
        "samples_total": len(X),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
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

    with open(word_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)

    return eval_result


def main():
    logger.info("=== STEP 1: IMPORT WORD RECOGNITION DATASETS ===")
    amrita_dir = PROJECT_ROOT / "data" / "raw" / "amrita_slr"

    # 1. Download & stage Amrita SLR subset
    amrita_records = download_amrita_subset(amrita_dir, max_per_class=8)

    # 2. Load HF word dataset
    hf_records = load_hf_word_dataset()

    all_records = amrita_records + hf_records
    logger.info("Combined dataset records: %d total (%d Amrita + %d HF)",
                len(all_records), len(amrita_records), len(hf_records))

    # 3. Process clips & extract landmarks
    X, y, signers, processed_records = process_all_clips(all_records)

    # Update landmarks manifest
    for rec in processed_records:
        rec_clean = {k: str(v) if isinstance(v, Path) else v for k, v in rec.items()}
        append_landmark_record(rec_clean)

    # 4. Train word model
    eval_result = build_and_train_word_model(X, y, signers, epochs=30, batch_size=16)
    logger.info("Word Model Training Complete! Final Test Accuracy: %.2f%%",
                eval_result["test_accuracy"] * 100.0)


if __name__ == "__main__":
    main()
