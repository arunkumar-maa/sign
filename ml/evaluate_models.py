"""
SignBridge AI — Evaluate Hand Pose Recognition Models.
Evaluates both static and dynamic hand pose models on the held-out test split.
Generates metrics:
- Overall Accuracy
- Precision, Recall, F1-Score per class
- Confusion matrix
- Model inference latency
Saves report to: models/hand_pose/evaluation_report.json
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "hand_pose"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hand_pose"


def evaluate_hand_pose_models():
    """Run full evaluation suite on static and dynamic hand pose models."""
    labels_file = MODEL_DIR / "labels.json"
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found at: {labels_file}")

    with open(labels_file, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)

    classes = labels_meta["classes"]
    id_to_label = {int(k): v for k, v in labels_meta["id_to_label"].items()}

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "num_classes": len(classes),
    }

    # 1. Evaluate Dynamic Model
    dynamic_model_path = MODEL_DIR / "isl_dynamic_handpose.keras"
    if dynamic_model_path.exists():
        logger.info("Evaluating Dynamic Model from: %s", dynamic_model_path)
        dynamic_model = tf.keras.models.load_model(str(dynamic_model_path))

        X_test_dyn = np.load(PROCESSED_DATASET_DIR / "X_test.npy")
        y_test_dyn = np.load(PROCESSED_DATASET_DIR / "y_test.npy")

        # Measure latency
        t0 = time.perf_counter()
        probs_dyn = dynamic_model.predict(X_test_dyn, verbose=0)
        avg_dyn_latency_ms = ((time.perf_counter() - t0) / len(X_test_dyn)) * 1000.0

        preds_dyn = np.argmax(probs_dyn, axis=-1)
        acc_dyn = float(accuracy_score(y_test_dyn, preds_dyn))

        unique_labels = sorted(list(set(y_test_dyn) | set(preds_dyn)))
        target_names = [id_to_label[i] for i in unique_labels]

        clf_rep_dyn = classification_report(
            y_test_dyn,
            preds_dyn,
            labels=unique_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        cm_dyn = confusion_matrix(y_test_dyn, preds_dyn, labels=unique_labels).tolist()

        report_data["dynamic_model"] = {
            "model_path": str(dynamic_model_path),
            "test_samples": int(len(X_test_dyn)),
            "accuracy": acc_dyn,
            "average_latency_ms": avg_dyn_latency_ms,
            "classification_report": clf_rep_dyn,
            "confusion_matrix": cm_dyn,
        }
        logger.info("Dynamic Model Test Accuracy: %.2f%% (Latency: %.2f ms/seq)", acc_dyn * 100.0, avg_dyn_latency_ms)

    # 2. Evaluate Static Model
    static_model_path = MODEL_DIR / "isl_static_handpose.keras"
    if static_model_path.exists():
        logger.info("Evaluating Static Model from: %s", static_model_path)
        static_model = tf.keras.models.load_model(str(static_model_path))

        X_test_stat = np.load(PROCESSED_DATASET_DIR / "static_X_test.npy")
        y_test_stat = np.load(PROCESSED_DATASET_DIR / "static_y_test.npy")

        t0 = time.perf_counter()
        probs_stat = static_model.predict(X_test_stat, verbose=0)
        avg_stat_latency_ms = ((time.perf_counter() - t0) / len(X_test_stat)) * 1000.0

        preds_stat = np.argmax(probs_stat, axis=-1)
        acc_stat = float(accuracy_score(y_test_stat, preds_stat))

        unique_labels_stat = sorted(list(set(y_test_stat) | set(preds_stat)))
        target_names_stat = [id_to_label[i] for i in unique_labels_stat]

        clf_rep_stat = classification_report(
            y_test_stat,
            preds_stat,
            labels=unique_labels_stat,
            target_names=target_names_stat,
            output_dict=True,
            zero_division=0,
        )
        cm_stat = confusion_matrix(y_test_stat, preds_stat, labels=unique_labels_stat).tolist()

        report_data["static_model"] = {
            "model_path": str(static_model_path),
            "test_samples": int(len(X_test_stat)),
            "accuracy": acc_stat,
            "average_latency_ms": avg_stat_latency_ms,
            "classification_report": clf_rep_stat,
            "confusion_matrix": cm_stat,
        }
        logger.info("Static Model Test Accuracy: %.2f%% (Latency: %.2f ms/sample)", acc_stat * 100.0, avg_stat_latency_ms)

    out_report_file = MODEL_DIR / "evaluation_report.json"
    with open(out_report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info("✅ Saved full evaluation report to: %s", out_report_file)
    return report_data


if __name__ == "__main__":
    evaluate_hand_pose_models()

