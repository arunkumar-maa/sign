"""
Model evaluation utilities.
Computes accuracy, precision, recall, F1, confusion matrix, per-class accuracy.
Saves results to evaluation_report.json.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    output_path: Optional[Path] = None,
) -> dict:
    """
    Evaluate a Keras model on the test set.
    
    Args:
        model: Trained Keras model.
        X_test: Test features.
        y_test: True class indices (integer-encoded).
        labels: List of class names.
        output_path: If provided, save report to this JSON file.
    
    Returns:
        Evaluation report dict.
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score
    )

    # Measure inference latency
    t0 = time.perf_counter()
    probs = model.predict(X_test, verbose=0)
    elapsed = (time.perf_counter() - t0) * 1000
    avg_latency_ms = elapsed / max(len(X_test), 1)

    y_pred = np.argmax(probs, axis=1)
    y_true = y_test.astype(int)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    per_class = {}
    for i, label in enumerate(labels):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[label] = float(accuracy_score(y_true[mask], y_pred[mask]))

    report = {
        "test_accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class,
        "avg_inference_latency_ms": avg_latency_ms,
        "num_test_samples": len(y_test),
        "num_classes": len(labels),
        "labels": labels,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_note": "All metrics are computed on the held-out test set only.",
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report saved to %s", output_path)

    logger.info(
        "Evaluation — accuracy: %.4f, F1: %.4f, latency: %.1fms",
        accuracy, f1, avg_latency_ms,
    )
    return report

