"""
SignBridge AI — Phase 5: Model Diagnostics and Verification Engine.
Audits trained model files, input/output tensor dimensions, network architectures,
label mappings, and verifies consistency between training and inference preprocessing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def diagnose_model(
    model_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Forensic inspection of a trained ISL recognition model and its label mapping.
    
    Returns:
        Structured dictionary with diagnostic findings.
    """
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "isl_word_model.keras"
    else:
        model_path = Path(model_path)

    if labels_path is None:
        labels_path = PROJECT_ROOT / "models" / "word_labels.json"
    else:
        labels_path = Path(labels_path)

    report: Dict[str, Any] = {
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "model_size_bytes": model_path.stat().st_size if model_path.exists() else 0,
        "labels_path": str(labels_path),
        "labels_exist": labels_path.exists(),
        "input_shape": None,
        "output_shape": None,
        "num_classes": 0,
        "architecture": [],
        "labels": [],
        "is_valid": False,
        "errors": [],
    }

    # 1. Check file existence
    if not model_path.exists():
        report["errors"].append(f"Model file not found at: {model_path}")
        return report

    if not labels_path.exists():
        report["errors"].append(f"Labels file not found at: {labels_path}")
        return report

    # 2. Inspect labels
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            labels = data.get("labels", [])
        elif isinstance(data, list):
            labels = data
        else:
            labels = []
        report["labels"] = labels
        report["num_classes"] = len(labels)
    except Exception as exc:
        report["errors"].append(f"Error reading labels file: {exc}")
        return report

    # 3. Load model and inspect shapes and layers
    try:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        model = tf.keras.models.load_model(str(model_path))
        report["input_shape"] = list(model.input_shape) if model.input_shape else None
        report["output_shape"] = list(model.output_shape) if model.output_shape else None

        for layer in model.layers:
            report["architecture"].append({
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": getattr(layer, "output_shape", None),
            })

        # 4. Consistency checks
        expected_classes = report["num_classes"]
        model_classes = report["output_shape"][-1] if report["output_shape"] else 0

        if expected_classes != model_classes:
            report["errors"].append(
                f"Mismatch between labels count ({expected_classes}) and model output units ({model_classes})"
            )
        else:
            report["is_valid"] = True

    except Exception as exc:
        report["errors"].append(f"Failed to load or inspect Keras model: {exc}")
        report["is_valid"] = False

    return report


if __name__ == "__main__":
    import pprint
    rep = diagnose_model()
    pprint.pprint(rep)

