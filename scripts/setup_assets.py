"""
SignBridge AI — Setup Required Model & MediaPipe Assets.
Ensures that all runtime model assets and MediaPipe task binaries are downloaded,
verified, and compatible before running the live recognition application.

Usage:
    python scripts/setup_assets.py
    python scripts/setup_assets.py --train  # Train missing models if datasets exist
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision.feature_schema import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    verify_model_compatibility,
    IncompatibleModelError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("setup_assets")

MEDIAPIPE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
TASK_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

ALPHABET_MODEL_PATH = PROJECT_ROOT / "models" / "isl_fingerspelling_model.keras"
ALPHABET_LABELS_PATH = PROJECT_ROOT / "models" / "fingerspelling_labels.json"

NUMBER_MODEL_PATH = PROJECT_ROOT / "models" / "isl_number_model.keras"
NUMBER_LABELS_PATH = PROJECT_ROOT / "models" / "number_labels.json"


def download_mediapipe_task() -> bool:
    """Download official MediaPipe Tasks hand_landmarker.task if missing."""
    TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TASK_PATH.exists() and TASK_PATH.stat().st_size > 1000000:
        logger.info("MediaPipe task binary already present: %s (%d bytes)", TASK_PATH, TASK_PATH.stat().st_size)
        return True

    logger.info("Downloading official MediaPipe HandLandmarker task asset from %s...", MEDIAPIPE_TASK_URL)
    try:
        req = urllib.request.Request(MEDIAPIPE_TASK_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(TASK_PATH, "wb") as f:
            data = resp.read()
            f.write(data)
        logger.info("Successfully downloaded hand_landmarker.task (%d bytes)", len(data))
        return True
    except Exception as exc:
        logger.error("Failed to download hand_landmarker.task: %s", exc)
        return False


def verify_models():
    """Verify presence and schema compatibility of trained models."""
    import tensorflow as tf

    status = {}

    # Alphabet
    if ALPHABET_MODEL_PATH.exists():
        try:
            m = tf.keras.models.load_model(str(ALPHABET_MODEL_PATH), compile=False)
            verify_model_compatibility(m, expected_dim=FEATURE_DIM, expected_seq_len=SEQUENCE_LENGTH)
            status["alphabet"] = {
                "available": True,
                "path": str(ALPHABET_MODEL_PATH),
                "input_shape": list(m.input_shape),
                "output_shape": list(m.output_shape),
            }
            logger.info("Alphabet Model: AVAILABLE & VERIFIED (Input: %s, Output: %s)", m.input_shape, m.output_shape)
        except IncompatibleModelError as ime:
            status["alphabet"] = {"available": False, "error": str(ime)}
            logger.error("Alphabet Model: INCOMPATIBLE SCHEMA: %s", ime)
        except Exception as e:
            status["alphabet"] = {"available": False, "error": str(e)}
            logger.error("Alphabet Model: FAILED LOADING: %s", e)
    else:
        status["alphabet"] = {
            "available": False,
            "error": "File not found (weights are gitignored by policy)",
            "train_cmd": "python scripts/train_alphabet.py",
        }
        logger.warning("Alphabet Model: NOT FOUND at %s", ALPHABET_MODEL_PATH)
        logger.info("-> To train: python scripts/train_alphabet.py")

    # Numbers
    if NUMBER_MODEL_PATH.exists():
        try:
            m = tf.keras.models.load_model(str(NUMBER_MODEL_PATH), compile=False)
            verify_model_compatibility(m, expected_dim=FEATURE_DIM, expected_seq_len=SEQUENCE_LENGTH)
            status["numbers"] = {
                "available": True,
                "path": str(NUMBER_MODEL_PATH),
                "input_shape": list(m.input_shape),
                "output_shape": list(m.output_shape),
            }
            logger.info("Number Model: AVAILABLE & VERIFIED (Input: %s, Output: %s)", m.input_shape, m.output_shape)
        except IncompatibleModelError as ime:
            status["numbers"] = {"available": False, "error": str(ime)}
            logger.error("Number Model: INCOMPATIBLE SCHEMA: %s", ime)
        except Exception as e:
            status["numbers"] = {"available": False, "error": str(e)}
            logger.error("Number Model: FAILED LOADING: %s", e)
    else:
        status["numbers"] = {
            "available": False,
            "error": "File not found (weights are gitignored by policy)",
            "train_cmd": "python scripts/train_numbers.py",
        }
        logger.warning("Number Model: NOT FOUND at %s", NUMBER_MODEL_PATH)
        logger.info("-> To train: python scripts/train_numbers.py")

    return status


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Asset Setup")
    parser.add_argument("--train", action="store_true", help="Automatically train missing models")
    args = parser.parse_args()

    logger.info("Step 1: Checking MediaPipe Tasks asset...")
    download_mediapipe_task()

    logger.info("Step 2: Checking trained Keras models...")
    status = verify_models()

    if args.train:
        if not status["alphabet"]["available"]:
            logger.info("Training missing alphabet model...")
            from scripts.train_alphabet import main as train_alph
            train_alph()
        if not status["numbers"]["available"]:
            logger.info("Training missing numbers model...")
            from scripts.train_numbers import main as train_num
            train_num()

    logger.info("Setup and verification complete.")


if __name__ == "__main__":
    main()

