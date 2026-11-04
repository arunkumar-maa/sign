from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "isl_signs"
VIDEO_DIR = DATA_ROOT / "videos"
FEATURE_DIR = DATA_ROOT / "features"
SAMPLES_MANIFEST = DATA_ROOT / "samples.jsonl"
LANDMARK_MANIFEST = DATA_ROOT / "landmarks.jsonl"
MODEL_DIR = PROJECT_ROOT / "models"
HAND_TASK_PATH = MODEL_DIR / "hand_landmarker.task"

HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

CATEGORIES = ("alphabet", "number")
MAX_HANDS = 2
FEATURE_DIM = 282  # two hands * (126 coordinates + 10 angles + 4 distances) + two presence flags
SEQUENCE_LENGTH = 30
RECORDING_FPS = 10.0
MIN_RECORDING_FRAMES = 10
MAX_RECORDING_SECONDS = 8.0

for directory in (VIDEO_DIR, FEATURE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
