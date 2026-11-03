"""
SignBridge AI — Hand Pose Dataset Manager for ISL 40 Words.
Manages discovery, label mapping, and train/val/test splits for the 40-word ISL dataset.
Maintains original ISL vocabulary labels without using ASL or synthetic mappings.
Gracefully handles imbalanced classes with few samples.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

RAW_DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "isl_isolated_40words"
PROCESSED_DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "hand_pose"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hand_pose"


class HandPoseDataset:
    """Manages the ISL 40-word isolated video dataset for hand pose recognition."""

    def __init__(self, raw_dir: Path = RAW_DATASET_DIR):
        self.raw_dir = Path(raw_dir)
        self.classes: List[str] = []
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.samples: List[Tuple[Path, str, int]] = []
        self._scan_dataset()

    def _scan_dataset(self):
        """Scan directory and index all MP4 video clips with their ISL classes."""
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw ISL dataset directory not found at: {self.raw_dir}")

        class_dirs = sorted([d for d in self.raw_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        self.label_to_id = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.id_to_label = {idx: cls_name for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for d in class_dirs:
            label = d.name
            label_id = self.label_to_id[label]
            mp4_files = sorted(list(d.glob("*.mp4")))
            for f in mp4_files:
                self.samples.append((f, label, label_id))

        logger.info(
            "HandPoseDataset: Found %d classes and %d total video clips in %s",
            len(self.classes),
            len(self.samples),
            self.raw_dir,
        )

    def save_labels(self, output_path: Path):
        """Save label dictionary to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_classes": len(self.classes),
            "classes": self.classes,
            "label_to_id": self.label_to_id,
            "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved label mapping to: %s", output_path)

    def get_stratified_splits(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[List[Tuple[Path, str, int]], List[Tuple[Path, str, int]], List[Tuple[Path, str, int]]]:
        """
        Partition samples per class into Train (approx 70%), Val (15%), Test (15%).
        For classes with only 1 sample, include it in Train (and duplicate in Val/Test so all splits have coverage).
        For classes with 2 samples, put 1 in Train and 1 in Test (and duplicate to Val).
        """
        rng = random.Random(random_state)
        by_class: Dict[int, List[Tuple[Path, str, int]]] = defaultdict(list)
        for s in self.samples:
            by_class[s[2]].append(s)

        train: List[Tuple[Path, str, int]] = []
        val: List[Tuple[Path, str, int]] = []
        test: List[Tuple[Path, str, int]] = []

        for lid, items in by_class.items():
            shuffled = list(items)
            rng.shuffle(shuffled)
            n = len(shuffled)

            if n == 1:
                train.append(shuffled[0])
                val.append(shuffled[0])
                test.append(shuffled[0])
            elif n == 2:
                train.append(shuffled[0])
                val.append(shuffled[1])
                test.append(shuffled[1])
            elif n == 3:
                train.append(shuffled[0])
                val.append(shuffled[1])
                test.append(shuffled[2])
            else:
                n_test = max(1, int(round(n * test_size)))
                n_val = max(1, int(round(n * val_size)))
                n_train = n - n_test - n_val
                if n_train <= 0:
                    n_train = 1
                    n_val = max(1, (n - 1) // 2)
                    n_test = n - n_train - n_val

                train.extend(shuffled[:n_train])
                val.extend(shuffled[n_train:n_train + n_val])
                test.extend(shuffled[n_train + n_val:])

        logger.info(
            "Splits created: Train=%d, Val=%d, Test=%d (Total unique samples=%d)",
            len(train),
            len(val),
            len(test),
            len(self.samples),
        )
        return train, val, test
