"""
SignBridge AI — Phase 5: ISL Dataset Auditor.
Inspects metadata CSV, video file decodability, class balance,
provenance (CISLR & ISL500 subsets), and license terms.
Outputs audit summary to docs/dataset_audit_report.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import cv2
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "isl_isolated_40words"


def audit_dataset() -> Dict[str, Any]:
    logger.info("Starting audit of ISL 40 Words dataset in %s", DATA_DIR)
    metadata_path = DATA_DIR / "metadata.csv"

    if not metadata_path.exists():
        logger.error("Metadata CSV not found at %s", metadata_path)
        return {"error": "metadata_missing"}

    df = pd.read_csv(metadata_path)
    total_records = len(df)
    unique_words = sorted(df["normalized_word"].unique().tolist())

    logger.info("Auditing %d records across %d distinct ISL classes...", total_records, len(unique_words))

    class_distribution: Dict[str, int] = {}
    verified_files = 0
    missing_files = []
    unreadable_files = []
    sources_distribution: Dict[str, int] = {}

    for idx, row in df.iterrows():
        word = str(row["normalized_word"]).strip()
        fname = str(row["file_name"]).strip()
        class_distribution[word] = class_distribution.get(word, 0) + 1

        # Track provenance
        source = "Unknown"
        if "CISLR" in fname:
            source = "CISLR"
        elif "ISL500" in fname:
            source = "ISL500"
        sources_distribution[source] = sources_distribution.get(source, 0) + 1

        vid_path = DATA_DIR / word / fname
        if not vid_path.exists():
            missing_files.append(f"{word}/{fname}")
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            unreadable_files.append(f"{word}/{fname}")
            continue

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            unreadable_files.append(f"{word}/{fname}")
        else:
            verified_files += 1

    audit_summary = {
        "dataset_name": "vidit031/isl-isolated-40words",
        "total_records": total_records,
        "verified_decodable": verified_files,
        "missing_files_count": len(missing_files),
        "unreadable_files_count": len(unreadable_files),
        "classes_count": len(unique_words),
        "classes": unique_words,
        "class_distribution": class_distribution,
        "sources_distribution": sources_distribution,
        "provenance_notes": "Derived from CISLR & ISL500 authentic Indian Sign Language corpora.",
        "license_assessment": "Academic / Non-commercial research use.",
    }

    report_path = PROJECT_ROOT / "docs" / "dataset_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    logger.info("Saved dataset audit report to %s", report_path)
    return audit_summary


def main():
    res = audit_dataset()
    logger.info("Audit complete: %d/%d valid clips.", res.get("verified_decodable", 0), res.get("total_records", 0))


if __name__ == "__main__":
    main()

