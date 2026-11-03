import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CATEGORIES, SAMPLES_MANIFEST, LANDMARK_MANIFEST

_LOCK = threading.RLock()


def clean_token(value: str, field_name: str, max_length: int = 48) -> str:
    value = (value or "").strip().upper()
    value = re.sub(r"\s+", "_", value)
    if not value or len(value) > max_length:
        raise ValueError(f"Enter a valid {field_name} (1–{max_length} characters).")
    if not re.fullmatch(r"[A-Z0-9_-]+", value):
        raise ValueError(f"{field_name} may contain only letters, numbers, underscores and hyphens.")
    return value


def validate_label(category: str, label: str) -> tuple[str, str]:
    category = (category or "").strip().lower()
    if category not in CATEGORIES:
        raise ValueError("Choose Word, Alphabet or Number mode.")
    label = clean_token(label, "label")
    if category == "alphabet" and not re.fullmatch(r"[A-Z]", label):
        raise ValueError("Alphabet labels must be one letter from A to Z.")
    if category == "number" and label not in {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"
    }:
        raise ValueError("Number labels must be 0–10 or ZERO–TEN. Confirm that the clip shows the ISL number sign, not just a finger count.")
    return category, label


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with _LOCK, path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}, line {line_number}: {exc}") from exc
    return rows


def sample_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {category: {} for category in CATEGORIES}
    for row in read_jsonl(SAMPLES_MANIFEST):
        category = row.get("category")
        label = row.get("label")
        if category in counts and label:
            counts[category][label] = counts[category].get(label, 0) + 1
    return counts


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_sample(record: dict[str, Any]) -> None:
    append_jsonl(SAMPLES_MANIFEST, record)


def append_landmark_record(record: dict[str, Any]) -> None:
    append_jsonl(LANDMARK_MANIFEST, record)
