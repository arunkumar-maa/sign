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


WORD_TO_DIGIT = {
    "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
    "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9",
}
VALID_NUMBER_CLASSES = [str(i) for i in range(10)]
VALID_NUMBER_SET = set(VALID_NUMBER_CLASSES)


def validate_label(category: str, label: str) -> tuple[str, str]:
    category = (category or "").strip().lower()
    if category not in CATEGORIES:
        raise ValueError("Choose Alphabet or Number mode.")
    label = clean_token(label, "label")
    if category == "alphabet" and not re.fullmatch(r"[A-Z]", label):
        raise ValueError("Alphabet labels must be a single uppercase letter from A to Z.")
    if category == "number":
        # Normalise source labels such as TWO to 2 during dataset collection/import
        normalized = WORD_TO_DIGIT.get(label, label)
        if normalized not in VALID_NUMBER_SET:
            raise ValueError(f"Number labels must be digits 0–9 (or ZERO–NINE), got: {label}")
        label = normalized
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


def upsert_landmark_record(record: dict[str, Any], path: Path = LANDMARK_MANIFEST) -> bool:
    """
    Insert or update a record in the landmark manifest JSONL by `sample_id`.
    Returns True if a new record was added or existing record was updated, False if identical.
    Guarantees idempotence: running import multiple times never creates duplicates.
    """
    sample_id = record.get("sample_id")
    if not sample_id:
        append_jsonl(path, record)
        return True

    with _LOCK:
        existing_rows = read_jsonl(path)
        updated = False
        found = False
        new_rows = []
        for row in existing_rows:
            if row.get("sample_id") == sample_id:
                found = True
                if row != record:
                    new_rows.append(record)
                    updated = True
                else:
                    new_rows.append(row)
            else:
                new_rows.append(row)

        if not found:
            new_rows.append(record)
            updated = True

        if updated:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for r in new_rows:
                    handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        return updated


def append_landmark_record(record: dict[str, Any]) -> None:
    if record.get("sample_id"):
        upsert_landmark_record(record)
    else:
        append_jsonl(LANDMARK_MANIFEST, record)
