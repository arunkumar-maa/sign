"""
SQLite database management for SignBridge AI conversation history.
"""

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_history (
    id TEXT PRIMARY KEY,
    gloss_sequence TEXT NOT NULL,
    english_text TEXT NOT NULL,
    translated_text TEXT,
    language_code TEXT,
    language_name TEXT,
    timestamp REAL NOT NULL,
    confirmed INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
)
"""


class Database:
    """Manages the SQLite database for conversation history."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            logger.info("Database ready at %s", db_path)
        except Exception as exc:
            logger.warning("Could not initialize database at %s (%s). Falling back to /tmp/signbridge.db or :memory:", db_path, exc)
            fallback = Path("/tmp/signbridge.db")
            try:
                fallback.parent.mkdir(parents=True, exist_ok=True)
                self._db_path = fallback
                self._init_db()
                logger.info("Fallback database ready at %s", fallback)
            except Exception as exc2:
                logger.warning("Could not initialize fallback database (%s). Falling back to in-memory SQLite.", exc2)
                self._db_path = Path(":memory:")
                self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute(CREATE_HISTORY_TABLE)
            conn.commit()

    def save_conversation(
        self,
        gloss_sequence: list[str],
        english_text: str,
        translated_text: str = "",
        language_code: str = "en",
        language_name: str = "English",
        confirmed: bool = True,
    ) -> str:
        """Save a conversation record. Returns the new record ID."""
        record_id = str(uuid.uuid4())
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO conversation_history
                   (id, gloss_sequence, english_text, translated_text,
                    language_code, language_name, timestamp, confirmed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id,
                    json.dumps(gloss_sequence),
                    english_text,
                    translated_text,
                    language_code,
                    language_name,
                    time.time(),
                    1 if confirmed else 0,
                ),
            )
            conn.commit()
        logger.debug("Saved conversation: %s", record_id)
        return record_id

    def load_history(self, limit: int = 100) -> list[dict]:
        """Return recent conversation records."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete_record(self, record_id: str) -> bool:
        """Delete a single record by ID. Returns True if deleted."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_history WHERE id = ?", (record_id,)
            )
            conn.commit()
        return cursor.rowcount > 0

    def delete_all(self) -> int:
        """Delete all records. Returns count deleted."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM conversation_history")
            conn.commit()
        return cursor.rowcount

    def export_transcript(self) -> str:
        """Export all records as a text transcript."""
        records = self.load_history(limit=10000)
        lines = ["SignBridge AI — Conversation Transcript", "=" * 50]
        for r in records:
            import datetime
            dt = datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"\n[{dt}]")
            lines.append(f"  ISL Glosses : {' '.join(r['gloss_sequence'])}")
            lines.append(f"  English     : {r['english_text']}")
            if r.get("translated_text"):
                lines.append(f"  {r.get('language_name','Translated')}: {r['translated_text']}")
        return "\n".join(lines)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["gloss_sequence"] = json.loads(d["gloss_sequence"])
        except Exception:
            d["gloss_sequence"] = []
        return d
