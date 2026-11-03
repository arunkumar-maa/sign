"""
Avatar animation mapping.
Loads the verified animation manifest and looks up clips by gesture.
Only clips with status='verified' are served.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_UNAVAILABLE_MSG = "Avatar animation for this sign is not yet supported."


class AnimationMapper:
    """
    Manages the animation manifest and resolves gesture names to clip paths.
    
    Only entries with status='verified' can be played.
    All other entries return the unavailable message.
    """

    def __init__(self, manifest_path: Path):
        self._manifest_path = manifest_path
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if not self._manifest_path.exists():
            logger.warning("Animation manifest not found at %s", self._manifest_path)
            return
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("animations", []):
                gesture = entry.get("gesture", "").upper()
                if gesture:
                    self._entries[gesture] = entry
            logger.info("Animation manifest loaded — %d entries", len(self._entries))
        except Exception as exc:
            logger.error("Failed to load animation manifest: %s", exc)

    def get_clip(self, gesture: str) -> Dict[str, Any]:
        """
        Look up a gesture's animation clip.
        
        Returns:
            dict with 'available' bool, optional 'path', 'source', 'license',
            and 'message' (human-readable status).
        """
        key = gesture.upper().strip()
        entry = self._entries.get(key)

        if not entry:
            return {
                "available": False,
                "gesture": key,
                "message": _UNAVAILABLE_MSG,
            }

        if entry.get("status") != "verified":
            return {
                "available": False,
                "gesture": key,
                "message": _UNAVAILABLE_MSG,
            }

        clip_path = entry.get("animation_path", "")
        if not clip_path or not Path(clip_path).exists():
            return {
                "available": False,
                "gesture": key,
                "message": _UNAVAILABLE_MSG,
            }

        return {
            "available": True,
            "gesture": key,
            "path": clip_path,
            "source": entry.get("source", "Unknown"),
            "license": entry.get("license", "Unknown"),
            "verified_by": entry.get("verified_by", "Unknown"),
            "message": f"Verified animation for {key}",
        }

    def list_supported(self) -> list[str]:
        """Return list of gestures with verified animations."""
        return [
            k for k, v in self._entries.items()
            if v.get("status") == "verified" and Path(v.get("animation_path", "x")).exists()
        ]

    def reload(self):
        """Reload the manifest from disk."""
        self._entries.clear()
        self._load()
