"""
Translation service.
Supports LibreTranslate API (primary) with an offline curated dictionary fallback.
"""

import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Translates text between supported languages.

    Primary backend: LibreTranslate (self-hosted or public).
    Fallback: offline curated ISL phrase dictionary.
    """

    def __init__(
        self,
        libre_url: str,
        libre_api_key: str,
        offline_dict: dict,
        timeout: int = 5,
    ):
        self._url = libre_url.rstrip("/")
        self._key = libre_api_key
        self._offline = offline_dict  # {target_lang: {phrase: translation}}
        self._timeout = timeout
        logger.info("TranslationService ready (url=%s, languages=%s)", self._url, list(self._offline.keys()))

    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        """
        Translate text.

        Returns:
            dict with keys: translated_text, source, method ('api' | 'offline' | 'passthrough')
        """
        if not text or not text.strip():
            return {"translated_text": "", "source": text, "method": "empty"}

        if source_lang == target_lang:
            return {"translated_text": text, "source": text, "method": "passthrough"}

        # 1. Try offline dictionary first (instant & reliable)
        offline_result = self._offline_lookup(text, source_lang, target_lang)
        if offline_result:
            return {"translated_text": offline_result, "source": text, "method": "offline"}

        # 2. Try LibreTranslate API if configured
        try:
            payload = {
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text",
            }
            if self._key:
                payload["api_key"] = self._key

            resp = requests.post(
                f"{self._url}/translate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            translated = data.get("translatedText", "")
            if translated and translated != text:
                return {"translated_text": translated, "source": text, "method": "api"}
        except Exception as exc:
            logger.debug("Online translation API skipped: %s", exc)

        # 3. Final fallback: return original text rather than empty string
        return {
            "translated_text": offline_result or text,
            "source": text,
            "method": "fallback",
        }

    def _offline_lookup(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Look up a phrase in the curated offline dictionary."""
        clean = re.sub(r'[!.,?]', '', text).strip().upper()
        lang_dict = self._offline.get(target_lang, {})
        if not lang_dict:
            return None

        # Direct match
        if clean in lang_dict:
            return lang_dict[clean]

        # Multi-word translation
        words = clean.split()
        if len(words) > 1:
            translated_words = []
            for w in words:
                translated_words.append(lang_dict.get(w, w))
            return " ".join(translated_words)

        return None

    def get_supported_languages(self) -> list[str]:
        """Get supported languages."""
        return ["en", "ta", "hi", "te", "ml", "kn"]
