"""
backend/app/services/translation_service.py
Service for translating anime synopses from English (AniList) into Brazilian Portuguese (pt-BR).
Maintains an atomic persistent cache in data/synopses_pt.json to ensure instant zero-latency delivery.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import httpx

logger = logging.getLogger("animerecsys.translation_service")


class TranslationService:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (Path(__file__).resolve().parent.parent.parent.parent / "data")
        self.cache_file = self.data_dir / "synopses_pt.json"
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Loads cached Portuguese synopses from disk."""
        if self.cache_file.exists():
            try:
                self._cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not load synopses cache: {e}")
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self) -> None:
        """Atomically persists cached synopses to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Could not save synopses cache: {e}")

    @staticmethod
    def _is_english(text: str) -> bool:
        if not text:
            return False
        markers = [" the ", " and ", " of ", " is ", " with ", " to ", " was ", " in ", " for ", " that "]
        t = f" {text.lower()} "
        return sum(1 for m in markers if m in t) >= 2

    def get_synopsis_pt(
        self,
        anime_id: int,
        english_synopsis: Optional[str],
        allow_network: bool = False,
    ) -> Optional[str]:
        """
        Retrieves the Portuguese synopsis for an anime.
        Checks local persistent cache first.
        If cache contains English or is missing, and allow_network is True,
        attempts a fast non-blocking translation to Portuguese.
        """
        key = str(anime_id)
        if key not in self._cache:
            self._load_cache()
        if key in self._cache and self._cache[key] and not self._is_english(self._cache[key]):
            return self._cache[key]

        if not english_synopsis or not english_synopsis.strip():
            return None

        # Clean HTML tags and markdown
        cleaned_en = (
            english_synopsis.replace("<br>", "\n")
            .replace("<br/>", "\n")
            .replace("<br />", "\n")
            .replace("<i>", "")
            .replace("</i>", "")
            .replace("<b>", "")
            .replace("</b>", "")
            .strip()
        )

        # Do not block during batch catalog / recommendation generation unless network allowed
        if not allow_network:
            return cleaned_en

        try:
            translated = self._translate_to_pt(cleaned_en)
            if translated and not self._is_english(translated):
                self._cache[key] = translated
                self._save_cache()
                return translated
        except Exception:
            pass

        return cleaned_en

    def set_synopsis_pt(self, anime_id: int, portuguese_text: str) -> None:
        """Manually registers a Portuguese synopsis in cache."""
        key = str(anime_id)
        self._cache[key] = portuguese_text.strip()
        self._save_cache()

    def _translate_to_pt(self, text: str) -> Optional[str]:
        """Translates text snippet from English to Portuguese with strict non-blocking timeout."""
        try:
            sample = text[:1000].strip()
            # 1. Try Google GTX translation (instant, high fidelity pt-BR)
            gtx_res = self._fetch_gtx(sample)
            if gtx_res and not self._is_english(gtx_res):
                return gtx_res

            # 2. Fallback to MyMemory
            mem_res = self._fetch_mymemory(sample[:350])
            if mem_res and not self._is_english(mem_res):
                return mem_res
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
        return None

    def _fetch_gtx(self, query: str) -> Optional[str]:
        """Performs translation via Google GTX with 3s timeout."""
        if not query.strip():
            return ""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(
                    "https://translate.googleapis.com/translate_a/single",
                    params={"client": "gtx", "sl": "en", "tl": "pt-BR", "dt": "t", "q": query.strip()},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if res.status_code == 200:
                    data = res.json()
                    translated = "".join([part[0] for part in data[0] if part and part[0]])
                    if translated:
                        return translated.strip()
        except Exception as e:
            logger.debug(f"GTX translation error: {e}")
        return None

    def _fetch_mymemory(self, query: str) -> Optional[str]:
        """Performs a single request to MyMemory API with strict 1.5s timeout."""
        if not query.strip():
            return ""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(
                    "https://api.mymemory.translated.net/get",
                    params={"q": query.strip(), "langpair": "en|pt-BR"},
                )
                if res.status_code == 200:
                    data = res.json()
                    translated = data.get("responseData", {}).get("translatedText")
                    if translated and not str(translated).startswith("MYMEMORY WARNING"):
                        return str(translated)
        except Exception:
            pass
        return None


# Global singleton instance
_translation_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service
