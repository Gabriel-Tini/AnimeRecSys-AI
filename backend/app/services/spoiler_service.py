"""
backend/app/services/spoiler_service.py
Centralized Spoiler Policy Lifecycle, Textual Justification Guard,
and Public DTO Sanitization Service for AnimeRecSys AI.
Strictly enforces Requirement R3 (Dual Spoiler Policy) and R6 (Defensive Security).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Sequence, Set, Tuple, Union

try:
    from app.models.anime import (
        KNOWN_SPOILER_TAGS,
        AnimeCatalogResponse,
        AnimeEntity,
        AnimePublicResponse,
        AnimeTag,
    )
    from app.models.recommendation import RecommendationResponse
except ImportError:
    try:
        from backend.app.models.anime import (
            KNOWN_SPOILER_TAGS,
            AnimeCatalogResponse,
            AnimeEntity,
            AnimePublicResponse,
            AnimeTag,
        )
        from backend.app.models.recommendation import RecommendationResponse
    except ImportError:
        from ..models.anime import (
            KNOWN_SPOILER_TAGS,
            AnimeCatalogResponse,
            AnimeEntity,
            AnimePublicResponse,
            AnimeTag,
        )
        from ..models.recommendation import RecommendationResponse

logger = logging.getLogger("animerecsys.spoiler_service")

# Canonical registry of AniList narrative plot spoiler tags
CANONICAL_SPOILER_TAGS: FrozenSet[str] = frozenset({
    "Plot Twist",
    "Betrayal",
    "Tragedy",
    "Conspiracy",
    "Memory Manipulation",
})

# Safe thematic descriptors for alternative semantic replacement
SAFE_THEMATIC_DESCRIPTORS: Dict[str, str] = {
    "plot twist": "narrative progression",
    "betrayal": "interpersonal conflict",
    "tragedy": "emotional drama",
    "conspiracy": "hidden mystery",
    "memory manipulation": "psychological intrigue",
}

# Extended sensitive plot triggers tested in E2E requirements
EXTENDED_PLOT_TRIGGERS: FrozenSet[str] = frozenset({
    "morre",
    "dies",
    "killer",
    "assassino",
    "traidor",
    "traição",
    "ending",
    "final",
    "spoiler",
})


def _build_adversarial_regex(term: str) -> re.Pattern:
    """
    Builds a word-bounded, punctuation-tolerant regex for a term.
    Allows non-word separators [.\\-_*\\s]* between consecutive letters.
    Handles plural forms for terms ending in 'y' or standard 's'.
    Uses negative lookarounds (?<![a-zA-Z0-9]) and (?![a-zA-Z0-9]) to avoid
    subword false positives (e.g. 'studies' != 'dies').
    """
    clean_words = term.strip().split()
    word_patterns = []

    for word in clean_words:
        letters = [re.escape(ch) for ch in word]
        if word.lower().endswith("y"):
            # Handle tragedy -> tragedies, conspiracy -> conspiracies
            stem = "".join(f"{ch}[.\\-_*\\s]*" for ch in letters[:-1])
            word_pat = f"{stem}(?:y|ies)"
        else:
            stem = "".join(f"{ch}[.\\-_*\\s]*" for ch in letters[:-1])
            last_ch = letters[-1]
            word_pat = f"{stem}{last_ch}(?:s)?"
        word_patterns.append(word_pat)

    # Combine words with mandatory separator between words
    combined = r"[.\-_*\s]+".join(word_patterns)
    # Bounded by non-alphanumeric chars
    pattern_str = rf"(?<![a-zA-Z0-9]){combined}(?![a-zA-Z0-9])"
    return re.compile(pattern_str, flags=re.IGNORECASE)


# Pre-compile adversarial patterns for all canonical spoiler tags
COMPILED_CANONICAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (tag.lower(), _build_adversarial_regex(tag))
    for tag in CANONICAL_SPOILER_TAGS
]

# Pre-compile exact word-bounded patterns for extended triggers
COMPILED_EXTENDED_PATTERNS: List[re.Pattern] = [
    re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE)
    for w in EXTENDED_PLOT_TRIGGERS
]

# Normalized lookup sets
CANONICAL_TAGS_LOWER: FrozenSet[str] = frozenset(t.lower() for t in CANONICAL_SPOILER_TAGS)
CANONICAL_TAGS_ALPHANUMERIC: FrozenSet[str] = frozenset(
    re.sub(r"[^a-z0-9]", "", t.lower()) for t in CANONICAL_SPOILER_TAGS
)


class SpoilerService:
    """
    Centralized service managing dual spoiler policy enforcement,
    text scanning & redaction, and public DTO transformation.
    """

    @classmethod
    def normalize_tag_name(cls, name: str) -> str:
        """Trims whitespace and lowercases a tag name."""
        return name.strip().lower()

    @classmethod
    def to_alphanumeric_key(cls, name: str) -> str:
        """Strips all non-alphanumeric characters for fuzzy evasion matching."""
        return re.sub(r"[^a-z0-9]", "", name.lower())

    @classmethod
    def is_spoiler_tag(cls, tag: Any) -> bool:
        """
        R3 Acceptance Criteria:
        Determines whether a tag is a narrative plot spoiler.
        Enforces case insensitivity, whitespace trimming, and punctuation tolerance.
        """
        if tag is None:
            return False

        # 1. AnimeTag model instance
        if isinstance(tag, AnimeTag):
            if tag.is_media_spoiler:
                return True
            raw_name = tag.name
        # 2. Dictionary representation
        elif isinstance(tag, dict):
            if bool(tag.get("is_media_spoiler", False) or tag.get("isMediaSpoiler", False)):
                return True
            raw_name = tag.get("name")
        # 3. String representation
        elif isinstance(tag, str):
            raw_name = tag
        # 4. Duck-typed object
        else:
            if getattr(tag, "is_media_spoiler", False) or getattr(tag, "isMediaSpoiler", False):
                return True
            raw_name = getattr(tag, "name", None)

        if not raw_name:
            return False

        name_str = str(raw_name).strip()
        if not name_str:
            return False
        name_lower = name_str.lower()

        # Check exact lower match
        if name_lower in CANONICAL_TAGS_LOWER:
            return True

        # Check separator-normalized match
        sep_normalized = re.sub(r"[\s\-_.:/]+", " ", name_lower).strip()
        if sep_normalized in CANONICAL_TAGS_LOWER:
            return True

        # Check alphanumeric collapsed match (e.g., 'P.l.o.t T.w.i.s.t' -> 'plottwist')
        alnum_key = re.sub(r"[^a-z0-9]", "", name_lower)
        if alnum_key in CANONICAL_TAGS_ALPHANUMERIC:
            return True

        return False

    @classmethod
    def filter_safe_tags(cls, tags: Sequence[Union[AnimeTag, Dict[str, Any], str, Any]]) -> List[str]:
        """
        Filters out all spoiler tags and returns a clean list of safe tag name strings.
        Safe tag names have leading/trailing whitespace trimmed.
        """
        safe: List[str] = []
        for t in tags:
            if cls.is_spoiler_tag(t):
                continue
            name: Optional[str] = None
            if isinstance(t, AnimeTag):
                name = t.name.strip()
            elif isinstance(t, dict):
                r = t.get("name")
                if r:
                    name = str(r).strip()
            elif isinstance(t, str):
                name = t.strip()
            elif hasattr(t, "name"):
                r = getattr(t, "name")
                if r:
                    name = str(r).strip()
            if name:
                safe.append(name)
        return safe

    @classmethod
    def has_spoiler(
        cls,
        text: Optional[str],
        custom_terms: Optional[Set[str]] = None,
        check_extended: bool = False,
    ) -> bool:
        """
        Audits text and returns True if any canonical spoiler tag or pattern is found.
        """
        if not text:
            return False

        # Check pre-compiled canonical adversarial patterns
        for _, pattern in COMPILED_CANONICAL_PATTERNS:
            if pattern.search(text):
                return True

        # Check custom terms if provided
        if custom_terms:
            for term in custom_terms:
                pat = _build_adversarial_regex(term)
                if pat.search(text):
                    return True

        # Check extended trigger keywords if requested
        if check_extended:
            for ext_pat in COMPILED_EXTENDED_PATTERNS:
                if ext_pat.search(text):
                    return True

        return False

    @classmethod
    def contains_spoiler_keywords(cls, text: Optional[str]) -> bool:
        """Audits text and returns True if any canonical or extended spoiler pattern is found."""
        return cls.has_spoiler(text, check_extended=True)

    @classmethod
    def sanitize_reason(
        cls,
        text: Optional[str],
        replacement: str = "[redacted]",
        custom_terms: Optional[Set[str]] = None,
        descriptors: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        R3/R6 Acceptance Criteria Enforcement:
        Audits explanation or review text and replaces sensitive spoiler terms.
        Applies word-bounded, punctuation-tolerant pattern matching.
        """
        if not text:
            return ""

        result = text

        # 1. Redact or replace canonical spoiler tags
        for tag_lower, pattern in COMPILED_CANONICAL_PATTERNS:
            repl = descriptors.get(tag_lower, replacement) if descriptors else replacement
            result = pattern.sub(repl, result)

        # 2. Redact custom terms if supplied
        if custom_terms:
            for term in custom_terms:
                pat = _build_adversarial_regex(term)
                repl = descriptors.get(term.lower(), replacement) if descriptors else replacement
                result = pat.sub(repl, result)

        return result

    @classmethod
    def sanitize_text(
        cls,
        text: Optional[str],
        spoiler_terms: Optional[Set[str]] = None,
        replacement: str = "[redacted]",
    ) -> str:
        """
        Alias matching the interface contract in PROJECT.md § Interface Contracts.
        """
        return cls.sanitize_reason(text, replacement=replacement, custom_terms=spoiler_terms)

    @classmethod
    def to_public_response(
        cls,
        anime: AnimeEntity,
        compatibility_score: Optional[float] = None,
        match_percentage: Optional[int] = None,
        origin_track: Optional[Literal["affinity", "discovery"]] = None,
        reason: Optional[str] = None,
    ) -> AnimePublicResponse:
        """
        Deterministic factory transforming an internal AnimeEntity into a spoiler-safe
        AnimePublicResponse. Guarantees zero spoiler leakage in tags and reason.
        """
        clean_tags = cls.filter_safe_tags(anime.tags)

        # Sanitize reason if provided
        safe_reason = cls.sanitize_reason(reason) if reason else None

        # Compute match percentage if not explicitly provided
        computed_pct = match_percentage
        if computed_pct is None and compatibility_score is not None:
            computed_pct = int(round(compatibility_score * 100))

        # Title resolution
        title_rom = anime.title_romaji
        if not title_rom and anime.title:
            title_rom = anime.title.romaji

        title_eng = anime.title_english
        if not title_eng and anime.title:
            title_eng = anime.title.english

        has_spoilers = any(cls.is_spoiler_tag(t) for t in anime.tags) or cls.contains_spoiler_keywords(anime.description or "")

        desc_pt = None
        if anime.description:
            try:
                from app.services.translation_service import get_translation_service
                desc_pt = get_translation_service().get_synopsis_pt(anime.id, anime.description)
            except Exception:
                desc_pt = anime.description

        return AnimePublicResponse(
            id=anime.id,
            title_romaji=title_rom,
            title_english=title_eng,
            cover_image=anime.cover_image,
            banner_image=anime.banner_image,
            format=anime.format,
            episodes=anime.episodes,
            status=anime.status,
            season_year=anime.season_year,
            average_score=anime.average_score,
            genres=list(anime.genres),
            tags=clean_tags,
            description=desc_pt or anime.description,
            has_spoiler_tags=has_spoilers,
            compatibility_score=compatibility_score,
            match_percentage=computed_pct,
            origin_track=origin_track,
            reason=safe_reason,
        )

    @classmethod
    def sanitize_anime_catalog(
        cls,
        catalog: Sequence[AnimeEntity],
    ) -> List[AnimePublicResponse]:
        """Batch transforms catalog entities into public DTOs with all spoilers stripped."""
        return [cls.to_public_response(entity) for entity in catalog]

    @classmethod
    def to_catalog_response(
        cls,
        catalog: Sequence[AnimeEntity],
        total: Optional[int] = None,
        page: int = 1,
        limit: int = 50,
    ) -> AnimeCatalogResponse:
        """Generates a paginated AnimeCatalogResponse with sanitized public items."""
        total_count = total if total is not None else len(catalog)
        public_items = cls.sanitize_anime_catalog(catalog)
        total_pages = max(1, (total_count + limit - 1) // limit)

        return AnimeCatalogResponse(
            items=public_items,
            total=total_count,
            page=page,
            limit=limit,
            total_pages=total_pages,
        )

    @classmethod
    def sanitize_recommendations(
        cls,
        response: RecommendationResponse,
    ) -> RecommendationResponse:
        """
        Validates and sanitizes a complete RecommendationResponse, ensuring that
        both affinity_track and discovery_track have sanitized tags and reasons.
        """
        for item in response.affinity_track:
            item.tags = cls.filter_safe_tags(item.tags)
            if item.reason:
                item.reason = cls.sanitize_reason(item.reason)

        for item in response.discovery_track:
            item.tags = cls.filter_safe_tags(item.tags)
            if item.reason:
                item.reason = cls.sanitize_reason(item.reason)

        return response


# Module-level convenience functions
is_spoiler_tag = SpoilerService.is_spoiler_tag
filter_safe_tags = SpoilerService.filter_safe_tags
has_spoiler = SpoilerService.has_spoiler
sanitize_reason = SpoilerService.sanitize_reason
sanitize_text = SpoilerService.sanitize_text
to_public_response = SpoilerService.to_public_response
sanitize_anime_catalog = SpoilerService.sanitize_anime_catalog
to_catalog_response = SpoilerService.to_catalog_response
sanitize_recommendations = SpoilerService.sanitize_recommendations
