"""
backend/app/models/anime.py
Domain models and DTOs for anime metadata, tags, catalog entities, and public responses.
Strict dual spoiler shielding enforced on AnimePublicResponse.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Set
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Canonical registry of known AniList spoiler tags for heuristic fallback
KNOWN_SPOILER_TAGS: Set[str] = {
    "Plot Twist",
    "Betrayal",
    "Tragedy",
    "Conspiracy",
    "Memory Manipulation",
}

# Pre-computed normalized lookup set (case-folded and whitespace-trimmed)
KNOWN_SPOILER_TAGS_NORMALIZED: Set[str] = {
    t.strip().lower() for t in KNOWN_SPOILER_TAGS
}

# Backward compatibility alias
KNOWN_SPOILER_TAGS_LOWER: Set[str] = KNOWN_SPOILER_TAGS_NORMALIZED



class AnimeTag(BaseModel):
    """
    Represents an AniList community-ranked tag or genre attribute.
    Internal representation preserves is_media_spoiler for vector space modeling.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: int = Field(..., description="AniList tag unique identifier")
    name: str = Field(..., min_length=1, description="Canonical name of the tag")
    rank: int = Field(
        default=100,
        ge=0,
        le=100,
        description="Community relevance score assigned by AniList in [0, 100]"
    )
    category: Optional[str] = Field(
        default=None,
        description="AniList tag category (e.g., 'Theme', 'Cast-Traits', 'Setting')"
    )
    is_media_spoiler: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_media_spoiler", "isMediaSpoiler"),
        serialization_alias="is_media_spoiler",
        description="Indicates whether this tag reveals critical plot spoilers"
    )


def is_spoiler_tag(tag: Any) -> bool:
    """
    R3 Defense: Heuristic and metadata check to determine whether a tag is a plot spoiler.
    Enforces canonical normalization (.strip().lower()) to prevent case and whitespace bypasses.
    
    Accepts:
    - AnimeTag instance
    - dict (with is_media_spoiler / isMediaSpoiler / name)
    - str (tag name)
    - Any duck-typed object with 'is_media_spoiler' or 'name'
    """
    if tag is None:
        return False

    # 1. AnimeTag model instance
    if isinstance(tag, AnimeTag):
        if tag.is_media_spoiler:
            return True
        return tag.name.strip().lower() in KNOWN_SPOILER_TAGS_NORMALIZED

    # 2. Raw dictionary representation (e.g. from JSON fixtures or GraphQL payloads)
    if isinstance(tag, dict):
        if bool(tag.get("is_media_spoiler", False) or tag.get("isMediaSpoiler", False)):
            return True
        raw_name = tag.get("name")
        if raw_name is not None:
            return str(raw_name).strip().lower() in KNOWN_SPOILER_TAGS_NORMALIZED
        return False

    # 3. Direct string tag name
    if isinstance(tag, str):
        return tag.strip().lower() in KNOWN_SPOILER_TAGS_NORMALIZED

    # 4. Duck-typing fallback
    if getattr(tag, "is_media_spoiler", False):
        return True
    if hasattr(tag, "name"):
        return str(getattr(tag, "name", "")).strip().lower() in KNOWN_SPOILER_TAGS_NORMALIZED

    return False


def extract_clean_tag_name(tag: Any) -> Optional[str]:
    """
    Extracts the sanitized string name of a tag if it is non-spoiler and non-empty.
    Trims leading/trailing whitespace. Returns None if tag is a spoiler or empty.
    """
    if is_spoiler_tag(tag):
        return None

    candidate: Optional[str] = None
    if isinstance(tag, AnimeTag):
        candidate = tag.name.strip()
    elif isinstance(tag, dict):
        raw_name = tag.get("name")
        if raw_name is not None:
            candidate = str(raw_name).strip()
    elif isinstance(tag, str):
        candidate = tag.strip()
    elif hasattr(tag, "name"):
        candidate = str(getattr(tag, "name", "")).strip()

    return candidate if candidate else None


class AnimeTitle(BaseModel):
    """
    Multi-language titles returned by AniList GraphQL API.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    romaji: str = Field(..., min_length=1, description="Romanized Japanese title")
    english: Optional[str] = Field(default=None, description="Official English translated title")
    native: Optional[str] = Field(default=None, description="Native script title (Kanji/Kana)")

    @property
    def display_title(self) -> str:
        """Returns English title if available, otherwise Romanized title."""
        return self.english or self.romaji


class AnimeEntity(BaseModel):
    """
    Complete internal anime entity stored in data/cache_catalog.json and used
    by vector_engine.py and gemini_service.py.
    Preserves all spoiler tags for maximum recommendation accuracy.
    Supports both flat JSON fixtures and nested AniList GraphQL payloads.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: int = Field(..., gt=0, description="Unique AniList media ID")
    title: Optional[AnimeTitle] = Field(default=None, description="Structured multi-language title")
    title_romaji: str = Field(default="", description="Flat Romaji title representation")
    title_english: Optional[str] = Field(default=None, description="Flat English title representation")
    title_native: Optional[str] = Field(default=None, description="Flat Native title representation")
    cover_image: str = Field(
        ...,
        validation_alias=AliasChoices("cover_image", "coverImage"),
        description="Poster cover image URL"
    )
    banner_image: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("banner_image", "bannerImage"),
        description="Panoramic banner image URL"
    )
    format: str = Field(default="TV", description="Media format (TV, MOVIE, ONA, OVA, SPECIAL)")
    episodes: Optional[int] = Field(default=None, ge=1, description="Total episode count")
    duration: Optional[int] = Field(default=None, ge=1, description="Episode runtime in minutes")
    status: str = Field(default="FINISHED", description="Airing status (FINISHED, RELEASING, etc.)")
    season_year: Optional[int] = Field(
        default=None,
        ge=1950,
        le=2100,
        validation_alias=AliasChoices("season_year", "seasonYear"),
        description="Year of broadcast release"
    )
    average_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        validation_alias=AliasChoices("average_score", "averageScore", "mean_score", "meanScore"),
        description="Community aggregate score out of 100"
    )
    popularity: int = Field(default=0, ge=0, description="Total followers / members on AniList")
    genres: List[str] = Field(default_factory=list, description="List of standard AniList genres")
    tags: List[AnimeTag] = Field(
        default_factory=list,
        description="Full list of tags including spoiler tags for internal vector modeling"
    )
    description: Optional[str] = Field(default=None, description="Synopsis or plot overview")
    prequel_ids: List[int] = Field(default_factory=list, description="List of AniList prequel/parent anime IDs")
    is_sequel: bool = Field(default=False, description="Whether this anime is a sequel or later season")

    @field_validator("cover_image", mode="before")
    @classmethod
    def extract_cover_image(cls, v: Any) -> str:
        """Handles AniList nested coverImage: {extraLarge: ..., large: ...} or direct URL string."""
        if isinstance(v, dict):
            return v.get("extraLarge") or v.get("large") or v.get("medium") or ""
        return str(v) if v else ""

    @field_validator("banner_image", mode="before")
    @classmethod
    def extract_banner_image(cls, v: Any) -> Optional[str]:
        if isinstance(v, dict):
            return v.get("bannerImage") or v.get("url")
        return str(v) if v else None

    @model_validator(mode="before")
    @classmethod
    def synchronize_titles(cls, data: Any) -> Any:
        """
        Ensures perfect interoperability between flat fixture fields (title_romaji)
        and nested AniList GraphQL fields (title: {romaji, english, native}).
        """
        if isinstance(data, dict):
            raw_title = data.get("title")
            if isinstance(raw_title, dict):
                data.setdefault("title_romaji", raw_title.get("romaji", ""))
                data.setdefault("title_english", raw_title.get("english"))
                data.setdefault("title_native", raw_title.get("native"))
            elif isinstance(raw_title, AnimeTitle):
                data.setdefault("title_romaji", raw_title.romaji)
                data.setdefault("title_english", raw_title.english)
                data.setdefault("title_native", raw_title.native)
            elif not raw_title and data.get("title_romaji"):
                data["title"] = {
                    "romaji": data["title_romaji"],
                    "english": data.get("title_english"),
                    "native": data.get("title_native"),
                }
            elif not data.get("title_romaji") and raw_title:
                data["title_romaji"] = str(raw_title)
        return data

    def to_public_response(
        self,
        compatibility_score: Optional[float] = None,
        match_percentage: Optional[int] = None,
        origin_track: Optional[Literal["affinity", "discovery"]] = None,
        reason: Optional[str] = None,
    ) -> "AnimePublicResponse":
        """
        Factory method that produces a sanitized public response with all spoiler tags stripped.
        """
        return AnimePublicResponse.from_entity(
            entity=self,
            compatibility_score=compatibility_score,
            match_percentage=match_percentage,
            origin_track=origin_track,
            reason=reason,
        )


class AnimePublicResponse(BaseModel):
    """
    Public presentation DTO returned by FastAPI endpoints (/api/catalog, /api/recommendations).
    R3 DEFENSE: Strictly sanitizes tags to ensure ZERO spoiler leakage to the UI client.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    id: int = Field(..., gt=0, description="AniList anime identifier")
    title_romaji: str = Field(..., min_length=1, description="Primary Romanized title")
    title_english: Optional[str] = Field(default=None, description="English translated title")
    cover_image: str = Field(..., description="Cover poster URL")
    banner_image: Optional[str] = Field(default=None, description="Panoramic banner URL")
    format: str = Field(default="TV", description="Media format")
    episodes: Optional[int] = Field(default=None, ge=1, description="Episode count")
    status: str = Field(default="FINISHED", description="Airing status")
    season_year: Optional[int] = Field(default=None, description="Broadcast year")
    average_score: Optional[int] = Field(default=None, ge=0, le=100, description="Average score [0, 100]")
    genres: List[str] = Field(default_factory=list, description="Publicly safe official genres")
    tags: List[str] = Field(
        default_factory=list,
        description="Sanitized list of tag names (strictly filtered, zero spoilers)"
    )
    description: Optional[str] = Field(default=None, description="Synopsis or plot overview")
    has_spoiler_tags: bool = Field(default=False, description="Whether this anime contains spoiler tags in narrative metadata")
    prequel_ids: List[int] = Field(default_factory=list, description="List of AniList prequel/parent anime IDs")
    is_sequel: bool = Field(default=False, description="Whether this anime is a sequel or later season")
    compatibility_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity affinity score in [0.0, 1.0]"
    )
    match_percentage: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Human-readable match percentage (e.g. 94%)"
    )
    origin_track: Optional[Literal["affinity", "discovery"]] = Field(
        default=None,
        description="Recommendation pipeline track origin"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Publicly safe non-spoiler justification for this recommendation"
    )

    @field_validator("tags", mode="before")
    @classmethod
    def sanitize_and_strip_spoilers(cls, v: Any) -> List[str]:
        """
        R3 Acceptance Criteria Enforcement:
        Filters out any tag where is_media_spoiler == True, and rejects any tag
        matching the canonical spoiler registry under ANY casing or whitespace variation.
        Accepts List[AnimeTag], List[dict], or List[str].
        Returns clean, trimmed, non-empty safe tag names.
        """
        if not v:
            return []
        safe_tags: List[str] = []
        for item in v:
            clean_name = extract_clean_tag_name(item)
            if clean_name is not None:
                safe_tags.append(clean_name)
        return safe_tags

    @classmethod
    def from_entity(
        cls,
        entity: AnimeEntity,
        compatibility_score: Optional[float] = None,
        match_percentage: Optional[int] = None,
        origin_track: Optional[Literal["affinity", "discovery"]] = None,
        reason: Optional[str] = None,
    ) -> "AnimePublicResponse":
        """
        Deterministic factory method guaranteeing that entity tags are stripped of all spoilers.
        """
        clean_tags: List[str] = []
        has_spoilers = False
        for t in entity.tags:
            if is_spoiler_tag(t):
                has_spoilers = True
            clean_name = extract_clean_tag_name(t)
            if clean_name is not None:
                clean_tags.append(clean_name)

        computed_pct = match_percentage
        if computed_pct is None and compatibility_score is not None:
            computed_pct = int(round(compatibility_score * 100))

        title_rom = entity.title_romaji
        if not title_rom and entity.title:
            title_rom = entity.title.romaji

        title_eng = entity.title_english
        if not title_eng and entity.title:
            title_eng = entity.title.english

        return cls(
            id=entity.id,
            title_romaji=title_rom,
            title_english=title_eng,
            cover_image=entity.cover_image,
            banner_image=entity.banner_image,
            format=entity.format,
            episodes=entity.episodes,
            status=entity.status,
            season_year=entity.season_year,
            average_score=entity.average_score,
            genres=list(entity.genres),
            tags=clean_tags,
            description=entity.description,
            has_spoiler_tags=has_spoilers,
            prequel_ids=list(getattr(entity, "prequel_ids", []) or []),
            is_sequel=bool(getattr(entity, "is_sequel", False)),
            compatibility_score=compatibility_score,
            match_percentage=computed_pct,
            origin_track=origin_track,
            reason=reason,
        )


class AnimeCatalogResponse(BaseModel):
    """
    Paginated catalog listing returned by GET /api/catalog.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    items: List[AnimePublicResponse] = Field(default_factory=list, description="Paginated items")
    total: int = Field(ge=0, description="Total count of available anime in local catalog")
    page: int = Field(default=1, ge=1, description="Current page number (1-indexed)")
    limit: int = Field(default=50, ge=1, le=100, description="Maximum items per page")
    total_pages: int = Field(default=1, ge=1, description="Total page count")
