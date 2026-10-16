"""
backend/app/models/recommendation.py
Domain models and DTOs for recommendations, Gemini semantic tuning, and API requests.
Enforces Layer 4 prompt injection clamping and 80/20 disjoint-track invariant.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any, List, Literal, Optional, Set
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

try:
    from app.models.anime import AnimePublicResponse, KNOWN_SPOILER_TAGS
except ImportError:
    try:
        from backend.app.models.anime import AnimePublicResponse, KNOWN_SPOILER_TAGS
    except ImportError:
        from .anime import AnimePublicResponse, KNOWN_SPOILER_TAGS


class SemanticTagDelta(BaseModel):
    """
    Represents an extracted semantic boost for a specific canonical tag.
    Structured output target for Google Gemini LLM API (google-genai).
    R6 DEFENSE: Clamps numeric outputs to safe mathematical boundaries.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    tag_name: str = Field(..., min_length=1, description="Canonical AniList tag name")
    impact_multiplier: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Impact direction and strength in [-1.0, 1.0]"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction certainty score in [0.0, 1.0]"
    )
    rationale: str = Field(
        default="",
        description="Concise rationale for technical audit log"
    )

    @field_validator("impact_multiplier", mode="before")
    @classmethod
    def clamp_impact_multiplier(cls, v: Any) -> float:
        """
        R6 Defensive clamping: bounds impact to [-1.0, 1.0].
        Neutralizes NaN inputs to safe neutral 0.0 to prevent unintentional max-boost exploits.
        Clamps extreme finite/infinite values to boundaries [-1.0, 1.0].
        """
        try:
            val = float(v)
            if math.isnan(val):
                return 0.0
            return max(-1.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        """
        R6 Defensive clamping: bounds confidence to [0.0, 1.0].
        Neutralizes NaN inputs to safe neutral 0.0.
        Clamps extreme finite/infinite values to boundaries [0.0, 1.0].
        """
        try:
            val = float(v)
            if math.isnan(val):
                return 0.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0


class SemanticTuningResponse(BaseModel):
    """
    Complete structured output schema bound to Google Gemini API (gemini-2.5-flash).
    Contains extracted tag adjustments and a non-spoiler public reason.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    identified_tags: List[SemanticTagDelta] = Field(
        default_factory=list,
        description="Extracted canonical tag adjustments"
    )
    public_safe_reason: str = Field(
        ...,
        description="User-facing concise justification explaining recommendation affinity without spoilers"
    )

    @field_validator("public_safe_reason")
    @classmethod
    def audit_spoiler_in_reason(cls, v: str) -> str:
        """
        R3/R6 Acceptance Criteria Enforcement:
        Heuristic check ensuring no explicit spoiler tag term appears in the public reason.
        Uses case-insensitive regex substitution to redact all casing variations
        (e.g., 'PLOT TWIST', 'Plot Twist', 'plot twist', 'pLoT tWiSt').
        """
        if not v:
            return v
        for spoiler in KNOWN_SPOILER_TAGS:
            spoiler_clean = str(spoiler).strip()
            if not spoiler_clean:
                continue
            pattern = re.compile(re.escape(spoiler_clean), flags=re.IGNORECASE)
            v = pattern.sub("[redacted]", v)
        return v


class RecommendationResponse(BaseModel):
    """
    Recommendation payload returned by GET /api/recommendations and POST /api/rate.
    MATHEMATICAL INVARIANT: S_aff ∩ S_disc = ∅ (Affinity and Discovery tracks are strictly disjoint).
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    affinity_track: List[AnimePublicResponse] = Field(
        default_factory=list,
        description="80% Affinity exploitation track items"
    )
    discovery_track: List[AnimePublicResponse] = Field(
        default_factory=list,
        description="20% Discovery exploration MMR track items"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when recommendations were computed"
    )

    @field_validator("generated_at", mode="before")
    @classmethod
    def parse_generated_at(cls, v: Any) -> datetime:
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    @model_validator(mode="after")
    def validate_disjoint_tracks(self) -> Self:
        """
        Enforces mathematical requirement R1 and Acceptance Criteria:
        No anime may appear in both affinity_track and discovery_track.
        """
        aff_ids: Set[int] = {item.id for item in self.affinity_track}
        disc_ids: Set[int] = {item.id for item in self.discovery_track}
        overlap = aff_ids & disc_ids
        if overlap:
            raise ValueError(
                f"Disjoint-track invariant violated: Anime IDs {overlap} appear in both "
                f"affinity_track and discovery_track (S_aff ∩ S_disc must be ∅)."
            )
        return self


class RateRequest(BaseModel):
    """
    Request DTO for POST /api/rate.
    Accepts both 'score' and 'rating', 'origin_track', and optional qualitative review.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    anime_id: int = Field(..., gt=0, description="AniList anime identifier")
    score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        validation_alias=AliasChoices("score", "rating"),
        description="Numerical score in [1.0, 10.0]"
    )
    origin_track: Literal["affinity", "discovery", "direct_search", "onboarding"] = Field(
        default="direct_search",
        description="Track where the user discovered this title"
    )
    user_review: Optional[str] = Field(
        default=None,
        max_length=300,
        validation_alias=AliasChoices("user_review", "qualitative_feedback", "user_review_raw"),
        description="Optional qualitative free-text comment (max 300 characters)"
    )


class TuneRequest(BaseModel):
    """
    Request DTO for POST /api/tune.
    Triggers Google Gemini qualitative analysis on user feedback.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    anime_id: int = Field(..., gt=0, description="AniList anime identifier")
    user_feedback: str = Field(
        ...,
        min_length=1,
        max_length=300,
        validation_alias=AliasChoices("user_feedback", "feedback", "review"),
        description="Qualitative feedback string capped at 300 characters"
    )
