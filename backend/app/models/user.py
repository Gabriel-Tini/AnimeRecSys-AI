"""
backend/app/models/user.py
Domain models and DTOs for user profile weights, hyperparameters, and watch history.
Maintains mathematical stability invariants for online vector learning.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from app.models.recommendation import SemanticTuningResponse
except ImportError:
    try:
        from backend.app.models.recommendation import SemanticTuningResponse
    except ImportError:
        from .recommendation import SemanticTuningResponse


class Hyperparameters(BaseModel):
    """
    Hyperparameters governing continuous learning, weight decay, and MMR diversity.
    Defaults match ARCHITECTURE.md Table 6.1 and FIXTURES.json mock_user_profile.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    eta: float = Field(
        default=0.15,
        ge=0.01,
        le=1.0,
        description="Learning rate per rating update (default 0.15)"
    )
    alpha: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Semantic boost sensitivity coefficient for Gemini tuning (default 0.25)"
    )
    gamma: float = Field(
        default=0.98,
        ge=0.50,
        le=1.0,
        description="Temporal weight decay factor applied to prior profile weights (default 0.98)"
    )
    lambda_mmr: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Maximal Marginal Relevance diversity parameter (default 0.50)"
    )


class GoogleUserAccount(BaseModel):
    """
    Represents a linked Google account used to persist and sync user progress.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    sub: str = Field(..., description="Unique Google account identifier or email hash")
    email: str = Field(..., description="User Google email address")
    name: str = Field(..., description="User display name")
    picture: Optional[str] = Field(default=None, description="Google profile avatar URL")
    connected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when Google account was connected"
    )
    last_synced_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of latest progress synchronization"
    )


class UserProfileData(BaseModel):
    """
    Persistent state of user preferences stored in data/user_profile.json.
    Tracks normalized L2 feature weights over thematic dimensions.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    user_id: str = Field(default="default_user", description="Single-user system identifier")
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the latest profile update"
    )
    total_ratings: int = Field(
        default=0,
        ge=0,
        description="Cumulative count of ratings processed"
    )
    feature_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Sparse dictionary of tag weights mapping tag_name -> weight"
    )
    hyperparameters: Hyperparameters = Field(
        default_factory=Hyperparameters,
        description="Hyperparameters governing model updates"
    )
    skipped_anime_ids: list[int] = Field(
        default_factory=list,
        description="Anime IDs skipped (not interested / not watched) with zero vector distortion"
    )
    dropped_anime_ids: list[int] = Field(
        default_factory=list,
        description="Anime IDs dropped by the user with Gemini negative semantic analysis"
    )
    favorite_anime_ids: list[int] = Field(
        default_factory=list,
        description="Anime IDs favorited by the user"
    )
    watchlist_anime_ids: list[int] = Field(
        default_factory=list,
        description="Anime IDs saved to 'Minha Lista / Quero Assistir'"
    )
    excluded_formats: list[str] = Field(
        default_factory=list,
        description="Media formats blocked from recommendations (e.g. ['MOVIE', 'OVA', 'ONA', 'SPECIAL'])"
    )
    blocked_genres: list[str] = Field(
        default_factory=list,
        description="Genres blocked by the user from appearing in recommendations (e.g. ['Mecha', 'Horror'])"
    )
    google_account: Optional[GoogleUserAccount] = Field(
        default=None,
        description="Linked Google account for cloud/local progress synchronization"
    )

    @field_validator("updated_at", mode="before")
    @classmethod
    def parse_datetime(cls, v: Any) -> datetime:
        """Handles ISO strings with 'Z' suffix or standard ISO format."""
        if isinstance(v, str):
            clean_str = v.replace("Z", "+00:00")
            return datetime.fromisoformat(clean_str)
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    @field_validator("hyperparameters", mode="before")
    @classmethod
    def parse_hyperparameters(cls, v: Any) -> Hyperparameters:
        """Allows direct dict parsing matching data/user_profile.json format."""
        if isinstance(v, dict):
            return Hyperparameters(**v)
        return v

    def is_cold_start(self) -> bool:
        """Returns True if the user has no recorded ratings or empty weights."""
        return self.total_ratings == 0 or len(self.feature_weights) == 0

    def compute_l2_norm(self) -> float:
        """Computes current Euclidean L2 norm of the feature weight vector."""
        if not self.feature_weights:
            return 0.0
        weights = np.array(list(self.feature_weights.values()), dtype=np.float64)
        return float(np.linalg.norm(weights))


class WatchHistoryEntry(BaseModel):
    """
    Interaction log entry saved in data/watch_history.json.
    Records ratings, qualitative feedback, origin track, and Gemini tuning payloads.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    id: str = Field(
        default_factory=lambda: f"wh-{uuid.uuid4().hex[:8]}",
        description="Unique history interaction ID"
    )
    anime_id: int = Field(..., gt=0, description="AniList anime identifier")
    anime_title: str = Field(..., min_length=1, description="Title of the rated anime")
    rating: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Numerical rating provided by user on [1.0, 10.0] scale"
    )
    origin_track: Literal["affinity", "discovery", "direct_search", "onboarding"] = Field(
        ...,
        description="Discovery channel that presented this anime"
    )
    interaction_status: Literal["watched", "skipped", "dropped", "watchlist"] = Field(
        default="watched",
        description="Status of user interaction (watched, skipped, dropped, watchlist)"
    )
    is_favorite: bool = Field(
        default=False,
        description="Whether the user marked this anime as a favorite"
    )
    drop_reason: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Written justification when the user dropped the anime"
    )
    user_review_raw: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Raw free-text qualitative review submitted by user (capped at 300 chars)"
    )
    qualitative_tuning: Optional[SemanticTuningResponse] = Field(
        default=None,
        description="Parsed Gemini semantic tuning response (if qualitative flow completed)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of interaction"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)
