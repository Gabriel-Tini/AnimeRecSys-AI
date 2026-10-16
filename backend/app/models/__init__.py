"""
backend/app/models/__init__.py
Central export for all AnimeRecSys AI domain models and DTOs.
"""

try:
    from app.models.anime import (
        AnimeCatalogResponse,
        AnimeEntity,
        AnimePublicResponse,
        AnimeTag,
        AnimeTitle,
        KNOWN_SPOILER_TAGS,
    )
    from app.models.recommendation import (
        RateRequest,
        RecommendationResponse,
        SemanticTagDelta,
        SemanticTuningResponse,
        TuneRequest,
    )
    from app.models.user import (
        Hyperparameters,
        UserProfileData,
        WatchHistoryEntry,
    )
except ImportError:
    from backend.app.models.anime import (
        AnimeCatalogResponse,
        AnimeEntity,
        AnimePublicResponse,
        AnimeTag,
        AnimeTitle,
        KNOWN_SPOILER_TAGS,
    )
    from backend.app.models.recommendation import (
        RateRequest,
        RecommendationResponse,
        SemanticTagDelta,
        SemanticTuningResponse,
        TuneRequest,
    )
    from backend.app.models.user import (
        Hyperparameters,
        UserProfileData,
        WatchHistoryEntry,
    )

__all__ = [
    # Anime models
    "AnimeTag",
    "AnimeTitle",
    "AnimeEntity",
    "AnimePublicResponse",
    "AnimeCatalogResponse",
    "KNOWN_SPOILER_TAGS",
    # User models
    "Hyperparameters",
    "UserProfileData",
    "WatchHistoryEntry",
    # Recommendation models
    "SemanticTagDelta",
    "SemanticTuningResponse",
    "RecommendationResponse",
    "RateRequest",
    "TuneRequest",
]
