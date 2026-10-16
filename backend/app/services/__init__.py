"""
backend/app/services/__init__.py
Core algorithmic and business logic services for AnimeRecSys AI.
"""

from .vector_engine import (
    VectorEngine,
    normalize_vector,
    l2_normalize,
    build_feature_vector,
    cosine_similarity,
    compute_similarity,
    score_candidates,
    get_cold_start_vector,
    CANONICAL_FIXTURE_TAGS,
    CANONICAL_ANILIST_GENRES,
    DEFAULT_CATEGORY_WEIGHTS,
)
from .learning_engine import LearningEngine
from .recommendation_engine import RecommendationEngine
from .gemini_service import (
    GeminiService,
    get_gemini_service,
    tune_preferences,
    apply_semantic_boost,
    apply_boost_to_vector,
    sanitize_user_feedback,
)
from .spoiler_service import (
    SpoilerService,
    is_spoiler_tag,
    filter_safe_tags,
    has_spoiler,
    sanitize_reason,
    sanitize_text,
    to_public_response,
    sanitize_anime_catalog,
    to_catalog_response,
    sanitize_recommendations,
    CANONICAL_SPOILER_TAGS,
    SAFE_THEMATIC_DESCRIPTORS,
)
from .anilist_service import (
    AniListService,
    TokenBucketRateLimiter,
    get_anilist_service,
    POPULAR_ANIME_QUERY,
    ANIME_RELATIONS_QUERY,
)

__all__ = [
    "VectorEngine",
    "normalize_vector",
    "l2_normalize",
    "build_feature_vector",
    "cosine_similarity",
    "compute_similarity",
    "score_candidates",
    "get_cold_start_vector",
    "CANONICAL_FIXTURE_TAGS",
    "CANONICAL_ANILIST_GENRES",
    "DEFAULT_CATEGORY_WEIGHTS",
    "LearningEngine",
    "RecommendationEngine",
    "GeminiService",
    "get_gemini_service",
    "tune_preferences",
    "apply_semantic_boost",
    "apply_boost_to_vector",
    "sanitize_user_feedback",
    "SpoilerService",
    "is_spoiler_tag",
    "filter_safe_tags",
    "has_spoiler",
    "sanitize_reason",
    "sanitize_text",
    "to_public_response",
    "sanitize_anime_catalog",
    "to_catalog_response",
    "sanitize_recommendations",
    "CANONICAL_SPOILER_TAGS",
    "SAFE_THEMATIC_DESCRIPTORS",
    "AniListService",
    "TokenBucketRateLimiter",
    "get_anilist_service",
    "POPULAR_ANIME_QUERY",
    "ANIME_RELATIONS_QUERY",
]

