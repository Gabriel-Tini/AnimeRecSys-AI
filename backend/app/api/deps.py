"""
backend/app/api/deps.py
FastAPI dependency injection providers for AnimeRecSys AI.
Provides injectable singletons and factory instances for storage,
engines, services, and system configuration.
Supports seamless test mocking via app.dependency_overrides.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from fastapi import Request

from app.core.config import Settings, get_settings
from app.core.storage_manager import StorageManager
from app.services.anilist_service import AniListService, get_anilist_service as _get_anilist_singleton
from app.services.gemini_service import GeminiService, get_gemini_service as _get_gemini_singleton
from app.services.learning_engine import LearningEngine
from app.services.recommendation_engine import RecommendationEngine
from app.services.spoiler_service import SpoilerService
from app.services.vector_engine import VectorEngine


# -----------------------------------------------------------------------------
# Internal Fallback Singletons (when app.state is not populated)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_fallback_storage_manager() -> StorageManager:
    settings = get_settings()
    return StorageManager(
        workspace_root=settings.workspace_root,
        data_dir=settings.data_dir,
        locks_dir=settings.locks_dir,
        fixtures_path=settings.fixtures_path,
        timeout=settings.filelock_timeout_seconds,
    )


@lru_cache(maxsize=1)
def _get_fallback_vector_engine() -> VectorEngine:
    settings = get_settings()
    return VectorEngine.from_fixtures(
        fixtures_path=settings.fixtures_path,
        include_genres=True,
    )


# -----------------------------------------------------------------------------
# Public Dependency Providers
# -----------------------------------------------------------------------------

def get_settings_dep(request: Request) -> Settings:
    """Provides application Settings singleton."""
    state = getattr(getattr(request, "app", None), "state", None)
    if state and hasattr(state, "settings") and state.settings is not None:
        return state.settings
    return get_settings()


# Backward compatibility alias
get_app_settings = get_settings_dep


def get_storage_manager(request: Request) -> StorageManager:
    """
    Returns the StorageManager instance from request.app.state if available,
    otherwise falls back to the configured singleton StorageManager.
    """
    state = getattr(getattr(request, "app", None), "state", None)
    if state and hasattr(state, "storage_manager") and state.storage_manager is not None:
        return state.storage_manager
    return _get_fallback_storage_manager()


def get_vector_engine(request: Request) -> VectorEngine:
    """
    Returns the VectorEngine instance from request.app.state if available,
    otherwise falls back to the canonical fixtures VectorEngine.
    """
    state = getattr(getattr(request, "app", None), "state", None)
    if state and hasattr(state, "vector_engine") and state.vector_engine is not None:
        return state.vector_engine
    return _get_fallback_vector_engine()


def get_learning_engine(request: Request) -> LearningEngine:
    """
    Provides LearningEngine wired with active StorageManager and settings.
    """
    state = getattr(getattr(request, "app", None), "state", None)
    if state and hasattr(state, "learning_engine") and state.learning_engine is not None:
        return state.learning_engine

    storage = get_storage_manager(request)
    settings = get_settings_dep(request)
    return LearningEngine(
        storage_manager=storage,
        default_eta=settings.rec_eta,
        default_gamma=settings.rec_gamma,
        default_alpha=settings.rec_alpha,
    )


def get_recommendation_engine(request: Request) -> RecommendationEngine:
    """
    Provides RecommendationEngine wired with active VectorEngine and StorageManager.
    """
    state = getattr(getattr(request, "app", None), "state", None)
    if state and hasattr(state, "recommendation_engine") and state.recommendation_engine is not None:
        return state.recommendation_engine

    vector_engine = get_vector_engine(request)
    storage = get_storage_manager(request)
    settings = get_settings_dep(request)
    return RecommendationEngine(
        vector_engine=vector_engine,
        storage_manager=storage,
        default_lambda=settings.rec_lambda_mmr,
    )


def get_gemini_service(request: Request) -> GeminiService:
    """Provides Google Gemini semantic tuning service."""
    state = getattr(getattr(request, "app", None), "state", None)
    if state and hasattr(state, "gemini_service") and state.gemini_service is not None:
        return state.gemini_service
    return _get_gemini_singleton()


# Backward compatibility alias
get_gemini = get_gemini_service


def get_spoiler_service(request: Request) -> SpoilerService:
    """Provides dual spoiler shielding and text sanitization service."""
    state = getattr(getattr(request, "app", None), "state", None)
    if state and hasattr(state, "spoiler_service") and state.spoiler_service is not None:
        return state.spoiler_service
    return SpoilerService()


# Backward compatibility alias
get_spoiler = get_spoiler_service


def get_anilist_service(request: Request) -> Optional[AniListService]:
    """Provides AniList GraphQL seeder and rate limiter service."""
    state = getattr(getattr(request, "app", None), "state", None)
    if state and hasattr(state, "anilist_service") and state.anilist_service is not None:
        return state.anilist_service
    settings = get_settings_dep(request)
    storage = get_storage_manager(request)
    return _get_anilist_singleton(settings=settings, storage_manager=storage)


# Backward compatibility alias
get_anilist_service_dep = get_anilist_service
