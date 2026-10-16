"""
backend/app/api/__init__.py
REST API routers and dependency injection providers for AnimeRecSys AI.
"""

from .deps import (
    get_anilist_service,
    get_app_settings,
    get_gemini,
    get_gemini_service,
    get_learning_engine,
    get_recommendation_engine,
    get_settings_dep,
    get_spoiler,
    get_spoiler_service,
    get_storage_manager,
    get_vector_engine,
)
from .routes import router

__all__ = [
    "router",
    "get_settings_dep",
    "get_app_settings",
    "get_storage_manager",
    "get_vector_engine",
    "get_learning_engine",
    "get_recommendation_engine",
    "get_gemini_service",
    "get_gemini",
    "get_spoiler_service",
    "get_spoiler",
    "get_anilist_service",
]
