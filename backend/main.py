"""
backend/main.py
AnimeRecSys AI — FastAPI Application Entrypoint & Composition Root.
Configures lifespan events, security middleware (CORS, logging),
global exception handlers with secret scrubbing, and REST router integration.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from app.core.config import SensitiveDataFilter, Settings, get_settings, setup_logging
    from app.core.storage_manager import (
        StorageCorruptedError,
        StorageError,
        StorageLockTimeoutError,
        StorageManager,
    )
    from app.services.anilist_service import AniListService
    from app.services.gemini_service import GeminiService
    from app.services.learning_engine import LearningEngine
    from app.services.recommendation_engine import RecommendationEngine
    from app.services.spoiler_service import SpoilerService
    from app.services.vector_engine import VectorEngine
    from app.api.routes import router as api_router
except ImportError:
    from backend.app.core.config import SensitiveDataFilter, Settings, get_settings, setup_logging
    from backend.app.core.storage_manager import (
        StorageCorruptedError,
        StorageError,
        StorageLockTimeoutError,
        StorageManager,
    )
    from backend.app.services.anilist_service import AniListService
    from backend.app.services.gemini_service import GeminiService
    from backend.app.services.learning_engine import LearningEngine
    from backend.app.services.recommendation_engine import RecommendationEngine
    from backend.app.services.spoiler_service import SpoilerService
    from backend.app.services.vector_engine import VectorEngine
    from backend.app.api.routes import router as api_router

logger = logging.getLogger("animerecsys.main")


# -----------------------------------------------------------------------------
# Application Lifespan Context Manager
# -----------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Executes startup bootstrapping and ensures graceful shutdown of tasks and resources.
    """
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    app.state.startup_time = time.time()
    app.state.background_tasks: Set[asyncio.Task] = set()

    # 1. Provision Storage Directories & Redacted Logging
    settings.ensure_directories()
    setup_logging(settings)
    logger.info("Starting up %s in %s mode...", settings.app_name, settings.environment)

    # 2. Initialize Atomic Storage Manager
    storage_manager = getattr(app.state, "storage_manager", None)
    if storage_manager is None:
        storage_manager = StorageManager(
            workspace_root=settings.workspace_root,
            data_dir=settings.data_dir,
            locks_dir=settings.locks_dir,
            fixtures_path=settings.fixtures_path,
            timeout=settings.filelock_timeout_seconds,
        )
        app.state.storage_manager = storage_manager

    # 3. Initialize AniList Service
    anilist_service = getattr(app.state, "anilist_service", None)
    if anilist_service is None:
        try:
            anilist_service = AniListService(settings=settings, storage_manager=storage_manager)
        except Exception as err:
            logger.warning("AniListService could not be loaded at startup: %s", err)
            anilist_service = None
        app.state.anilist_service = anilist_service

    # 4. Catalog Ingestion & Pre-seeding
    catalog = []
    try:
        catalog = storage_manager.load_catalog()
    except Exception as exc:
        logger.warning("Initial catalog load failed: %s. Initiating fixture recovery.", exc)

    # If catalog is absent or sparse (< 6 items), trigger AniList Seeder or fallback
    if not catalog or len(catalog) < 6:
        if anilist_service is not None and hasattr(anilist_service, "seed_catalog"):
            logger.info("Catalog empty or sparse (< 6 items). Triggering AniList Seeder...")
            try:
                # Seed with 10s timeout to prevent blocking startup
                seeded = await asyncio.wait_for(
                    anilist_service.seed_catalog(target_count=200, limit=200),
                    timeout=10.0,
                )
                if seeded:
                    catalog = seeded
            except Exception as seeder_err:
                logger.warning(
                    "AniList seeder encountered error (%s); falling back to fixtures.", seeder_err
                )
                catalog = storage_manager.load_catalog()
        else:
            catalog = storage_manager.load_catalog()

    logger.info("Active catalog size initialized: %d items.", len(catalog))
    app.state.catalog_cache = catalog

    # 5. Initialize Mathematical & LLM Singleton Services
    if not hasattr(app.state, "vector_engine") or app.state.vector_engine is None:
        try:
            vector_engine = VectorEngine.from_catalog(catalog=catalog)
        except Exception:
            vector_engine = VectorEngine.from_fixtures(settings.fixtures_path)
        app.state.vector_engine = vector_engine

    if not hasattr(app.state, "learning_engine") or app.state.learning_engine is None:
        learning_engine = LearningEngine(
            storage_manager=storage_manager,
            default_eta=settings.rec_eta,
            default_gamma=settings.rec_gamma,
            default_alpha=settings.rec_alpha,
            neutral_pivot=5.5,
        )
        app.state.learning_engine = learning_engine

    if not hasattr(app.state, "recommendation_engine") or app.state.recommendation_engine is None:
        recommendation_engine = RecommendationEngine(
            vector_engine=app.state.vector_engine,
            storage_manager=storage_manager,
            default_lambda=settings.rec_lambda_mmr,
        )
        app.state.recommendation_engine = recommendation_engine

    if not hasattr(app.state, "gemini_service") or app.state.gemini_service is None:
        gemini_service = GeminiService(settings=settings)
        app.state.gemini_service = gemini_service

    if not hasattr(app.state, "spoiler_service") or app.state.spoiler_service is None:
        spoiler_service = SpoilerService()
        app.state.spoiler_service = spoiler_service

    logger.info("All singleton services successfully bound to application state.")

    yield  # Application is running and serving requests

    # -------------------------------------------------------------------------
    # Shutdown Sequence
    # -------------------------------------------------------------------------
    logger.info("Initiating graceful shutdown of %s...", settings.app_name)

    # Drain and cancel active background tasks
    if hasattr(app.state, "background_tasks") and app.state.background_tasks:
        logger.info("Cancelling %d running background task(s)...", len(app.state.background_tasks))
        for task in list(app.state.background_tasks):
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*app.state.background_tasks, return_exceptions=True),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, Exception) as task_err:
            logger.warning("Background task cancellation timed out or had error: %s", task_err)
        app.state.background_tasks.clear()

    # Close AniList / external HTTP client sessions
    if anilist_service is not None and hasattr(anilist_service, "close"):
        try:
            res = anilist_service.close()
            if asyncio.iscoroutine(res):
                await res
        except Exception as close_err:
            logger.warning("Error closing AniList service client: %s", close_err)

    logger.info("Application shutdown complete.")


# -----------------------------------------------------------------------------
# Application Factory
# -----------------------------------------------------------------------------
def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Factory function creating and configuring the FastAPI application instance.
    Supports isolated testing configurations and custom settings overrides.
    """
    app_settings = settings or get_settings()

    app = FastAPI(
        title=app_settings.app_name,
        description="Continuous Hybrid Anime Recommendation Platform with LLM-in-the-Loop Semantic Tuning",
        version="1.0.0",
        docs_url="/docs" if app_settings.debug or app_settings.environment != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    # -------------------------------------------------------------------------
    # Middleware: CORS Strict Whitelist
    # -------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS", "HEAD", "PUT", "DELETE"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )

    # -------------------------------------------------------------------------
    # Global Exception Handlers (Secret Redaction & Standardized Schemas)
    # -------------------------------------------------------------------------
    redaction_filter = SensitiveDataFilter()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """
        Catches Pydantic request validation errors, recurses through error details
        to scrub any API keys or tokens, and returns HTTP 422 with standardized JSON.
        """
        sanitized_details = []
        for error in exc.errors():
            sanitized_details.append(redaction_filter._clean_record_arg(error))

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "message": "Invalid request parameters or payload.",
                "detail": sanitized_details,
                "status_code": 422,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """
        Catches standard HTTP exceptions (404, 400, 405, 429, etc.), sanitizes the detail text,
        and returns standardized JSON.
        """
        clean_detail = redaction_filter.redact_text(str(exc.detail))
        error_title = {
            404: "Not Found",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            405: "Method Not Allowed",
            429: "Too Many Requests",
        }.get(exc.status_code, "HTTP Error")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": error_title,
                "message": clean_detail,
                "detail": clean_detail,
                "status_code": exc.status_code,
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(StorageLockTimeoutError)
    async def storage_lock_timeout_handler(request: Request, exc: StorageLockTimeoutError) -> JSONResponse:
        """
        Catches storage concurrency timeout, logging forensic context while
        returning a clean HTTP 503 retryable response to the client.
        """
        logger.warning("Storage lock contention on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Storage Lock Timeout",
                "message": "Storage resource is busy under high concurrent contention. Please retry shortly.",
                "detail": "Data locking timeout exceeded.",
                "status_code": 503,
            },
        )

    @app.exception_handler(StorageError)
    async def storage_error_handler(request: Request, exc: StorageError) -> JSONResponse:
        """
        Catches generic storage errors and returns HTTP 500 with sanitized message.
        """
        logger.error("Storage error encountered on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Storage Error",
                "message": "An error occurred while accessing persistent storage.",
                "detail": "Persistent storage failure.",
                "status_code": 500,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all exception handler preventing internal stack trace disclosure
        and redacting sensitive parameters from application logs.
        """
        logger.exception("Unhandled application error during request to %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later.",
                "detail": "Internal server error.",
                "status_code": 500,
            },
        )

    # -------------------------------------------------------------------------
    # Core Health & Status Endpoints
    # -------------------------------------------------------------------------
    @app.get("/health", tags=["System"])
    async def direct_health_probe(request: Request) -> Dict[str, Any]:
        """Direct health probe endpoint at /health."""
        catalog = getattr(request.app.state, "catalog_cache", None)
        if catalog is None:
            sm = getattr(request.app.state, "storage_manager", None)
            catalog = sm.load_catalog() if sm else []
        start_time = getattr(request.app.state, "startup_time", time.time())
        return {
            "status": "healthy",
            "catalog_size": len(catalog),
            "version": "1.0.0",
            "gemini_configured": app_settings.is_gemini_configured,
            "uptime_seconds": round(time.time() - start_time, 2),
        }

    @app.get("/", tags=["System"])
    async def root_probe() -> Dict[str, Any]:
        """Root metadata endpoint."""
        return {
            "name": app_settings.app_name,
            "status": "running",
            "version": "1.0.0",
            "docs_url": "/docs" if app_settings.debug or app_settings.environment != "production" else None,
        }

    # -------------------------------------------------------------------------
    # REST API Router Registration
    # -------------------------------------------------------------------------
    app.include_router(api_router)
    logger.info("Successfully mounted REST API router.")

    return app


# -----------------------------------------------------------------------------
# Module Default ASGI Application Instance
# -----------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run("backend.main:app", host=cfg.host, port=cfg.port, reload=cfg.debug)
