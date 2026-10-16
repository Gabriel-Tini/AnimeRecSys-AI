"""
tests/unit/test_main_app.py
Comprehensive unit test suite for FastAPI Application Entrypoint, Lifespan & Security Middleware.
Verifies:
- Lifespan startup directory creation, catalog pre-seeding, fallback resilience, and singletons.
- Lifespan shutdown background task draining and AniList client closure.
- CORS strict whitelist (localhost:5173, 127.0.0.1:5173), wildcard rejection, preflight headers.
- Global exception handlers: 404, 405, 422, 500, 503, and Gemini API key scrubbing.
- Health and root endpoint responses.
- AsyncClient integration and dependency injection overrides.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from httpx import ASGITransport, AsyncClient
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.deps import get_vector_engine
from app.core.config import SensitiveDataFilter, Settings
from app.core.storage_manager import (
    StorageCorruptedError,
    StorageError,
    StorageLockTimeoutError,
    StorageManager,
)
from app.models.anime import AnimeEntity
from app.services.anilist_service import AniListService
from app.services.gemini_service import GeminiService
from app.services.learning_engine import LearningEngine
from app.services.recommendation_engine import RecommendationEngine
from app.services.spoiler_service import SpoilerService
from app.services.vector_engine import VectorEngine
from backend.main import create_app


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_workspace_dir():
    """Hermetic isolated directory for application testing."""
    tmp_dir = tempfile.mkdtemp(prefix="main_app_test_")
    root_path = Path(tmp_dir)

    # Copy fixtures
    fixtures_src = Path(__file__).resolve().parents[2] / ".antigravity" / "FIXTURES.json"
    fixtures_dst_dir = root_path / ".antigravity"
    fixtures_dst_dir.mkdir(parents=True, exist_ok=True)
    if fixtures_src.exists():
        shutil.copy2(fixtures_src, fixtures_dst_dir / "FIXTURES.json")

    yield root_path
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def hermetic_settings(temp_workspace_dir):
    """Provides hermetic settings pointing to temp_workspace_dir."""
    return Settings(
        custom_workspace_root=temp_workspace_dir,
        cors_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        debug=True,
        log_level="DEBUG",
        filelock_timeout_seconds=5.0,
    )


@pytest.fixture
def app_instance(hermetic_settings):
    """Creates a configured FastAPI application instance."""
    return create_app(hermetic_settings)


@pytest.fixture
def test_client(app_instance):
    """Synchronous test client with automated lifespan context manager."""
    with TestClient(app_instance, raise_server_exceptions=False) as client:
        yield client


# ============================================================================
# Class 1: TestLifespanStartupAndShutdown (6 Tests)
# ============================================================================

class TestLifespanStartupAndShutdown:
    """Verifies startup directory creation, singletons, and shutdown cleanup."""

    def test_lifespan_creates_storage_directories(self, hermetic_settings):
        data_dir = hermetic_settings.data_dir
        locks_dir = hermetic_settings.locks_dir
        if data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)

        app = create_app(hermetic_settings)
        with TestClient(app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200

        assert data_dir.exists()
        assert locks_dir.exists()

    def test_lifespan_binds_singletons_to_app_state(self, app_instance):
        with TestClient(app_instance) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200

            state = app_instance.state
            assert isinstance(state.storage_manager, StorageManager)
            assert isinstance(state.vector_engine, VectorEngine)
            assert isinstance(state.learning_engine, LearningEngine)
            assert isinstance(state.recommendation_engine, RecommendationEngine)
            assert isinstance(state.gemini_service, GeminiService)
            assert isinstance(state.spoiler_service, SpoilerService)

    def test_lifespan_preseeds_catalog_from_existing_cache(self, hermetic_settings):
        # Pre-seed cache_catalog.json
        sm = StorageManager(workspace_root=hermetic_settings.workspace_root)
        catalog = sm.load_catalog()  # Loads from fixtures
        assert len(catalog) >= 6

        app = create_app(hermetic_settings)
        with TestClient(app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200
            assert len(app.state.catalog_cache) >= 6

    def test_lifespan_seeder_fallback_on_network_failure(self, hermetic_settings):
        app = create_app(hermetic_settings)

        # Mock seed_catalog to raise ConnectError
        with patch.object(
            AniListService,
            "seed_catalog",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Network unreachable"),
        ):
            with TestClient(app) as client:
                resp = client.get("/api/health")
                assert resp.status_code == 200
                assert app.state.catalog_cache is not None
                assert len(app.state.catalog_cache) >= 6

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_cancels_background_tasks(self, hermetic_settings):
        app = create_app(hermetic_settings)
        task_ref = None

        async with app.router.lifespan_context(app):
            async def dummy_job():
                await asyncio.sleep(100)

            task = asyncio.create_task(dummy_job())
            app.state.background_tasks.add(task)
            task_ref = task

        # Upon lifespan exit
        assert task_ref.cancelled() is True or task_ref.done()
        assert len(app.state.background_tasks) == 0


    def test_lifespan_shutdown_closes_anilist_client(self, hermetic_settings):
        app = create_app(hermetic_settings)
        close_mock = AsyncMock()

        with TestClient(app) as client:
            client.get("/api/health")
            if app.state.anilist_service:
                app.state.anilist_service.close = close_mock

        close_mock.assert_awaited_once()


# ============================================================================
# Class 2: TestCORSMiddleware (5 Tests)
# ============================================================================

class TestCORSMiddleware:
    """Verifies strict CORS origin whitelist and security rejections."""

    def test_cors_preflight_allowed_origin_localhost(self, test_client):
        resp = test_client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert resp.headers.get("access-control-allow-credentials") == "true"

    def test_cors_preflight_allowed_origin_127(self, test_client):
        resp = test_client.options(
            "/api/health",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"

    def test_cors_preflight_disallowed_origin_rejected(self, test_client):
        resp = test_client.options(
            "/api/health",
            headers={
                "Origin": "http://evil-hacker.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_simple_get_headers_injected(self, test_client):
        resp = test_client.get(
            "/api/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_wildcard_rejection_at_settings_level(self):
        with pytest.raises(ValueError, match="wildcard origin '\\*' is strictly prohibited"):
            Settings(cors_origins=["*"])


# ============================================================================
# Class 3: TestGlobalExceptionHandlers (6 Tests)
# ============================================================================

class TestGlobalExceptionHandlers:
    """Verifies standardized JSON error formats and secret scrubbing."""

    def test_404_not_found_standardized_json(self, test_client):
        resp = test_client.get("/api/nonexistent_route_abc123")
        assert resp.status_code == 404
        data = resp.json()
        assert data == {
            "error": "Not Found",
            "message": "Not Found",
            "detail": "Not Found",
            "status_code": 404,
        }

    def test_405_method_not_allowed_standardized_json(self, test_client):
        resp = test_client.post("/api/health")
        assert resp.status_code == 405
        data = resp.json()
        assert data["status_code"] == 405
        assert data["error"] == "Method Not Allowed"

    def test_422_request_validation_error_standardized_json(self, test_client):
        resp = test_client.post("/api/rate", json={"score": "not_a_number"})
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"] == "Validation Error"
        assert isinstance(data["detail"], list)
        assert data["status_code"] == 422

    def test_422_validation_error_scrubs_gemini_api_keys(self, test_client):
        payload = {"score": "AIzaSyA1234567890abcdefghijklmnopqrstuv"}
        resp = test_client.post("/api/rate", json=payload)
        assert resp.status_code == 422
        assert "AIzaSy" not in resp.text
        assert "[REDACTED_GEMINI_KEY]" in resp.text

    def test_500_uncaught_exception_standardized_json_and_log_scrubbing(self, hermetic_settings):
        app = create_app(hermetic_settings)

        @app.get("/test_crash")
        def crash_route():
            raise RuntimeError("Unexpected error with key AIzaSyCrashKey1234567890abcdefgh")

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/test_crash")
            assert resp.status_code == 500
            assert "AIzaSy" not in resp.text
            assert resp.json() == {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later.",
                "detail": "Internal server error.",
                "status_code": 500,
            }

    def test_storage_lock_timeout_returns_503(self, hermetic_settings):
        app = create_app(hermetic_settings)

        @app.get("/test_lock")
        def lock_route():
            raise StorageLockTimeoutError(Path("data.json"), Path("data.lock"), 10.0)

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/test_lock")
            assert resp.status_code == 503
            data = resp.json()
            assert data["error"] == "Storage Lock Timeout"
            assert data["status_code"] == 503


# ============================================================================
# Class 4: TestHealthAndRootEndpoints (4 Tests)
# ============================================================================

class TestHealthAndRootEndpoints:
    """Verifies health probes, metadata, and OpenAPI availability."""

    def test_health_probe_returns_200_and_catalog_size(self, test_client):
        resp = test_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "healthy")
        assert data["catalog_size"] >= 6
        assert data["version"] == "1.0.0"

    def test_health_probe_alias_on_root_health(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "healthy")
        assert data["catalog_size"] >= 6
        assert data["version"] == "1.0.0"

    def test_root_endpoint_metadata(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "AnimeRecSys AI"
        assert data["status"] == "running"
        assert data["version"] == "1.0.0"

    def test_openapi_and_docs_availability(self, test_client):
        resp_docs = test_client.get("/docs")
        assert resp_docs.status_code == 200
        resp_openapi = test_client.get("/openapi.json")
        assert resp_openapi.status_code == 200


# ============================================================================
# Class 5: TestAsyncClientIntegration (4 Tests)
# ============================================================================

class TestAsyncClientIntegration:
    """Verifies asynchronous client operations and dependency overrides."""

    @pytest.mark.asyncio
    async def test_async_client_health_probe(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["status"] in ("ok", "healthy")

    @pytest.mark.asyncio
    async def test_async_client_cors_preflight(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.options(
                "/api/health",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert resp.status_code == 200
            assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    @pytest.mark.asyncio
    async def test_async_client_404_handler(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/nonexistent_xyz")
            assert resp.status_code == 404
            assert resp.json()["error"] == "Not Found"

    def test_dependency_injection_override(self, app_instance):
        mock_vec_engine = MagicMock()
        mock_vec_engine.dimension = 42

        app_instance.dependency_overrides[get_vector_engine] = lambda: mock_vec_engine
        with TestClient(app_instance) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200

        app_instance.dependency_overrides.clear()
