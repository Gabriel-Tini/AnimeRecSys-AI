"""
tests/unit/test_anilist_service.py
Comprehensive unit test suite for AniList GraphQL Seeder and Token Bucket Rate Limiter.
Validates:
- Token bucket capacity, refill rate, deficit reservation, and thread safety.
- HTTP 429 Retry-After parsing and exponential backoff circuit breaking.
- GraphQL query structural compliance (isMediaSpoiler, pagination, relations).
- Response mapping to AnimeEntity with spoiler preservation.
- Catalog seeder 4-page ingestion, deduplication, and atomic persistence.
- Transparent offline resilience and fallback to cache_catalog.json and FIXTURES.json.
- Background catalog expansion filtering out manga relations.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.config import Settings
from app.core.storage_manager import StorageManager
from app.models.anime import AnimeEntity, AnimeTag
from app.services.anilist_service import (
    ANIME_RELATIONS_QUERY,
    POPULAR_ANIME_QUERY,
    AniListService,
    TokenBucketRateLimiter,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_manager():
    """Provides an isolated StorageManager in a temporary workspace directory."""
    temp_dir = tempfile.mkdtemp(prefix="anilist_test_sm_")
    root_path = Path(temp_dir)
    fixtures_src = Path(__file__).resolve().parents[2] / ".antigravity" / "FIXTURES.json"
    fixtures_dst_dir = root_path / ".antigravity"
    fixtures_dst_dir.mkdir(parents=True, exist_ok=True)
    if fixtures_src.exists():
        shutil.copy2(fixtures_src, fixtures_dst_dir / "FIXTURES.json")

    sm = StorageManager(
        workspace_root=root_path,
        data_dir=root_path / "data",
        locks_dir=root_path / "data" / ".locks",
        fixtures_path=fixtures_dst_dir / "FIXTURES.json",
        timeout=5.0,
    )
    yield sm
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_anilist_media():
    """Returns a representative raw AniList GraphQL media JSON payload."""
    return {
        "id": 9253,
        "title": {
            "romaji": "Steins;Gate",
            "english": "Steins;Gate",
            "native": "STEINS;GATE",
        },
        "format": "TV",
        "episodes": 24,
        "duration": 24,
        "status": "FINISHED",
        "seasonYear": 2011,
        "averageScore": 90,
        "meanScore": 89,
        "popularity": 350000,
        "genres": ["Drama", "Sci-Fi", "Suspense"],
        "tags": [
            {
                "id": 101,
                "name": "Time Manipulation",
                "category": "Theme",
                "rank": 98,
                "isMediaSpoiler": False,
            },
            {
                "id": 102,
                "name": "Plot Twist",
                "category": "Theme",
                "rank": 95,
                "isMediaSpoiler": True,
            },
            {
                "id": 103,
                "name": "Tragedy",
                "category": "Theme",
                "rank": 90,
                "isMediaSpoiler": True,
            },
        ],
        "coverImage": {
            "extraLarge": "https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx9253.jpg",
            "large": "https://s4.anilist.co/file/anilistcdn/media/anime/cover/medium/bx9253.jpg",
            "medium": "https://s4.anilist.co/file/anilistcdn/media/anime/cover/small/bx9253.jpg",
        },
        "bannerImage": "https://s4.anilist.co/file/anilistcdn/media/anime/banner/9253.jpg",
        "description": "An eccentric scientist accidentally creates a time machine.",
    }


# ============================================================================
# 1. TestTokenBucketTiming (5 tests)
# ============================================================================

class TestTokenBucketTiming:
    """Verifies token bucket capacity, acquisition, and refill timing."""

    def test_initial_capacity_is_60(self):
        limiter = TokenBucketRateLimiter(capacity=60.0, refill_rate=1.0)
        assert limiter.capacity == 60.0
        assert limiter.tokens == 60.0

    def test_try_acquire_decrements_tokens(self):
        limiter = TokenBucketRateLimiter(capacity=60.0, refill_rate=1.0)
        success = limiter.try_acquire(1.0)
        assert success is True
        assert abs(limiter.tokens - 59.0) < 0.1

    def test_capacity_exhaustion(self):
        limiter = TokenBucketRateLimiter(capacity=5.0, refill_rate=0.0)
        for _ in range(5):
            assert limiter.try_acquire(1.0) is True
        assert limiter.try_acquire(1.0) is False

    def test_refill_rate_calculation(self):
        limiter = TokenBucketRateLimiter(capacity=60.0, refill_rate=10.0)
        # Drain all 60 tokens
        for _ in range(60):
            limiter.try_acquire(1.0)
        assert limiter.tokens < 1.0

        # Simulate 0.5s clock advance
        with limiter._lock:
            limiter._last_refill -= 0.5
        assert limiter.tokens >= 4.9

    def test_capacity_ceiling(self):
        limiter = TokenBucketRateLimiter(capacity=60.0, refill_rate=10.0)
        with limiter._lock:
            limiter._last_refill -= 3600.0  # 1 hour ago
        assert limiter.tokens == 60.0


# ============================================================================
# 2. TestRateLimiterConcurrency (3 tests)
# ============================================================================

class TestRateLimiterConcurrency:
    """Verifies async scheduling, virtual deficit, and multithreaded thread-safety."""

    @pytest.mark.asyncio
    async def test_async_acquire_waits_for_refill(self):
        # Fast refill rate: 50 tokens/sec
        limiter = TokenBucketRateLimiter(capacity=1.0, refill_rate=50.0)
        limiter.try_acquire(1.0)  # Empty bucket

        start = time.monotonic()
        waited = await limiter.acquire(1.0)
        elapsed = time.monotonic() - start
        assert waited == pytest.approx(0.02, abs=1e-3)
        assert elapsed >= 0.01


    @pytest.mark.asyncio
    async def test_concurrent_async_scheduling(self):
        # 10 tokens/sec refill rate
        limiter = TokenBucketRateLimiter(capacity=0.0, refill_rate=20.0)
        limiter._tokens = 0.0

        waited_times = []

        async def worker():
            w = await limiter.acquire(1.0)
            waited_times.append(w)

        tasks = [asyncio.create_task(worker()) for _ in range(5)]
        await asyncio.gather(*tasks)

        assert len(waited_times) == 5
        # Reservations should be strictly increasing due to deficit accumulation
        assert waited_times == sorted(waited_times)

    def test_multithreaded_token_safety(self):
        limiter = TokenBucketRateLimiter(capacity=100.0, refill_rate=10.0)
        acquired = []

        def worker():
            for _ in range(5):
                if limiter.try_acquire(1.0):
                    acquired.append(1)
                time.sleep(0.001)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(acquired) <= 105
        assert limiter.tokens >= 0.0


# ============================================================================
# 3. TestHttp429Backoff (4 tests)
# ============================================================================

class TestHttp429Backoff:
    """Verifies HTTP 429 Retry-After circuit breaker, exponential backoff, and recovery."""

    def test_trigger_backoff_sets_circuit_breaker(self):
        limiter = TokenBucketRateLimiter(capacity=60.0, default_retry_after=10.0)
        backoff = limiter.trigger_backoff(15.0)
        assert backoff == 15.0
        assert limiter.is_rate_limited() is True
        assert limiter.tokens == 0.0
        assert limiter.consecutive_429 == 1

    def test_exponential_backoff_on_consecutive_429(self):
        limiter = TokenBucketRateLimiter(capacity=60.0, default_retry_after=10.0)
        b1 = limiter.trigger_backoff(10.0)
        b2 = limiter.trigger_backoff(10.0)
        b3 = limiter.trigger_backoff(10.0)
        assert b1 == 10.0
        assert b2 == 20.0  # 10 * 2^1
        assert b3 == 40.0  # 10 * 2^2
        assert limiter.consecutive_429 == 3

    def test_record_success_resets_backoff_counter(self):
        limiter = TokenBucketRateLimiter(capacity=60.0, default_retry_after=10.0)
        limiter.trigger_backoff(10.0)
        limiter.trigger_backoff(10.0)
        assert limiter.consecutive_429 == 2
        limiter.record_success()
        assert limiter.consecutive_429 == 0

    def test_requests_blocked_during_backoff(self):
        limiter = TokenBucketRateLimiter(capacity=60.0, default_retry_after=60.0)
        limiter.trigger_backoff(60.0)
        assert limiter.try_acquire(1.0) is False


# ============================================================================
# 4. TestGraphQLQueryGeneration (3 tests)
# ============================================================================

class TestGraphQLQueryGeneration:
    """Verifies that GraphQL query documents match AniList schema and requirements."""

    def test_popular_anime_query_contains_required_fields(self):
        required_fields = [
            "isMediaSpoiler",
            "averageScore",
            "meanScore",
            "popularity",
            "coverImage",
            "bannerImage",
            "description",
            "POPULARITY_DESC",
        ]
        for field in required_fields:
            assert field in POPULAR_ANIME_QUERY, f"Field '{field}' missing from POPULAR_ANIME_QUERY"

    def test_relations_query_contains_both_relations_and_recommendations(self):
        assert "relations" in ANIME_RELATIONS_QUERY
        assert "recommendations" in ANIME_RELATIONS_QUERY
        assert "mediaRecommendation" in ANIME_RELATIONS_QUERY
        assert "isMediaSpoiler" in ANIME_RELATIONS_QUERY

    def test_query_variables_schema(self):
        assert "$page: Int" in POPULAR_ANIME_QUERY
        assert "$perPage: Int" in POPULAR_ANIME_QUERY
        assert "$id: Int" in ANIME_RELATIONS_QUERY


# ============================================================================
# 5. TestAniListResponseMapping (4 tests)
# ============================================================================

class TestAniListResponseMapping:
    """Verifies schema parsing from raw AniList GraphQL to AnimeEntity."""

    def test_map_nested_media_to_anime_entity(self, sample_anilist_media):
        entity = AnimeEntity.model_validate(sample_anilist_media)
        assert entity.id == 9253
        assert entity.title_romaji == "Steins;Gate"
        assert entity.title_english == "Steins;Gate"
        assert entity.cover_image == sample_anilist_media["coverImage"]["extraLarge"]
        assert entity.banner_image == sample_anilist_media["bannerImage"]
        assert len(entity.genres) == 3
        assert len(entity.tags) == 3

    def test_preserves_is_media_spoiler_in_entity(self, sample_anilist_media):
        entity = AnimeEntity.model_validate(sample_anilist_media)
        spoiler_tags = [t for t in entity.tags if t.is_media_spoiler]
        assert len(spoiler_tags) == 2
        spoiler_names = {t.name for t in spoiler_tags}
        assert "Plot Twist" in spoiler_names
        assert "Tragedy" in spoiler_names

    def test_average_score_fallback_to_mean_score(self, sample_anilist_media):
        payload = dict(sample_anilist_media)
        payload["averageScore"] = None
        payload["meanScore"] = 88
        entity = AniListService.parse_media_item(payload)
        assert entity.average_score == 88

    def test_graceful_handling_of_missing_images_and_description(self, sample_anilist_media):
        payload = dict(sample_anilist_media)
        payload["bannerImage"] = None
        payload["description"] = None
        entity = AnimeEntity.model_validate(payload)
        assert entity.banner_image is None
        assert entity.description is None


# ============================================================================
# 6. TestCatalogSeeder (4 tests)
# ============================================================================

class TestCatalogSeeder:
    """Verifies 4-page ingestion, caching early-return, and offline fallbacks."""

    @pytest.mark.asyncio
    async def test_seed_catalog_four_pages_top_200(self, temp_storage_manager, sample_anilist_media):
        # Create mock HTTP responses for 4 pages of 50 items
        def make_page_response(page: int):
            media_list = []
            for i in range(50):
                item = dict(sample_anilist_media)
                item["id"] = (page - 1) * 50 + i + 1
                item["title"] = {"romaji": f"Anime Title {item['id']}"}
                media_list.append(item)
            return {
                "data": {
                    "Page": {
                        "pageInfo": {
                            "currentPage": page,
                            "hasNextPage": page < 4,
                            "total": 200,
                        },
                        "media": media_list,
                    }
                }
            }

        client = AsyncMock()
        client.is_closed = False

        async def mock_post(url, json=None):
            page = json["variables"]["page"]
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = make_page_response(page)
            return resp

        client.post = mock_post

        service = AniListService(
            storage_manager=temp_storage_manager,
            http_client=client,
        )

        catalog = await service.seed_catalog(target_count=200, per_page=50, force=True)
        assert len(catalog) == 200
        assert service.catalog_source == "network"
        assert service.is_offline is False

        # Verify disk persistence
        persisted = temp_storage_manager.load_catalog()
        assert len(persisted) == 200

    @pytest.mark.asyncio
    async def test_seed_catalog_early_return_if_populated(self, temp_storage_manager, sample_anilist_media):
        # Seed 10 items in storage
        entities = []
        for i in range(10):
            item = dict(sample_anilist_media)
            item["id"] = 1000 + i
            entities.append(AnimeEntity.model_validate(item))
        temp_storage_manager.save_catalog(entities)

        client = AsyncMock()
        service = AniListService(
            storage_manager=temp_storage_manager,
            http_client=client,
        )

        # Target 10 items with force=False
        catalog = await service.seed_catalog(target_count=10, force=False)
        assert len(catalog) == 10
        assert service.catalog_source == "cache"
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_transparent_offline_fallback_on_network_error(self, temp_storage_manager):
        client = AsyncMock()
        client.is_closed = False
        client.post.side_effect = httpx.ConnectError("Connection refused by host")

        service = AniListService(
            storage_manager=temp_storage_manager,
            http_client=client,
        )

        # Must NOT raise exception; should fall back to fixtures
        catalog = await service.seed_catalog(target_count=200, force=True)
        assert len(catalog) >= 6
        assert service.is_offline is True
        assert service.catalog_source in ("fixtures", "cache")

    @pytest.mark.asyncio
    async def test_mock_mode_loads_fixtures_instantly(self, temp_storage_manager):
        service = AniListService(
            storage_manager=temp_storage_manager,
            mock_mode=True,
        )
        catalog = await service.seed_catalog(target_count=200, force=True)
        assert len(catalog) >= 6
        assert service.is_offline is True
        assert service.catalog_source in ("fixtures", "cache")


# ============================================================================
# 7. TestBackgroundExpansion (3 tests)
# ============================================================================

class TestBackgroundExpansion:
    """Verifies adjacent titles relations crawling, manga filtering, and error isolation."""

    @pytest.mark.asyncio
    async def test_expand_catalog_adds_adjacent_titles(self, temp_storage_manager, sample_anilist_media):
        relations_resp = {
            "data": {
                "Media": {
                    "id": 9253,
                    "title": {"romaji": "Steins;Gate"},
                    "relations": {
                        "edges": [
                            {
                                "relationType": "ALTERNATIVE",
                                "node": {
                                    **sample_anilist_media,
                                    "id": 21127,
                                    "title": {"romaji": "Steins;Gate 0"},
                                    "type": "ANIME",
                                },
                            }
                        ]
                    },
                    "recommendations": {
                        "nodes": [
                            {
                                "mediaRecommendation": {
                                    **sample_anilist_media,
                                    "id": 108632,
                                    "title": {"romaji": "Erased"},
                                    "type": "ANIME",
                                }
                            }
                        ]
                    },
                }
            }
        }

        client = AsyncMock()
        client.is_closed = False
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = relations_resp
        client.post.return_value = resp

        service = AniListService(
            storage_manager=temp_storage_manager,
            http_client=client,
        )

        added = await service.expand_catalog_for_anime(9253)
        assert added == 2

        catalog = temp_storage_manager.load_catalog()
        ids = {a.id for a in catalog}
        assert 21127 in ids
        assert 108632 in ids

    @pytest.mark.asyncio
    async def test_expand_catalog_filters_out_manga(self, temp_storage_manager, sample_anilist_media):
        relations_resp = {
            "data": {
                "Media": {
                    "id": 9253,
                    "relations": {
                        "edges": [
                            {
                                "relationType": "ADAPTATION",
                                "node": {
                                    **sample_anilist_media,
                                    "id": 47521,
                                    "title": {"romaji": "Steins;Gate Manga"},
                                    "type": "MANGA",  # Should be filtered out!
                                },
                            }
                        ]
                    },
                    "recommendations": {"nodes": []},
                }
            }
        }

        client = AsyncMock()
        client.is_closed = False
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = relations_resp
        client.post.return_value = resp

        service = AniListService(
            storage_manager=temp_storage_manager,
            http_client=client,
        )

        added = await service.expand_catalog_for_anime(9253)
        assert added == 0
        catalog = temp_storage_manager.load_catalog()
        assert 47521 not in {a.id for a in catalog}

    @pytest.mark.asyncio
    async def test_expand_catalog_network_failure_is_isolated(self, temp_storage_manager):
        client = AsyncMock()
        client.is_closed = False
        client.post.side_effect = httpx.TimeoutException("AniList timeout")

        service = AniListService(
            storage_manager=temp_storage_manager,
            http_client=client,
        )

        # Must NOT raise exception; returns 0 cleanly
        added = await service.expand_catalog_for_anime(9253)
        assert added == 0
