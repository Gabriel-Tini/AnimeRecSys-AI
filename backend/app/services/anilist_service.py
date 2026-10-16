"""
backend/app/services/anilist_service.py
AniList GraphQL Seeder, Token-Bucket Rate Limiter & Catalog Expansion Service.
Guarantees strict 60 req/min rate control, HTTP 429 Retry-After backoff,
atomic local JSON synchronization, and transparent offline fallback.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.storage_manager import StorageManager
from app.models.anime import AnimeCatalogResponse, AnimeEntity, AnimePublicResponse

logger = logging.getLogger("animerecsys.anilist_service")

# ============================================================================
# GraphQL Queries
# ============================================================================

POPULAR_ANIME_QUERY = """
query GetPopularAnime($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
      perPage
    }
    media(type: ANIME, sort: [POPULARITY_DESC]) {
      id
      title {
        romaji
        english
        native
      }
      format
      episodes
      duration
      status
      seasonYear
      averageScore
      meanScore
      popularity
      genres
      tags {
        id
        name
        category
        rank
        isMediaSpoiler
      }
      coverImage {
        extraLarge
        large
        medium
      }
      bannerImage
      description(asHtml: false)
    }
  }
}
""".strip()

ANIME_RELATIONS_QUERY = """
query GetAnimeRelations($id: Int) {
  Media(id: $id, type: ANIME) {
    id
    title {
      romaji
      english
      native
    }
    relations {
      edges {
        relationType
        node {
          id
          type
          title {
            romaji
            english
            native
          }
          format
          episodes
          duration
          status
          seasonYear
          averageScore
          meanScore
          popularity
          genres
          tags {
            id
            name
            category
            rank
            isMediaSpoiler
          }
          coverImage {
            extraLarge
            large
            medium
          }
          bannerImage
          description(asHtml: false)
        }
      }
    }
    recommendations(sort: [RATING_DESC], perPage: 10) {
      nodes {
        mediaRecommendation {
          id
          type
          title {
            romaji
            english
            native
          }
          format
          episodes
          duration
          status
          seasonYear
          averageScore
          meanScore
          popularity
          genres
          tags {
            id
            name
            category
            rank
            isMediaSpoiler
          }
          coverImage {
            extraLarge
            large
            medium
          }
          bannerImage
          description(asHtml: false)
        }
      }
    }
  }
}
""".strip()


# ============================================================================
# Token Bucket Rate Limiter
# ============================================================================

class TokenBucketRateLimiter:
    """
    Thread-safe and async-safe Token Bucket Rate Limiter.
    Enforces maximum 60 requests per minute (1.0 token/second) with HTTP 429
    Retry-After backoff and deficit reservation for concurrent coroutines.
    """

    def __init__(
        self,
        capacity: float = 60.0,
        refill_rate: float = 1.0,  # 1.0 token/second = 60 req/min
        default_retry_after: float = 60.0,
    ) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.default_retry_after = float(default_retry_after)

        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._retry_after_until = 0.0
        self._consecutive_429 = 0
        self._lock = threading.Lock()

    @property
    def tokens(self) -> float:
        """Returns the current available token count after accounting for elapsed time."""
        with self._lock:
            now = time.monotonic()
            self._refill_locked(now)
            return self._tokens

    @property
    def consecutive_429(self) -> int:
        """Returns the count of consecutive HTTP 429 backoff events."""
        with self._lock:
            return self._consecutive_429

    @property
    def retry_after_until(self) -> float:
        """Returns the monotonic timestamp until which requests are backed off."""
        with self._lock:
            return self._retry_after_until

    def _refill_locked(self, now: float) -> None:
        """Internal helper to refill tokens based on elapsed monotonic time."""
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

    def is_rate_limited(self) -> bool:
        """Checks if client is currently in an active HTTP 429 backoff period."""
        with self._lock:
            return time.monotonic() < self._retry_after_until

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Non-blocking token acquisition check.
        Returns True if tokens were available and decremented, False otherwise.
        """
        with self._lock:
            now = time.monotonic()
            if now < self._retry_after_until:
                return False
            self._refill_locked(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def acquire(self, tokens: float = 1.0) -> float:
        """
        Asynchronously acquires tokens, sleeping if the bucket has a deficit
        or if backoff is active. Uses deficit reservation to avoid thundering herds.
        Returns the duration waited in seconds.
        """
        wait_time = 0.0
        with self._lock:
            now = time.monotonic()
            base_time = max(now, self._retry_after_until)
            self._refill_locked(now)

            if self._tokens >= tokens and base_time <= now:
                self._tokens -= tokens
                return 0.0

            deficit = max(0.0, tokens - self._tokens)
            refill_delay = deficit / self.refill_rate if self.refill_rate > 0 else 0.0
            wait_time = (base_time - now) + refill_delay
            # Virtual deficit reservation: advance debt
            self._tokens -= tokens

        if wait_time > 0.0:
            await asyncio.sleep(wait_time)
        return wait_time

    def acquire_sync(self, tokens: float = 1.0) -> float:
        """
        Synchronous counterpart to acquire(), sleeping via time.sleep() if needed.
        Returns the duration waited in seconds.
        """
        wait_time = 0.0
        with self._lock:
            now = time.monotonic()
            base_time = max(now, self._retry_after_until)
            self._refill_locked(now)

            if self._tokens >= tokens and base_time <= now:
                self._tokens -= tokens
                return 0.0

            deficit = max(0.0, tokens - self._tokens)
            refill_delay = deficit / self.refill_rate if self.refill_rate > 0 else 0.0
            wait_time = (base_time - now) + refill_delay
            self._tokens -= tokens

        if wait_time > 0.0:
            time.sleep(wait_time)
        return wait_time

    def trigger_backoff(self, retry_after_seconds: Optional[float] = None) -> float:
        """
        Sets circuit-breaker backoff upon encountering an HTTP 429 response.
        Applies exponential scaling: backoff = min(300.0, base_delay * 2^consecutive_429).
        Zeroes out current bucket tokens.
        """
        with self._lock:
            now = time.monotonic()
            base_delay = (
                float(retry_after_seconds)
                if retry_after_seconds is not None and retry_after_seconds > 0
                else self.default_retry_after
            )
            exp_factor = 2 ** min(self._consecutive_429, 4)
            backoff = min(300.0, base_delay * exp_factor)
            self._retry_after_until = now + backoff
            self._tokens = 0.0
            self._consecutive_429 += 1
            logger.warning(
                "Rate limiter backoff triggered: consecutive=%d, backoff=%.1fs, resume_at=%.1f",
                self._consecutive_429,
                backoff,
                self._retry_after_until,
            )
            return backoff

    def record_success(self) -> None:
        """Resets the consecutive 429 counter after a successful request."""
        with self._lock:
            self._consecutive_429 = 0


# ============================================================================
# AniList Service
# ============================================================================

class AniListService:
    """
    AniList GraphQL client managing catalog seeding, adjacent title expansion,
    rate limiting, and transparent offline resilience.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        storage_manager: Optional[StorageManager] = None,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        mock_mode: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage_manager = storage_manager or StorageManager(
            workspace_root=self.settings.workspace_root
        )
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(
            capacity=float(self.settings.anilist_rate_limit_per_minute),
            refill_rate=float(self.settings.anilist_rate_limit_per_minute) / 60.0,
            default_retry_after=float(self.settings.anilist_retry_after_default),
        )
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self.mock_mode = mock_mode

        # Service state metadata
        self.is_offline: bool = False
        self.catalog_source: str = "uninitialized"
        self.last_sync_time: Optional[str] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization of httpx.AsyncClient."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.anilist_request_timeout_seconds),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "AnimeRecSysAI/1.0",
                },
            )
            self._owns_http_client = True
        return self._http_client

    async def close(self) -> None:
        """Closes the underlying HTTP client session if owned by this service."""
        if self._owns_http_client and self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    @classmethod
    def parse_media_item(cls, raw_item: Dict[str, Any]) -> AnimeEntity:
        """
        Parses raw AniList media dictionary into AnimeEntity.
        Handles averageScore fallback to meanScore if averageScore is null.
        """
        item = dict(raw_item)
        if item.get("averageScore") is None and item.get("meanScore") is not None:
            item["averageScore"] = item["meanScore"]
        if item.get("average_score") is None and item.get("mean_score") is not None:
            item["average_score"] = item["mean_score"]
        return AnimeEntity.model_validate(item)

    async def seed_catalog(
        self,
        target_count: int = 200,
        per_page: int = 50,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> List[AnimeEntity]:
        """
        Seeds the local catalog with popular anime titles from AniList GraphQL API.
        Default target: 200 titles across 4 pages of 50 items.
        
        Resilience guarantees:
        - If force is False and cache already contains >= target items, skips network.
        - If mock_mode is active or rate_limiter is backed off, falls back to cache/fixtures.
        - If network error (429, 5xx, timeout, ConnectError) occurs, gracefully falls back
          to cache_catalog.json or .antigravity/FIXTURES.json without raising unhandled exceptions.
        """
        target = limit if limit is not None else target_count

        # 1. Check existing catalog if not forced
        if not force and self.storage_manager.catalog_path.exists():
            existing = self.storage_manager.load_catalog()
            if len(existing) >= target:
                self.catalog_source = "cache"
                self.is_offline = False
                logger.info(
                    "Catalog already populated with %d titles (>= %d target). Seeding skipped.",
                    len(existing),
                    target,
                )
                return existing

        # 2. Check mock mode or active rate limit
        if self.mock_mode or self.rate_limiter.is_rate_limited():
            logger.info(
                "Operating in offline/mock mode (mock=%s, rate_limited=%s). Diverting to local fallback.",
                self.mock_mode,
                self.rate_limiter.is_rate_limited(),
            )
            return self._execute_offline_fallback()

        # 3. Dispatched network ingestion
        num_pages = max(1, math.ceil(target / per_page))
        fetched_items: List[AnimeEntity] = []
        network_failed = False
        client = self._get_client()

        for page in range(1, num_pages + 1):
            # Check if target already reached
            if len(fetched_items) >= target:
                break

            await self.rate_limiter.acquire(1.0)

            try:
                payload = {
                    "query": POPULAR_ANIME_QUERY,
                    "variables": {"page": page, "perPage": per_page},
                }
                resp = await client.post(self.settings.anilist_api_url, json=payload)

                if resp.status_code == 429:
                    retry_after_hdr = resp.headers.get("Retry-After")
                    retry_sec = float(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else None
                    self.rate_limiter.trigger_backoff(retry_sec)
                    network_failed = True
                    break

                if resp.status_code != 200:
                    logger.warning("AniList responded with status %d on page %d", resp.status_code, page)
                    network_failed = True
                    break

                self.rate_limiter.record_success()
                data = resp.json().get("data", {}).get("Page", {})
                media_list = data.get("media", [])
                page_info = data.get("pageInfo", {})

                for item in media_list:
                    try:
                        entity = self.parse_media_item(item)
                        fetched_items.append(entity)
                    except ValidationError as val_err:
                        logger.warning("Failed to validate AniList media item: %s", val_err)

                if not page_info.get("hasNextPage", True):
                    break

            except (httpx.RequestError, httpx.TimeoutException, Exception) as exc:
                logger.warning("Network error while seeding AniList catalog page %d: %s", page, exc)
                network_failed = True
                break

        # 4. Persistence and Fallback Logic
        if fetched_items:
            if force or not self.storage_manager.catalog_path.exists():
                merged = fetched_items
            else:
                existing_catalog = self.storage_manager.load_catalog()
                by_id: Dict[int, AnimeEntity] = {a.id: a for a in existing_catalog}
                for item in fetched_items:
                    by_id[item.id] = item
                merged = list(by_id.values())

            self.storage_manager.save_catalog(merged)
            self.is_offline = False
            self.catalog_source = "network"
            self.last_sync_time = datetime.now(timezone.utc).isoformat()
            logger.info("Successfully seeded catalog with %d titles from AniList network.", len(merged))
            return merged


        # If network failed or yielded 0 items, transparently recover
        logger.warning("Network seeding produced 0 items. Activating transparent offline fallback.")
        return self._execute_offline_fallback()

    def _execute_offline_fallback(self) -> List[AnimeEntity]:
        """Loads catalog from local cache or recovers from canonical fixtures."""
        try:
            cached = self.storage_manager.load_catalog()
            if cached and len(cached) > 0:
                self.is_offline = True
                self.catalog_source = "cache"
                return cached
        except Exception as exc:
            logger.warning("Failed loading local cache during fallback: %s", exc)

        # Recover from fixtures
        fixtures_catalog = self.storage_manager._recover_catalog_from_fixtures()
        self.is_offline = True
        self.catalog_source = "fixtures"
        return fixtures_catalog

    async def expand_catalog_for_anime(self, anime_id: int) -> int:
        """
        Asynchronous background task crawling adjacent titles (relations and recommendations).
        Filters out manga relations and deduplicates candidates.
        Saves new items atomically via storage_manager.upsert_catalog_items().
        Returns count of newly added anime.
        Completely isolated: catches all exceptions and logs warnings without raising.
        """
        if self.mock_mode or self.rate_limiter.is_rate_limited():
            return 0

        try:
            await self.rate_limiter.acquire(1.0)
            client = self._get_client()

            payload = {
                "query": ANIME_RELATIONS_QUERY,
                "variables": {"id": anime_id},
            }
            resp = await client.post(self.settings.anilist_api_url, json=payload)

            if resp.status_code == 429:
                retry_after_hdr = resp.headers.get("Retry-After")
                retry_sec = float(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else None
                self.rate_limiter.trigger_backoff(retry_sec)
                return 0

            if resp.status_code != 200:
                logger.warning(
                    "AniList relations query failed for anime %d (status %d)", anime_id, resp.status_code
                )
                return 0

            self.rate_limiter.record_success()
            data = resp.json().get("data", {}).get("Media", {})
            if not data:
                return 0

            candidate_dict: Dict[int, Any] = {}

            # 1. Process relations (strictly filter out MANGA)
            edges = data.get("relations", {}).get("edges", [])
            for edge in edges:
                node = edge.get("node")
                if node and node.get("type") == "ANIME" and "id" in node:
                    candidate_dict[node["id"]] = node

            # 2. Process recommendations (strictly filter out MANGA)
            recs = data.get("recommendations", {}).get("nodes", [])
            for rec in recs:
                rec_node = rec.get("mediaRecommendation")
                if rec_node and rec_node.get("type") == "ANIME" and "id" in rec_node:
                    candidate_dict[rec_node["id"]] = rec_node

            # 3. Validate into AnimeEntity models
            valid_entities: List[AnimeEntity] = []
            for cid, raw_media in candidate_dict.items():
                try:
                    entity = self.parse_media_item(raw_media)
                    valid_entities.append(entity)
                except ValidationError as val_err:
                    logger.debug("Validation failed for adjacent anime %d: %s", cid, val_err)

            if not valid_entities:
                return 0

            # 4. Atomic upsert to catalog
            added_count = self.storage_manager.upsert_catalog_items(valid_entities)
            logger.info(
                "Background expansion for anime %d completed: %d new titles added to catalog.",
                anime_id,
                added_count,
            )
            return added_count

        except Exception as exc:
            logger.warning("Background expansion encountered isolated error for anime %d: %s", anime_id, exc)
            return 0

    def get_catalog_page(
        self,
        page: int = 1,
        limit: int = 50,
        search: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> AnimeCatalogResponse:
        """
        Retrieves a paginated list of catalog items with optional title search and genre filter.
        Applies strict spoiler scrubbing via AnimePublicResponse (zero spoiler leakage).
        """
        catalog = self.storage_manager.load_catalog()
        filtered = catalog

        if search and search.strip():
            token = search.strip().lower()
            filtered = [
                a for a in filtered
                if (a.title_romaji and token in a.title_romaji.lower())
                or (a.title_english and token in a.title_english.lower())
                or (a.title_native and token in a.title_native.lower())
                or (a.description and token in a.description.lower())
            ]

        if genre and genre.strip():
            target_genre = genre.strip().lower()
            filtered = [
                a for a in filtered
                if any(g.strip().lower() == target_genre for g in a.genres)
            ]

        total = len(filtered)
        total_pages = max(1, math.ceil(total / limit)) if total > 0 else 1

        start = (page - 1) * limit
        end = start + limit
        page_slice = filtered[start:end]

        # Convert to public response ensuring zero spoiler leakage
        public_items: List[AnimePublicResponse] = [
            anime.to_public_response()
            for anime in page_slice
        ]

        return AnimeCatalogResponse(
            items=public_items,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
        )

    def search_anime_online(self, search_term: str, limit: int = 10) -> List[AnimeEntity]:
        """
        Performs synchronous live query to AniList GraphQL for a search term.
        Validates media items into AnimeEntity instances, auto-translates descriptions,
        persists new items to local storage, and returns the entities.
        """
        if not search_term or not search_term.strip():
            return []

        search_query = """
        query ($search: String, $perPage: Int) {
          Page(page: 1, perPage: $perPage) {
            media(search: $search, type: ANIME, sort: [POPULARITY_DESC]) {
              id
              title {
                romaji
                english
                native
              }
              format
              episodes
              duration
              status
              seasonYear
              averageScore
              meanScore
              popularity
              genres
              tags {
                id
                name
                category
                rank
                isMediaSpoiler
              }
              coverImage {
                extraLarge
                large
              }
              bannerImage
              description(asHtml: false)
            }
          }
        }
        """.strip()

        try:
            with httpx.Client(timeout=4.0) as client:
                resp = client.post(
                    self.settings.anilist_api_url,
                    json={"query": search_query, "variables": {"search": search_term.strip(), "perPage": limit}},
                    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code != 200:
                    return []

                data = resp.json().get("data", {}).get("Page", {}).get("media", [])
                if not data:
                    return []

                new_entities = []
                for item in data:
                    try:
                        entity = self.parse_media_item(item)
                        new_entities.append(entity)
                    except Exception:
                        continue

                if new_entities:
                    try:
                        from app.services.translation_service import get_translation_service
                        trans = get_translation_service()
                        for ent in new_entities:
                            if ent.description:
                                pt = trans.get_synopsis_pt(ent.id, ent.description, allow_network=True)
                                if pt:
                                    ent.description = pt
                    except Exception as trans_err:
                        logger.debug("Translation during search failed: %s", trans_err)

                    self.storage_manager.upsert_catalog_items(new_entities)
                    return new_entities

        except Exception as exc:
            logger.warning("Live AniList search failed: %s", exc)

        return []

    async def fetch_user_media_lists(
        self,
        username: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetches user anime lists (COMPLETED, PLANNING, DROPPED, CURRENT) from AniList GraphQL.
        Works with either public username or authenticated OAuth token.
        """
        client = self._get_client()
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        # If username is omitted but OAuth token is provided, resolve viewer's username
        resolved_username = username.strip() if username else None
        if not resolved_username and access_token:
            viewer_query = """
            query {
              Viewer {
                name
              }
            }
            """
            try:
                v_resp = await client.post(
                    self.settings.anilist_api_url,
                    json={"query": viewer_query},
                    headers=headers,
                    timeout=8.0,
                )
                if v_resp.status_code == 200:
                    v_data = v_resp.json().get("data", {}).get("Viewer", {})
                    resolved_username = v_data.get("name")
            except Exception as e:
                logger.warning(f"Failed to fetch Viewer with OAuth token: {e}")

        query = """
        query ($userName: String) {
          MediaListCollection(userName: $userName, type: ANIME) {
            lists {
              name
              status
              entries {
                id
                mediaId
                status
                score(format: POINT_10_DECIMAL)
                media {
                  id
                  title {
                    romaji
                    english
                    native
                  }
                  coverImage {
                    large
                    medium
                  }
                  bannerImage
                  format
                  episodes
                  status
                  seasonYear
                  averageScore
                  genres
                  description(asHtml: false)
                  tags {
                    id
                    name
                    rank
                    isMediaSpoiler
                  }
                }
              }
            }
          }
        }
        """
        payload = {
            "query": query,
            "variables": {"userName": resolved_username} if resolved_username else {},
        }
        try:
            resp = await client.post(
                self.settings.anilist_api_url,
                json=payload,
                headers=headers,
                timeout=12.0,
            )
        except httpx.TimeoutException:
            raise RuntimeError("Tempo limite de resposta da API do AniList excedido (12s). Tente novamente.")
        except Exception as exc:
            raise RuntimeError(f"Erro de conexão com AniList: {str(exc)}")

        if resp.status_code == 404:
            raise ValueError(f"Usuário '{resolved_username or 'desconhecido'}' não encontrado na AniList. Verifique o nome digitado.")
        if resp.status_code != 200:
            raise RuntimeError(f"AniList API error: status {resp.status_code} - {resp.text}")

        data = resp.json().get("data", {})
        return data

    def get_anime_by_id(self, anime_id: int) -> Optional[AnimeEntity]:
        """Returns internal AnimeEntity by ID from catalog cache, or None if not found."""
        catalog = self.storage_manager.load_catalog()
        return next((a for a in catalog if a.id == anime_id), None)

    def get_health_status(self) -> Dict[str, Any]:
        """Returns service diagnostics for health check."""
        catalog = self.storage_manager.load_catalog()
        is_limited = self.rate_limiter.is_rate_limited()
        return {
            "status": "rate_limited" if is_limited else "healthy",
            "catalog_size": len(catalog),
            "catalog_source": self.catalog_source,
            "is_offline": self.is_offline,
            "rate_limiter_tokens": round(self.rate_limiter.tokens, 2),
            "consecutive_429": self.rate_limiter.consecutive_429,
            "last_sync_time": self.last_sync_time or "never",
        }


# ============================================================================
# Singleton Accessor
# ============================================================================

_singleton_anilist_service: Optional[AniListService] = None
_singleton_lock = threading.Lock()


def get_anilist_service(
    settings: Optional[Settings] = None,
    storage_manager: Optional[StorageManager] = None,
    mock_mode: bool = False,
) -> AniListService:
    """
    Returns the singleton AniListService instance, creating it if needed.
    """
    global _singleton_anilist_service
    with _singleton_lock:
        if _singleton_anilist_service is None:
            _singleton_anilist_service = AniListService(
                settings=settings,
                storage_manager=storage_manager,
                mock_mode=mock_mode,
            )
        return _singleton_anilist_service
