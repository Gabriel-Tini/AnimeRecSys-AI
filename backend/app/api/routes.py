"""
backend/app/api/routes.py
FastAPI Router for AnimeRecSys AI REST API.
Implements:
- GET /api/catalog (paginated catalog listing with search query parameter q, genre filter, safe public response DTOs with zero spoiler leakage)
- GET /api/recommendations (returns RecommendationResponse with 80% Affinity Track and 20% Discovery Track MMR lambda=0.50, mutually disjoint, watch history excluded, zero spoiler tags)
- POST /api/rate (submits rating 1.0-10.0 and optional review text, updates user profile vector and records watch history, returns updated profile and refreshed recommendations)
- POST /api/tune (two-turn qualitative tuning endpoint with natural language feedback, invokes GeminiService.tune_profile, boosts profile vector alpha=0.25, enriches watch history entry, and returns updated profile)
- GET /api/profile (returns UserProfileData: weights, hyperparameters, total_ratings)
- POST /api/reset (resets user profile to cold start and clears watch history)
- GET /api/health (returns status 'ok', catalog size, cache freshness, and Gemini status)
"""

from __future__ import annotations

from datetime import datetime, timezone
import base64
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.api.deps import (
    get_anilist_service,
    get_anilist_service_dep,
    get_app_settings,
    get_gemini,
    get_gemini_service,
    get_learning_engine,
    get_recommendation_engine,
    get_settings_dep,
    get_spoiler,
    get_spoiler_service,
    get_storage_manager,
)
from app.core.config import Settings
from app.core.storage_manager import StorageManager
from app.models.anime import AnimeCatalogResponse, AnimeEntity, AnimePublicResponse
from app.models.recommendation import (
    RecommendationResponse,
    SemanticTagDelta,
    SemanticTuningResponse,
)
from app.models.user import GoogleUserAccount, UserProfileData, WatchHistoryEntry
from app.services.anilist_service import AniListService
from app.services.gemini_service import GeminiService
from app.services.learning_engine import LearningEngine
from app.services.recommendation_engine import RecommendationEngine
from app.services.spoiler_service import SpoilerService

router = APIRouter(prefix="/api", tags=["AnimeRecSys API"])


# ============================================================================
# Request & Response DTOs
# ============================================================================

class RatingRequest(BaseModel):
    """
    Request DTO for POST /api/rate.
    Accepts ratings on [1.0, 10.0] scale with aliases ('rating', 'score').
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    anime_id: int = Field(..., gt=0, description="AniList anime identifier")
    rating: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        validation_alias=AliasChoices("rating", "score"),
        serialization_alias="rating",
        description="User numerical rating on [1.0, 10.0] scale"
    )
    origin_track: Literal["affinity", "discovery", "direct_search", "onboarding"] = Field(
        default="direct_search",
        description="Track where the user discovered this title"
    )
    user_review: Optional[str] = Field(
        default=None,
        max_length=300,
        validation_alias=AliasChoices("user_review", "qualitative_feedback", "review", "user_review_raw"),
        description="Optional qualitative free-text comment (max 300 characters)"
    )


# Backward compatibility alias
RateRequest = RatingRequest


class EditRatingRequest(BaseModel):
    """
    Request DTO for POST /api/rate/edit.
    Updates an existing rating and recalibrates user preferences across the entire history.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )

    anime_id: int = Field(..., gt=0, description="AniList anime identifier")
    rating: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        validation_alias=AliasChoices("rating", "score"),
        serialization_alias="rating",
        description="Updated user rating on [1.0, 10.0] scale"
    )
    user_review: Optional[str] = Field(
        default=None,
        max_length=300,
        validation_alias=AliasChoices("user_review", "qualitative_feedback", "review", "user_review_raw"),
        description="Optional updated qualitative comment"
    )



class TuningRequest(BaseModel):
    """
    Request DTO for POST /api/tune.
    Accepts user feedback string with aliases ('user_feedback', 'feedback', 'review').
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
    )

    anime_id: int = Field(..., gt=0, description="AniList anime identifier")
    user_feedback: str = Field(
        ...,
        max_length=300,
        validation_alias=AliasChoices("user_feedback", "feedback", "review"),
        description="Qualitative feedback string capped at 300 characters"
    )



# Backward compatibility alias
TuneRequest = TuningRequest


class ResetRequest(BaseModel):
    """Request DTO for POST /api/reset."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    confirm: bool = Field(
        default=True,
        description="Confirmation flag to reset profile and watch history"
    )


class RateResponse(BaseModel):
    """Response DTO for POST /api/rate."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = Field(default=True, description="Operation status flag")
    new_profile: UserProfileData = Field(..., description="Updated user profile state")
    updated_recommendations: Optional[RecommendationResponse] = Field(
        default=None,
        description="Fresh recommendations computed against updated profile"
    )
    history_entry: Optional[WatchHistoryEntry] = Field(
        default=None,
        description="Newly recorded interaction in watch history"
    )
    message: Optional[str] = Field(default="Rating recorded successfully.")


class TuneResponse(BaseModel):
    """Response DTO for POST /api/tune."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = Field(default=True, description="Operation status flag")
    tuning_response: SemanticTuningResponse = Field(..., description="Structured output extracted by Gemini")
    new_profile: UserProfileData = Field(..., description="Updated user profile after semantic boost")
    history_entry: Optional[WatchHistoryEntry] = Field(default=None, description="Enriched watch history entry")


class ResetResponse(BaseModel):
    """Response DTO for POST /api/reset."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = Field(default=True, description="Operation status flag")
    message: str = Field(default="User profile and watch history successfully reset to cold-start state.")
    profile: Optional[UserProfileData] = Field(default=None, description="Re-initialized cold start profile")


class HealthResponse(BaseModel):
    """Response DTO for GET /api/health."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    status: str = Field(default="ok", description="Service health status ('ok' or 'healthy')")
    catalog_size: int = Field(..., ge=0, description="Total anime records in local cache")
    cache_freshness: str = Field(..., description="Cache modification timestamp or status")
    gemini_status: str = Field(..., description="Google Gemini API readiness state")
    version: str = Field(default="1.0.0", description="Backend service version")


class SkipRequest(BaseModel):
    """Request DTO for POST /api/skip."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    anime_id: int = Field(..., gt=0, description="AniList anime identifier to skip or restore")
    unskip: bool = Field(default=False, description="If True, restores the skipped anime back to pool")


class SkipResponse(BaseModel):
    """Response DTO for POST /api/skip."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = Field(default=True)
    skipped_anime_ids: List[int] = Field(default_factory=list)
    skipped_animes: List[AnimePublicResponse] = Field(default_factory=list)
    new_profile: UserProfileData = Field(...)
    updated_recommendations: Optional[RecommendationResponse] = Field(default=None)
    message: str = Field(default="Anime skipped without distorting L2 preference weights.")


class DropRequest(BaseModel):
    """Request DTO for POST /api/drop."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True, str_strip_whitespace=True)

    anime_id: int = Field(..., gt=0, description="AniList anime identifier")
    drop_reason: str = Field(
        ...,
        min_length=1,
        max_length=300,
        validation_alias=AliasChoices("drop_reason", "reason", "user_feedback"),
        description="Written justification explaining why the user dropped this anime"
    )
    origin_track: Literal["affinity", "discovery", "direct_search", "onboarding"] = Field(
        default="direct_search",
        description="Discovery track where the anime was found"
    )


class DropResponse(BaseModel):
    """Response DTO for POST /api/drop."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = Field(default=True)
    tuning_response: SemanticTuningResponse = Field(...)
    new_profile: UserProfileData = Field(...)
    updated_recommendations: Optional[RecommendationResponse] = Field(default=None)
    history_entry: Optional[WatchHistoryEntry] = Field(default=None)
    message: str = Field(default="Drop registrado e preferências ajustadas pelo nosso algoritmo.")


class ToggleListRequest(BaseModel):
    """Request DTO for POST /api/favorite and POST /api/watchlist."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    anime_id: int = Field(..., gt=0, description="AniList anime identifier")


class ToggleListResponse(BaseModel):
    """Response DTO for POST /api/favorite and POST /api/watchlist."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = Field(default=True)
    active: bool = Field(..., description="True if added, False if removed")
    anime_ids: List[int] = Field(default_factory=list)
    new_profile: UserProfileData = Field(...)


class HistoryResponse(BaseModel):
    """Response DTO for GET /api/history."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    items: List[WatchHistoryEntry] = Field(default_factory=list)
    skipped_anime_ids: List[int] = Field(default_factory=list)
    dropped_anime_ids: List[int] = Field(default_factory=list)
    favorite_anime_ids: List[int] = Field(default_factory=list)
    watchlist_anime_ids: List[int] = Field(default_factory=list)
    skipped_animes: List[AnimePublicResponse] = Field(default_factory=list)
    dropped_animes: List[AnimePublicResponse] = Field(default_factory=list)
    favorite_animes: List[AnimePublicResponse] = Field(default_factory=list)
    watchlist_animes: List[AnimePublicResponse] = Field(default_factory=list)


class FilterPreferencesRequest(BaseModel):
    """Request DTO for POST /api/filters to block formats (MOVIE, OVA, etc.) and genres."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    excluded_formats: Optional[List[str]] = Field(
        default=None,
        description="Formats to exclude from recommendations (e.g. ['MOVIE', 'OVA', 'ONA', 'SPECIAL'])"
    )
    blocked_genres: Optional[List[str]] = Field(
        default=None,
        description="Genres to block from recommendations (e.g. ['Mecha', 'Horror', 'Romance'])"
    )


class FilterPreferencesResponse(BaseModel):
    """Response DTO for POST /api/filters."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = Field(default=True)
    excluded_formats: List[str] = Field(default_factory=list)
    blocked_genres: List[str] = Field(default_factory=list)
    new_profile: UserProfileData = Field(...)
    updated_recommendations: RecommendationResponse = Field(...)
    message: str = Field(default="Filtros de formato e bloqueio de gêneros atualizados.")


class GoogleAuthRequest(BaseModel):
    """Request DTO for POST /api/auth/google."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True, str_strip_whitespace=True)

    credential: Optional[str] = Field(default=None, description="Optional Google Identity Services JWT credential")
    email: Optional[str] = Field(default=None, max_length=150, description="Google account email")
    name: Optional[str] = Field(default=None, max_length=100, description="Google account display name")
    picture: Optional[str] = Field(default=None, max_length=500, description="Google avatar URL")
    sub: Optional[str] = Field(default=None, max_length=120, description="Google subject identifier")


class GoogleAuthResponse(BaseModel):
    """Response DTO for POST /api/auth/google, /api/auth/sync, and /api/auth/logout."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = Field(default=True)
    new_profile: UserProfileData = Field(...)
    watch_history: List[WatchHistoryEntry] = Field(default_factory=list)
    updated_recommendations: Optional[RecommendationResponse] = Field(default=None)
    message: str = Field(default="Progresso sincronizado com sua conta Google.")


class SynopsisResponse(BaseModel):
    """Response DTO for GET /api/anime/{anime_id}/synopsis."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    anime_id: int
    synopsis_pt: Optional[str] = None
    synopsis_en: Optional[str] = None
    has_spoiler_tags: bool = False


class AniListSyncRequest(BaseModel):
    """Request DTO for POST /api/auth/anilist/sync."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True, str_strip_whitespace=True)

    username: Optional[str] = Field(default=None, max_length=100, description="AniList public username")
    access_token: Optional[str] = Field(default=None, description="Optional OAuth2 access token")


class AniListSyncResponse(BaseModel):
    """Response DTO for POST /api/auth/anilist/sync."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    success: bool = True
    username: str
    imported_watched: int
    imported_watchlist: int
    imported_dropped: int
    imported_favorites: int
    new_profile: UserProfileData
    updated_recommendations: Optional[RecommendationResponse] = None
    message: str


def _resolve_anime_list(
    ids: List[int],
    catalog_by_id: Dict[int, AnimeEntity],
    spoiler_svc: SpoilerService,
) -> List[AnimePublicResponse]:
    """Resolves a list of anime IDs into spoiler-safe AnimePublicResponse DTOs."""
    resolved: List[AnimePublicResponse] = []
    for aid in ids:
        entity = catalog_by_id.get(aid)
        if entity is not None:
            resolved.append(spoiler_svc.to_public_response(entity))
        else:
            resolved.append(
                AnimePublicResponse(
                    id=aid,
                    title_romaji=f"Anime #{aid}",
                    title_english=f"Anime #{aid}",
                    cover_image="https://s4.anilist.co/file/anilistcdn/media/anime/cover/medium/default.jpg",
                    banner_image=None,
                    format="TV",
                    episodes=12,
                    status="FINISHED",
                    season_year=2024,
                    average_score=75,
                    popularity=10000,
                    genres=[],
                    tags=[],
                )
            )
    return resolved


def _sync_google_account_snapshot(storage: StorageManager, profile: UserProfileData) -> None:
    """
    Saves the user's current profile and watch history into data/accounts/<account_id>.json
    whenever a Google account is linked.
    """
    acct = getattr(profile, "google_account", None)
    if acct is None or not acct.email:
        return
    try:
        accounts_dir = storage.data_dir / "accounts"
        accounts_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", acct.email.strip().lower())
        acct_path = accounts_dir / f"{safe_id}.json"
        history = storage.load_watch_history()
        payload = {
            "google_account": acct.model_dump(mode="json"),
            "profile": profile.model_dump(mode="json"),
            "watch_history": [h.model_dump(mode="json") for h in history],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        acct_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ============================================================================
# API Routes
# ============================================================================

@router.get(
    "/catalog",
    response_model=AnimeCatalogResponse,
    status_code=status.HTTP_200_OK,
    summary="Paginated catalog listing with search query parameter q, genre filter, and spoiler scrubbing",
)
def get_catalog(
    page: int = Query(default=1, ge=1, description="Page number, 1-indexed"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum items per page"),
    q: Optional[str] = Query(default=None, max_length=100, description="Search query string"),
    genre: Optional[str] = Query(default=None, max_length=50, description="Genre filter term"),
    storage: StorageManager = Depends(get_storage_manager),
    spoiler_svc: SpoilerService = Depends(get_spoiler_service),
    anilist_svc: Optional[AniListService] = Depends(get_anilist_service_dep),
) -> AnimeCatalogResponse:
    """
    Returns a paginated list of catalog anime items filtered by search query and genre.
    Enforces strict R3 spoiler shielding on all returned AnimePublicResponse DTOs.
    """
    catalog = storage.load_catalog()
    filtered = catalog

    # 1. Search Query Filter (q)
    if q and q.strip():
        query_token = q.strip().lower()
        title_matches = []
        desc_matches = []
        for a in filtered:
            r = (a.title_romaji or "").lower()
            e = (a.title_english or "").lower()
            n = (a.title_native or "").lower()
            if query_token in r or query_token in e or query_token in n:
                title_matches.append(a)
            elif a.description and query_token in a.description.lower():
                desc_matches.append(a)

        # Dynamic On-Demand Live AniList Search
        # If fewer than 6 title matches found locally, query AniList live to ensure 100% catalog coverage
        if len(title_matches) < 6 and len(query_token) >= 2 and anilist_svc is not None:
            try:
                online_found = anilist_svc.search_anime_online(q.strip(), limit=8)
                existing_ids = {a.id for a in title_matches + desc_matches}
                for item in online_found:
                    if item.id not in existing_ids:
                        title_matches.append(item)
                        existing_ids.add(item.id)
            except Exception as search_err:
                logger.warning("Dynamic AniList live search warning: %s", search_err)

        filtered = title_matches + desc_matches

    # 2. Genre Filter
    if genre and genre.strip():
        genre_token = genre.strip().lower()
        filtered = [
            a for a in filtered
            if any(g.strip().lower() == genre_token for g in a.genres)
        ]

    total = len(filtered)
    total_pages = max(1, math.ceil(total / limit)) if total > 0 else 1

    # 3. Slicing
    start = (page - 1) * limit
    end = start + limit
    page_items = filtered[start:end]

    # 4. Strict Dual Spoiler Shielding (R3)
    public_items: List[AnimePublicResponse] = [
        spoiler_svc.to_public_response(anime)
        for anime in page_items
    ]

    return AnimeCatalogResponse(
        items=public_items,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
    )


@router.get(
    "/recommendations",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Dual-track recommendations with 80% Affinity and 20% Discovery MMR",
)
def get_recommendations(
    top_k: int = Query(default=10, ge=1, le=50, description="Total recommendation count"),
    lambda_param: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="MMR trade-off diversity override"),
    storage: StorageManager = Depends(get_storage_manager),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> RecommendationResponse:
    """
    Returns dual-track recommendations:
    - 80% Affinity exploitation track
    - 20% Discovery exploration MMR track
    Guarantees: S_aff ∩ S_disc = ∅, watch history exclusion, zero spoiler leakage.
    """
    catalog = storage.load_catalog()
    profile = storage.load_user_profile()
    history = storage.load_watch_history()

    return rec_engine.generate_recommendations(
        profile=profile,
        catalog=catalog,
        history=history,
        top_k=top_k,
        lambda_param=lambda_param,
    )


@router.post(
    "/rate",
    response_model=RateResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit anime rating and update continuous user profile vector",
)
def rate_anime(
    request: RatingRequest,
    background_tasks: BackgroundTasks,
    storage: StorageManager = Depends(get_storage_manager),
    learning_engine: LearningEngine = Depends(get_learning_engine),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
    gemini_svc: GeminiService = Depends(get_gemini_service),
    anilist_svc: Optional[AniListService] = Depends(get_anilist_service),
) -> RateResponse:
    """
    Submits numerical rating [1.0, 10.0] and optional review text.
    Updates user profile via online gradient delta + temporal decay + L2 projection.
    Records interaction in watch history and returns updated profile and fresh recommendations.
    Schedules background catalog expansion via AniListService.
    """
    catalog = storage.load_catalog()
    anime = next((a for a in catalog if a.id == request.anime_id), None)
    if anime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anime ID {request.anime_id} not found in catalog.",
        )

    # Guard against duplicate evaluations: reject new rating if already evaluated
    current_history = storage.load_watch_history()
    existing_entry = next(
        (h for h in current_history if h.anime_id == request.anime_id and h.interaction_status == "watched"),
        None,
    )
    if existing_entry is not None:
        anime_name = anime.title_romaji or anime.title_english or f"Anime #{anime.id}"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"A obra '{anime_name}' já foi avaliada com nota {existing_entry.rating}/10. "
                "Novas avaliações diretas estão bloqueadas para evitar duplicatas e proteger o algoritmo. "
                "Para alterar sua avaliação, utilize a opção 'Editar Nota' em Minha Lista."
            ),
        )

    # Sanitize user review text if supplied (Layer 1 Defense)
    sanitized_review: Optional[str] = None
    if request.user_review:
        sanitized_review = gemini_svc.sanitize_user_feedback(request.user_review, max_length=300)

    # Record rating in continuous learning engine (atomic read-modify-write)
    updated_profile, history_entry = learning_engine.record_rating(
        anime_id=request.anime_id,
        rating=request.rating,
        origin_track=request.origin_track,
        user_review=sanitized_review,
        qualitative_tuning=None,
    )

    # Schedule adjacent catalog crawl asynchronously in background
    if anilist_svc is not None and hasattr(anilist_svc, "expand_catalog_for_anime"):
        background_tasks.add_task(anilist_svc.expand_catalog_for_anime, request.anime_id)

    # Generate fresh recommendations immediately
    updated_history = storage.load_watch_history()
    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=catalog,
        history=updated_history,
        top_k=10,
    )
    _sync_google_account_snapshot(storage, updated_profile)

    return RateResponse(
        success=True,
        new_profile=updated_profile,
        updated_recommendations=updated_recs,
        history_entry=history_entry,
        message="Rating recorded successfully.",
    )


@router.post(
    "/rate/edit",
    response_model=RateResponse,
    status_code=status.HTTP_200_OK,
    summary="Update/edit an existing anime rating and recalibrate user profile feature weights",
)
def edit_anime_rating(
    request: EditRatingRequest,
    storage: StorageManager = Depends(get_storage_manager),
    learning_engine: LearningEngine = Depends(get_learning_engine),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
    gemini_svc: GeminiService = Depends(get_gemini_service),
) -> RateResponse:
    """
    Edits a previously recorded rating in watch history.
    Recalibrates user profile feature weights across all ratings and returns updated profile + fresh recommendations.
    """
    catalog = storage.load_catalog()
    catalog_by_id = {a.id: a for a in catalog}
    anime = catalog_by_id.get(request.anime_id)
    if anime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anime ID {request.anime_id} not found in catalog.",
        )

    sanitized_review: Optional[str] = None
    if request.user_review:
        sanitized_review = gemini_svc.sanitize_user_feedback(request.user_review, max_length=300)

    # Update or add entry in watch history
    current_history = storage.load_watch_history()
    target_entry = None
    updated_history = []
    now_utc = datetime.now(timezone.utc)

    for entry in current_history:
        if entry.anime_id == request.anime_id:
            updated_entry = entry.model_copy(
                update={
                    "rating": request.rating,
                    "interaction_status": "watched",
                    "timestamp": now_utc,
                    "user_review_raw": sanitized_review if sanitized_review is not None else entry.user_review_raw,
                }
            )
            updated_history.append(updated_entry)
            target_entry = updated_entry
        else:
            updated_history.append(entry)

    if target_entry is None:
        anime_title = anime.title_romaji or anime.title_english or f"Anime #{anime.id}"
        target_entry = WatchHistoryEntry(
            id=f"wh-{uuid.uuid4().hex[:8]}",
            anime_id=anime.id,
            anime_title=anime_title,
            rating=request.rating,
            origin_track="direct_search",
            interaction_status="watched",
            user_review_raw=sanitized_review,
            timestamp=now_utc,
        )
        updated_history.append(target_entry)

    storage.save_watch_history(updated_history)

    # Recalibrate user profile weights from scratch over watch history to guarantee zero drift
    def recalibrate_modifier(prof: UserProfileData) -> UserProfileData:
        temp_prof = prof.model_copy(
            update={
                "feature_weights": {},
                "total_ratings": 0,
                "updated_at": now_utc,
            }
        )
        for h in updated_history:
            if h.interaction_status == "watched" and h.anime_id in catalog_by_id:
                ent = catalog_by_id[h.anime_id]
                temp_prof = learning_engine.update_profile_rating(
                    profile=temp_prof,
                    rated_anime=ent,
                    rating=h.rating,
                    qualitative_tuning=h.qualitative_tuning,
                )
        return temp_prof

    updated_profile = storage.update_user_profile(recalibrate_modifier)
    _sync_google_account_snapshot(storage, updated_profile)

    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=catalog,
        history=updated_history,
        top_k=10,
    )

    return RateResponse(
        success=True,
        new_profile=updated_profile,
        updated_recommendations=updated_recs,
        history_entry=target_entry,
        message=f"Avaliação de '{target_entry.anime_title}' atualizada para {request.rating}/10! Pesos do algoritmo recalibrados.",
    )


class CancelRatingRequest(BaseModel):
    """Request DTO for POST /api/rate/cancel."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    anime_id: int = Field(..., gt=0, description="AniList anime identifier whose rating should be cancelled")


@router.post(
    "/rate/cancel",
    response_model=RateResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel/delete an anime rating and recalibrate user profile weights",
)
def cancel_anime_rating(
    request: CancelRatingRequest,
    storage: StorageManager = Depends(get_storage_manager),
    learning_engine: LearningEngine = Depends(get_learning_engine),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> RateResponse:
    """
    Removes a rating from watch history and recalibrates user profile feature weights from scratch
    so an accidental evaluation leaves zero distortion in the algorithm.
    """
    catalog = storage.load_catalog()
    catalog_by_id = {a.id: a for a in catalog}
    current_history = storage.load_watch_history()
    updated_history = [h for h in current_history if h.anime_id != request.anime_id]
    storage.save_watch_history(updated_history)
    now_utc = datetime.now(timezone.utc)

    def recalibrate_after_cancel(prof: UserProfileData) -> UserProfileData:
        temp_prof = prof.model_copy(
            update={
                "feature_weights": {},
                "total_ratings": 0,
                "updated_at": now_utc,
            }
        )
        for h in updated_history:
            if h.interaction_status == "watched" and h.anime_id in catalog_by_id:
                ent = catalog_by_id[h.anime_id]
                temp_prof = learning_engine.update_profile_rating(
                    profile=temp_prof,
                    rated_anime=ent,
                    rating=h.rating,
                    qualitative_tuning=h.qualitative_tuning,
                )
        return temp_prof

    updated_profile = storage.update_user_profile(recalibrate_after_cancel)
    _sync_google_account_snapshot(storage, updated_profile)

    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=catalog,
        history=updated_history,
        top_k=10,
    )

    return RateResponse(
        success=True,
        new_profile=updated_profile,
        updated_recommendations=updated_recs,
        history_entry=None,
        message="Avaliação cancelada e pesos recalibrados com sucesso!",
    )




@router.post(
    "/tune",
    response_model=TuneResponse,
    status_code=status.HTTP_200_OK,
    summary="Two-turn qualitative tuning endpoint with natural language feedback",
)
def tune_preferences(
    request: TuningRequest,
    storage: StorageManager = Depends(get_storage_manager),
    learning_engine: LearningEngine = Depends(get_learning_engine),
    gemini_svc: GeminiService = Depends(get_gemini_service),
) -> TuneResponse:
    """
    Invokes Gemini qualitative semantic tuning on user feedback.
    Applies semantic boost alpha=0.25 to user profile while maintaining unit L2 norm invariant.
    Enriches watch history entry and returns updated profile.
    """
    clean_feedback = request.user_feedback.strip()
    if not clean_feedback:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User feedback cannot be empty or whitespace only.",
        )

    catalog = storage.load_catalog()
    anime = next((a for a in catalog if a.id == request.anime_id), None)
    if anime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anime ID {request.anime_id} not found in catalog.",
        )

    current_profile = storage.load_user_profile()

    # Gemini Inference under 4-Layer Defense
    tuning_response = gemini_svc.tune_profile(
        user_feedback=clean_feedback,
        rated_anime=anime,
        current_profile=current_profile,
    )

    # Atomic semantic boost applied to user profile (alpha=0.25, unit L2 norm preserved)
    def boost_modifier(prof: UserProfileData) -> UserProfileData:
        return learning_engine.apply_semantic_boost(profile=prof, tuning=tuning_response)

    updated_profile = storage.update_user_profile(boost_modifier)

    # Enrich latest watch history interaction
    enriched_entry = storage.enrich_latest_watch_history(
        anime_id=request.anime_id,
        user_review_raw=clean_feedback,
        qualitative_tuning=tuning_response,
    )

    return TuneResponse(
        success=True,
        tuning_response=tuning_response,
        new_profile=updated_profile,
        history_entry=enriched_entry,
    )


@router.get(
    "/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve watch history and user lists (skipped, dropped, favorites, watchlist)",
)
def get_watch_history_and_lists(
    storage: StorageManager = Depends(get_storage_manager),
    spoiler_svc: SpoilerService = Depends(get_spoiler_service),
) -> HistoryResponse:
    """
    Returns full watch history alongside skipped, dropped, favorite, and watchlist anime IDs
    AND their resolved spoiler-safe AnimePublicResponse objects from the full catalog.
    """
    profile = storage.load_user_profile()
    history = storage.load_watch_history()
    catalog = storage.load_catalog()
    catalog_by_id = {a.id: a for a in catalog}

    skipped_ids = list(getattr(profile, "skipped_anime_ids", []))
    dropped_ids = list(getattr(profile, "dropped_anime_ids", []))
    favorite_ids = list(getattr(profile, "favorite_anime_ids", []))
    watchlist_ids = list(getattr(profile, "watchlist_anime_ids", []))

    return HistoryResponse(
        items=history,
        skipped_anime_ids=skipped_ids,
        dropped_anime_ids=dropped_ids,
        favorite_anime_ids=favorite_ids,
        watchlist_anime_ids=watchlist_ids,
        skipped_animes=_resolve_anime_list(skipped_ids, catalog_by_id, spoiler_svc),
        dropped_animes=_resolve_anime_list(dropped_ids, catalog_by_id, spoiler_svc),
        favorite_animes=_resolve_anime_list(favorite_ids, catalog_by_id, spoiler_svc),
        watchlist_animes=_resolve_anime_list(watchlist_ids, catalog_by_id, spoiler_svc),
    )


@router.post(
    "/skip",
    response_model=SkipResponse,
    status_code=status.HTTP_200_OK,
    summary="Skip or unskip an anime without distorting user L2 preference weights",
)
def skip_anime(
    request: SkipRequest,
    storage: StorageManager = Depends(get_storage_manager),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
    spoiler_svc: SpoilerService = Depends(get_spoiler_service),
) -> SkipResponse:
    """
    Marks an anime as 'skipped / not interested' (or restores it when unskip=True)
    with ZERO change to feature_weights (Delta u = 0), excluding it from recommendations.
    """
    catalog = storage.load_catalog()
    catalog_by_id = {a.id: a for a in catalog}
    anime = catalog_by_id.get(request.anime_id)
    if anime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anime ID {request.anime_id} not found in catalog.",
        )

    def skip_modifier(prof: UserProfileData) -> UserProfileData:
        skipped = list(getattr(prof, "skipped_anime_ids", []))
        if request.unskip:
            skipped = [aid for aid in skipped if aid != request.anime_id]
        else:
            if request.anime_id not in skipped:
                skipped.append(request.anime_id)
        return prof.model_copy(
            update={
                "skipped_anime_ids": skipped,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    updated_profile = storage.update_user_profile(skip_modifier)
    _sync_google_account_snapshot(storage, updated_profile)
    history = storage.load_watch_history()
    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=catalog,
        history=history,
        top_k=10,
    )

    return SkipResponse(
        success=True,
        skipped_anime_ids=updated_profile.skipped_anime_ids,
        skipped_animes=_resolve_anime_list(
            updated_profile.skipped_anime_ids, catalog_by_id, spoiler_svc
        ),
        new_profile=updated_profile,
        updated_recommendations=updated_recs,
        message=(
            "Obra restaurada para recomendações."
            if request.unskip
            else "Obra pulada sem alterar os pesos do seu perfil L2."
        ),
    )


@router.post(
    "/drop",
    response_model=DropResponse,
    status_code=status.HTTP_200_OK,
    summary="Drop an anime with written justification analyzed by Gemini for negative semantic tuning",
)
def drop_anime(
    request: DropRequest,
    storage: StorageManager = Depends(get_storage_manager),
    learning_engine: LearningEngine = Depends(get_learning_engine),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
    gemini_svc: GeminiService = Depends(get_gemini_service),
) -> DropResponse:
    """
    Records that the user dropped an anime and sends the written justification to Gemini
    to identify and penalize the specific elements/tags that caused the drop.
    """
    clean_reason = request.drop_reason.strip()
    if not clean_reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Drop reason cannot be empty.",
        )

    catalog = storage.load_catalog()
    anime = next((a for a in catalog if a.id == request.anime_id), None)
    if anime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anime ID {request.anime_id} not found in catalog.",
        )

    updated_profile, history_entry, tuning_response = learning_engine.record_drop(
        anime_id=request.anime_id,
        drop_reason=clean_reason,
        origin_track=request.origin_track,
        gemini_service=gemini_svc,
    )
    _sync_google_account_snapshot(storage, updated_profile)

    updated_history = storage.load_watch_history()
    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=catalog,
        history=updated_history,
        top_k=10,
    )

    return DropResponse(
        success=True,
        tuning_response=tuning_response,
        new_profile=updated_profile,
        updated_recommendations=updated_recs,
        history_entry=history_entry,
        message="Drop registrado e preferências ajustadas pelo nosso algoritmo.",
    )


@router.post(
    "/favorite",
    response_model=ToggleListResponse,
    status_code=status.HTTP_200_OK,
    summary="Toggle favorite status for an anime",
)
def toggle_favorite(
    request: ToggleListRequest,
    storage: StorageManager = Depends(get_storage_manager),
) -> ToggleListResponse:
    """Toggles an anime ID in the user's favorite_anime_ids list."""
    active_state = False

    def fav_modifier(prof: UserProfileData) -> UserProfileData:
        nonlocal active_state
        favs = list(getattr(prof, "favorite_anime_ids", []))
        if request.anime_id in favs:
            favs = [aid for aid in favs if aid != request.anime_id]
            active_state = False
        else:
            favs.append(request.anime_id)
            active_state = True
        return prof.model_copy(
            update={
                "favorite_anime_ids": favs,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    updated_profile = storage.update_user_profile(fav_modifier)
    _sync_google_account_snapshot(storage, updated_profile)
    return ToggleListResponse(
        success=True,
        active=active_state,
        anime_ids=updated_profile.favorite_anime_ids,
        new_profile=updated_profile,
    )


@router.post(
    "/watchlist",
    response_model=ToggleListResponse,
    status_code=status.HTTP_200_OK,
    summary="Toggle watchlist ('Minha Lista / Quero Assistir') status for an anime",
)
def toggle_watchlist(
    request: ToggleListRequest,
    storage: StorageManager = Depends(get_storage_manager),
) -> ToggleListResponse:
    """Toggles an anime ID in the user's watchlist_anime_ids list."""
    active_state = False

    def wl_modifier(prof: UserProfileData) -> UserProfileData:
        nonlocal active_state
        wl = list(getattr(prof, "watchlist_anime_ids", []))
        if request.anime_id in wl:
            wl = [aid for aid in wl if aid != request.anime_id]
            active_state = False
        else:
            wl.append(request.anime_id)
            active_state = True
        return prof.model_copy(
            update={
                "watchlist_anime_ids": wl,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    updated_profile = storage.update_user_profile(wl_modifier)
    _sync_google_account_snapshot(storage, updated_profile)
    return ToggleListResponse(
        success=True,
        active=active_state,
        anime_ids=updated_profile.watchlist_anime_ids,
        new_profile=updated_profile,
    )


@router.post(
    "/filters",
    response_model=FilterPreferencesResponse,
    status_code=status.HTTP_200_OK,
    summary="Update blocked formats (e.g. MOVIE, OVA) and blocked genres (e.g. Mecha, Horror)",
)
def update_recommendation_filters(
    request: FilterPreferencesRequest,
    storage: StorageManager = Depends(get_storage_manager),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> FilterPreferencesResponse:
    """
    Updates the user's persistent format and genre blocking filters and recalculates recommendations.
    """
    def filter_modifier(prof: UserProfileData) -> UserProfileData:
        updates: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if request.excluded_formats is not None:
            clean_fmts = []
            seen_fmt = set()
            for f in request.excluded_formats:
                fu = str(f).strip().upper()
                if fu and fu not in seen_fmt:
                    seen_fmt.add(fu)
                    clean_fmts.append(fu)
            updates["excluded_formats"] = clean_fmts

        if request.blocked_genres is not None:
            clean_genres = []
            seen_g = set()
            for g in request.blocked_genres:
                gc = str(g).strip()
                if gc and gc.lower() not in seen_g:
                    seen_g.add(gc.lower())
                    clean_genres.append(gc)
            updates["blocked_genres"] = clean_genres

        return prof.model_copy(update=updates)

    updated_profile = storage.update_user_profile(filter_modifier)
    _sync_google_account_snapshot(storage, updated_profile)

    catalog = storage.load_catalog()
    history = storage.load_watch_history()
    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=catalog,
        history=history,
        top_k=10,
    )

    return FilterPreferencesResponse(
        success=True,
        excluded_formats=updated_profile.excluded_formats,
        blocked_genres=updated_profile.blocked_genres,
        new_profile=updated_profile,
        updated_recommendations=updated_recs,
        message="Filtros de recomendação aplicados com sucesso.",
    )


@router.post(
    "/auth/google",
    response_model=GoogleAuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Sign in with Google to save and synchronize user recommendation progress",
)
def google_sign_in(
    request: GoogleAuthRequest,
    storage: StorageManager = Depends(get_storage_manager),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> GoogleAuthResponse:
    """
    Authenticates or links a Google account (via Google Identity Services JWT credential
    or direct Google profile payload) and synchronizes progress in data/accounts/.
    """
    email = (request.email or "").strip()
    name = (request.name or "").strip()
    picture = (request.picture or "").strip() or None
    sub = (request.sub or "").strip()

    # Decode Google Identity Services JWT if credential token was provided
    if request.credential and "." in request.credential:
        try:
            parts = request.credential.split(".")
            if len(parts) >= 2:
                padded = parts[1] + "=" * (-len(parts[1]) % 4)
                decoded_bytes = base64.urlsafe_b64decode(padded.encode("utf-8"))
                claims = json.loads(decoded_bytes.decode("utf-8"))
                email = email or str(claims.get("email", "")).strip()
                name = name or str(claims.get("name", "")).strip()
                picture = picture or str(claims.get("picture", "")).strip() or None
                sub = sub or str(claims.get("sub", "")).strip()
        except Exception:
            pass

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="E-mail da conta Google é obrigatório para salvar o progresso.",
        )

    if not name:
        name = email.split("@")[0].replace(".", " ").title()

    if not sub:
        sub = "google-" + hashlib.sha256(email.lower().encode("utf-8")).hexdigest()[:16]

    now_utc = datetime.now(timezone.utc)
    google_acct = GoogleUserAccount(
        sub=sub,
        email=email,
        name=name,
        picture=picture,
        connected_at=now_utc,
        last_synced_at=now_utc,
    )

    accounts_dir = storage.data_dir / "accounts"
    accounts_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", email.lower())
    acct_path = accounts_dir / f"{safe_id}.json"

    current_profile = storage.load_user_profile()
    current_history = storage.load_watch_history()

    has_local_activity = (
        current_profile.total_ratings > 0
        or len(current_profile.skipped_anime_ids) > 0
        or len(current_profile.dropped_anime_ids) > 0
        or len(current_profile.watchlist_anime_ids) > 0
        or len(current_profile.favorite_anime_ids) > 0
    )

    # Restore saved Google account progress if local profile is empty and backup exists
    if not has_local_activity and acct_path.exists():
        try:
            saved_data = json.loads(acct_path.read_text(encoding="utf-8"))
            if "profile" in saved_data:
                restored_prof = UserProfileData.model_validate(saved_data["profile"])
                restored_prof = restored_prof.model_copy(
                    update={"google_account": google_acct, "updated_at": now_utc}
                )
                storage.save_user_profile(restored_prof)
                current_profile = restored_prof
            if "watch_history" in saved_data and isinstance(saved_data["watch_history"], list):
                restored_hist = [
                    WatchHistoryEntry.model_validate(h) for h in saved_data["watch_history"]
                ]
                storage.save_watch_history(restored_hist)
                current_history = restored_hist
        except Exception:
            pass

    # Link Google account to active profile and persist snapshot
    updated_profile = storage.update_user_profile(
        lambda prof: prof.model_copy(
            update={
                "user_id": sub,
                "google_account": google_acct,
                "updated_at": now_utc,
            }
        )
    )
    _sync_google_account_snapshot(storage, updated_profile)

    catalog = storage.load_catalog()
    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=catalog,
        history=current_history,
        top_k=10,
    )

    return GoogleAuthResponse(
        success=True,
        new_profile=updated_profile,
        watch_history=current_history,
        updated_recommendations=updated_recs,
        message=f"Conectado como {name} ({email}). Seu progresso está salvo e sincronizado!",
    )


@router.post(
    "/auth/sync",
    response_model=GoogleAuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Explicitly save and synchronize current user progress with linked Google account",
)
def sync_google_progress(
    storage: StorageManager = Depends(get_storage_manager),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> GoogleAuthResponse:
    """Explicitly syncs current profile and watch history to the linked Google account backup."""
    now_utc = datetime.now(timezone.utc)

    def sync_modifier(prof: UserProfileData) -> UserProfileData:
        if prof.google_account is None:
            return prof
        updated_acct = prof.google_account.model_copy(update={"last_synced_at": now_utc})
        return prof.model_copy(update={"google_account": updated_acct, "updated_at": now_utc})

    updated_profile = storage.update_user_profile(sync_modifier)
    _sync_google_account_snapshot(storage, updated_profile)
    history = storage.load_watch_history()
    catalog = storage.load_catalog()
    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=catalog,
        history=history,
        top_k=10,
    )

    return GoogleAuthResponse(
        success=True,
        new_profile=updated_profile,
        watch_history=history,
        updated_recommendations=updated_recs,
        message="Progresso salvo e sincronizado na sua conta Google!",
    )


@router.post(
    "/auth/logout",
    response_model=GoogleAuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Disconnect active Google session while keeping saved progress backup",
)
def google_logout(
    storage: StorageManager = Depends(get_storage_manager),
) -> GoogleAuthResponse:
    """Saves final snapshot and unlinks google_account from active session."""
    current_profile = storage.load_user_profile()
    _sync_google_account_snapshot(storage, current_profile)

    updated_profile = storage.update_user_profile(
        lambda prof: prof.model_copy(
            update={"google_account": None, "updated_at": datetime.now(timezone.utc)}
        )
    )
    history = storage.load_watch_history()
    return GoogleAuthResponse(
        success=True,
        new_profile=updated_profile,
        watch_history=history,
        message="Sessão Google desconectada. Seu backup permanece salvo.",
    )


@router.get(
    "/profile",
    response_model=UserProfileData,
    status_code=status.HTTP_200_OK,
    summary="Retrieve current user profile data and hyperparameters",
)
def get_user_profile(
    storage: StorageManager = Depends(get_storage_manager),
) -> UserProfileData:
    """
    Returns current UserProfileData (user_id, updated_at, total_ratings, feature_weights, hyperparameters).
    """
    return storage.load_user_profile()


@router.get(
    "/anime/{anime_id}/synopsis",
    response_model=SynopsisResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Portuguese synopsis and spoiler status for an anime",
)
def get_anime_synopsis(
    anime_id: int,
    storage: StorageManager = Depends(get_storage_manager),
    spoiler_svc: SpoilerService = Depends(get_spoiler_service),
) -> SynopsisResponse:
    catalog = storage.load_catalog()
    anime = next((a for a in catalog if a.id == anime_id), None)
    if anime is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anime não encontrado no catálogo.")

    from app.services.translation_service import get_translation_service
    trans_svc = get_translation_service()
    synopsis_pt = trans_svc.get_synopsis_pt(anime.id, anime.description, allow_network=True)
    has_spoilers = any(spoiler_svc.is_spoiler_tag(t) for t in anime.tags) or spoiler_svc.contains_spoiler_keywords(anime.description or "")

    return SynopsisResponse(
        anime_id=anime_id,
        synopsis_pt=synopsis_pt or anime.description,
        synopsis_en=anime.description,
        has_spoiler_tags=has_spoilers,
    )


@router.get(
    "/season/releases",
    status_code=status.HTTP_200_OK,
    summary="Retrieve current seasonal anime releases",
)
def get_season_releases() -> Dict[str, Any]:
    """
    Returns current season releases with studios, airing episode metadata,
    scores, and Brazilian Portuguese synopses.
    """
    seasonal_file = Path(__file__).resolve().parent.parent.parent.parent / "data" / "seasonal_releases.json"
    if not seasonal_file.exists():
        alt_file = Path("data/seasonal_releases.json")
        if alt_file.exists():
            seasonal_file = alt_file

    if seasonal_file.exists():
        try:
            return json.loads(seasonal_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read seasonal_releases.json: %s", exc)

    return {
        "season": "ATUAL",
        "year": datetime.now().year,
        "items": [],
        "message": "Nenhum lançamento encontrado no momento."
    }


@router.post(
    "/auth/anilist/sync",
    response_model=AniListSyncResponse,
    status_code=status.HTTP_200_OK,
    summary="Synchronize completed, planning, dropped, and favorite animes from AniList",
)
async def sync_anilist_lists(
    request: AniListSyncRequest,
    storage: StorageManager = Depends(get_storage_manager),
    learning_engine: LearningEngine = Depends(get_learning_engine),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine),
    anilist_svc: AniListService = Depends(get_anilist_service_dep),
) -> AniListSyncResponse:
    raw_user = (request.username or "").strip()
    token = (request.access_token or "").strip() or None

    if not raw_user and not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Informe seu nome de usuário da AniList ou um token de acesso OAuth.",
        )

    try:
        data = await anilist_svc.fetch_user_media_lists(username=raw_user if raw_user else None, access_token=token)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Não foi possível sincronizar com a AniList: {str(exc)}",
        )

    collection = data.get("MediaListCollection", {})
    lists = collection.get("lists", [])
    if not lists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhuma lista de anime pública encontrada para '{raw_user}'. Verifique se o perfil existe e não está privado na AniList.",
        )

    # Ingest entities into local catalog cache
    current_catalog = storage.load_catalog()
    catalog_by_id = {a.id: a for a in current_catalog}
    new_entities = []

    imported_watched = 0
    imported_watchlist = 0
    imported_dropped = 0

    watched_entries_to_learn = []
    watchlist_ids_to_add = set()
    dropped_ids_to_add = set()
    favorite_ids_to_add = set()

    for l in lists:
        l_status = str(l.get("status") or "").upper()
        l_name = str(l.get("name") or "").lower()
        entries = l.get("entries", [])

        for entry in entries:
            media = entry.get("media")
            if not media or not media.get("id"):
                continue

            mid = int(media["id"])
            if mid not in catalog_by_id:
                try:
                    parsed_entity = anilist_svc.parse_media_item(media)
                    new_entities.append(parsed_entity)
                    catalog_by_id[mid] = parsed_entity
                except Exception:
                    pass

            score_val = float(entry.get("score") or 0.0)
            if score_val > 10.0:
                score_val = score_val / 10.0
            if score_val <= 0.0:
                score_val = 7.5

            if l_status == "COMPLETED" or "completed" in l_name or "assistido" in l_name:
                imported_watched += 1
                watched_entries_to_learn.append((mid, score_val))
                if score_val >= 9.0:
                    favorite_ids_to_add.add(mid)
            elif l_status == "PLANNING" or "planning" in l_name or "assistir" in l_name:
                imported_watchlist += 1
                watchlist_ids_to_add.add(mid)
            elif l_status == "DROPPED" or "dropped" in l_name:
                imported_dropped += 1
                dropped_ids_to_add.add(mid)

    if new_entities:
        storage.save_catalog(list(catalog_by_id.values()))

    # Apply batch updates to user profile (pure math, avoiding recursive FileLock deadlock)
    current_prof = storage.load_user_profile()
    updated_wl = list(set(current_prof.watchlist_anime_ids + list(watchlist_ids_to_add)))
    updated_dropped = list(set(current_prof.dropped_anime_ids + list(dropped_ids_to_add)))
    updated_favs = list(set(current_prof.favorite_anime_ids + list(favorite_ids_to_add)))

    def prof_modifier(prof: UserProfileData) -> UserProfileData:
        temp_prof = prof.model_copy(
            update={
                "watchlist_anime_ids": updated_wl,
                "dropped_anime_ids": updated_dropped,
                "favorite_anime_ids": updated_favs,
            }
        )
        for mid, sc in watched_entries_to_learn[:35]:
            anime_ent = catalog_by_id.get(mid)
            if anime_ent:
                try:
                    temp_prof = learning_engine.update_profile_rating(
                        profile=temp_prof,
                        rated_anime=anime_ent,
                        rating=sc,
                    )
                except Exception:
                    pass
        return temp_prof

    updated_profile = storage.update_user_profile(prof_modifier)

    # Ingest watched entries into watch history without duplicate keys
    current_history = storage.load_watch_history()
    existing_history_ids = {h.anime_id for h in current_history}
    new_history_entries = []
    now_utc = datetime.now(timezone.utc)
    for mid, sc in watched_entries_to_learn:
        if mid not in existing_history_ids and mid in catalog_by_id:
            anime_ent = catalog_by_id[mid]
            new_history_entries.append(
                WatchHistoryEntry(
                    id=str(uuid.uuid4()),
                    anime_id=mid,
                    anime_title=anime_ent.title_romaji or anime_ent.title_english or f"Anime #{mid}",
                    rating=sc,
                    timestamp=now_utc,
                    interaction_status="watched",
                    origin_track="onboarding",
                )
            )
            existing_history_ids.add(mid)

    if new_history_entries:
        storage.save_watch_history(current_history + new_history_entries)

    _sync_google_account_snapshot(storage, updated_profile)

    fresh_catalog = storage.load_catalog()
    fresh_history = storage.load_watch_history()
    updated_recs = rec_engine.generate_recommendations(
        profile=updated_profile,
        catalog=fresh_catalog,
        history=fresh_history,
        top_k=10,
    )

    return AniListSyncResponse(
        success=True,
        username=raw_user or "AniList User",
        imported_watched=imported_watched,
        imported_watchlist=imported_watchlist,
        imported_dropped=imported_dropped,
        imported_favorites=len(favorite_ids_to_add),
        new_profile=updated_profile,
        updated_recommendations=updated_recs,
        message=f"Sincronização concluída com sucesso da AniList! {imported_watched} assistidos, {imported_watchlist} na lista, {imported_dropped} dropados.",
    )


@router.post(
    "/reset",
    response_model=ResetResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset user profile to cold start and clear watch history",
)
def reset_user_data(
    request: Optional[ResetRequest] = None,
    storage: StorageManager = Depends(get_storage_manager),
) -> ResetResponse:
    """
    Resets user profile to cold start ({total_ratings: 0, feature_weights: {}})
    and clears watch history ([]), preserving linked Google account.
    """
    profile, _ = storage.reset_all_user_data()
    _sync_google_account_snapshot(storage, profile)
    return ResetResponse(
        success=True,
        message="Preferências e histórico resetados para o estado inicial.",
        profile=profile,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check reporting catalog size, cache freshness, and Gemini status",
)
def get_health(
    storage: StorageManager = Depends(get_storage_manager),
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    """
    Returns system status, catalog size, cache freshness, and Gemini readiness.
    """
    catalog = storage.load_catalog()
    catalog_size = len(catalog)

    freshness = "unknown"
    if storage.catalog_path.exists():
        try:
            mtime = storage.catalog_path.stat().st_mtime
            freshness = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except Exception:
            freshness = "available"

    gemini_status = "configured" if settings.is_gemini_configured else "offline_fallback"

    return HealthResponse(
        status="ok",
        catalog_size=catalog_size,
        cache_freshness=freshness,
        gemini_status=gemini_status,
        version="1.0.0",
    )


@router.get(
    "/media/cover/{anime_id}",
    summary="Serve cached anime cover image from local disk with auto-healing fallback",
)
def get_anime_cover_image(anime_id: int):
    cover_path = Path("data/media/covers") / f"{anime_id}.jpg"
    if cover_path.exists() and cover_path.stat().st_size > 500:
        return FileResponse(
            cover_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
    storage = StorageManager()
    catalog = storage.load_catalog()
    anime = next((a for a in catalog if a.id == anime_id), None)
    if anime and anime.cover_image:
        return RedirectResponse(anime.cover_image)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cover not found")


@router.get(
    "/media/banner/{anime_id}",
    summary="Serve cached anime banner image from local disk with auto-healing fallback",
)
def get_anime_banner_image(anime_id: int):
    banner_path = Path("data/media/banners") / f"{anime_id}.jpg"
    if banner_path.exists() and banner_path.stat().st_size > 500:
        return FileResponse(
            banner_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
    storage = StorageManager()
    catalog = storage.load_catalog()
    anime = next((a for a in catalog if a.id == anime_id), None)
    if anime and anime.banner_image:
        return RedirectResponse(anime.banner_image)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Banner not found")

