/**
 * frontend/src/services/api.ts
 * Decoupled Asynchronous Service Layer for AnimeRecSys AI.
 *
 * Rules:
 * 1. UI components NEVER import fixtures directly; all data access goes through this service.
 * 2. Automatically probes FastAPI backend health (http://127.0.0.1:8000/api/health).
 * 3. When backend is online, routes all calls to live FastAPI endpoints.
 * 4. When backend is offline, seamlessly falls back to local fixture state with reactive
 *    mathematical updates (L2 normalization, gamma decay, 80/20 MMR split, and spoiler stripping).
 */

import {
  MOCK_GEMINI_RESPONSE,
  MOCK_USER_PROFILE,
  MOCK_WATCH_HISTORY,
  SAMPLE_ANIMES_FULL,
  TAGS_VOCABULARY,
  type RawAnimeFixture,
} from '../mocks/fixtures';
import type {
  AniListSyncRequest,
  AniListSyncResponse,
  AnimeCatalogResponse,
  AnimePublic,
  DropResponse,
  DropSubmission,
  FilterPreferencesPayload,
  FilterPreferencesResponse,
  GoogleAuthPayload,
  GoogleAuthResponse,
  HealthResponse,
  HistoryResponse,
  RateResponse,
  RatingSubmission,
  RecommendationPayload,
  SemanticTagDelta,
  SemanticTuningResponse,
  SkipResponse,
  SynopsisResponse,
  ToggleListResponse,
  TuneResponse,
  TuneSubmission,
  UserProfileData,
  WatchHistoryEntry,
} from '../types';

const API_BASE_URL = '/api';
const SPOILER_TAG_NAMES = new Set(
  TAGS_VOCABULARY.filter((t) => t.is_media_spoiler).map((t) => t.name.toLowerCase())
);

// Reactive in-memory fallback state when backend is offline
let fallbackProfile: UserProfileData = {
  ...structuredClone(MOCK_USER_PROFILE),
  skipped_anime_ids: [],
  dropped_anime_ids: [],
  favorite_anime_ids: [],
  watchlist_anime_ids: [],
  excluded_formats: [],
  blocked_genres: [],
  google_account: null,
};
let fallbackHistory: WatchHistoryEntry[] = structuredClone(MOCK_WATCH_HISTORY);
let isBackendOnlineCache: boolean | null = null;

/**
 * Client-side defensive spoiler filter (R3 defense-in-depth).
 * Ensures no tag with is_media_spoiler: true ever reaches the UI DOM.
 */
export function sanitizePublicAnime(raw: RawAnimeFixture | AnimePublic): AnimePublic {
  const safeTags: string[] = [];
  for (const tag of raw.tags) {
    if (typeof tag === 'string') {
      if (!SPOILER_TAG_NAMES.has(tag.trim().toLowerCase())) {
        safeTags.push(tag);
      }
    } else if (tag && typeof tag === 'object' && !tag.is_media_spoiler) {
      if (!SPOILER_TAG_NAMES.has(tag.name.trim().toLowerCase())) {
        safeTags.push(tag.name);
      }
    }
  }

  return {
    id: raw.id,
    title_romaji: raw.title_romaji,
    title_english: raw.title_english ?? null,
    title_native: raw.title_native ?? null,
    cover_image: raw.cover_image,
    banner_image: raw.banner_image ?? null,
    format: raw.format,
    episodes: raw.episodes ?? null,
    duration: raw.duration ?? null,
    status: raw.status,
    season_year: raw.season_year ?? null,
    average_score: raw.average_score ?? null,
    popularity: raw.popularity,
    genres: [...raw.genres],
    tags: safeTags,
    compatibility_score: 'compatibility_score' in raw ? raw.compatibility_score : null,
    origin_track: 'origin_track' in raw ? raw.origin_track : null,
  };
}

/**
 * Helper to normalize a weight dictionary to unit L2 norm.
 */
function normalizeWeightsL2(weights: Record<string, number>): Record<string, number> {
  const entries = Object.entries(weights);
  const sumSq = entries.reduce((acc, [, val]) => acc + val * val, 0);
  const norm = Math.sqrt(sumSq);
  if (norm < 1e-9) return {};
  const normalized: Record<string, number> = {};
  for (const [key, val] of entries) {
    const v = Number((val / (norm + 1e-9)).toFixed(6));
    if (Math.abs(v) > 1e-6) {
      normalized[key] = v;
    }
  }
  return normalized;
}

/**
 * Computes offline fallback recommendations with 80/20 split, history/skip/drop exclusion,
 * and format/genre blocking filters.
 */
function computeFallbackRecommendations(
  profile: UserProfileData,
  history: WatchHistoryEntry[]
): RecommendationPayload {
  const excludedIds = new Set<number>([
    ...history.map((h) => h.anime_id),
    ...(profile.skipped_anime_ids ?? []),
    ...(profile.dropped_anime_ids ?? []),
  ]);
  const blockedFormats = new Set((profile.excluded_formats ?? []).map((f) => f.trim().toUpperCase()));
  const blockedGenres = new Set((profile.blocked_genres ?? []).map((g) => g.trim().toLowerCase()));

  const candidates = SAMPLE_ANIMES_FULL.filter((a) => {
    if (excludedIds.has(a.id)) return false;
    if (blockedFormats.size > 0 && blockedFormats.has((a.format || 'TV').trim().toUpperCase())) {
      return false;
    }
    if (
      blockedGenres.size > 0 &&
      a.genres.some((g) => blockedGenres.has(g.trim().toLowerCase()))
    ) {
      return false;
    }
    return true;
  });
  const pool = candidates;

  const scored = pool.map((anime) => {
    const rawFeatureMap: Record<string, number> = {};
    for (const g of anime.genres) rawFeatureMap[g] = 1.0;
    for (const t of anime.tags) rawFeatureMap[t.name] = Math.max(rawFeatureMap[t.name] ?? 0, t.rank / 100.0);
    const normAnime = normalizeWeightsL2(rawFeatureMap);

    let dot = 0;
    for (const [feat, w] of Object.entries(normAnime)) {
      dot += (profile.feature_weights[feat] ?? 0) * w;
    }
    const score = Math.max(0, Math.min(1, Number(((dot + 1) / 2).toFixed(3))));
    return { anime, score };
  });

  scored.sort((a, b) => b.score - a.score);

  const total = scored.length;
  const affinityCount = Math.max(1, Math.floor(total * 0.8));
  const affinitySlice = scored.slice(0, affinityCount);
  const discoverySlice = scored.slice(affinityCount);

  return {
    affinity_track: affinitySlice.map(({ anime, score }) => ({
      ...sanitizePublicAnime(anime),
      compatibility_score: score,
      origin_track: 'affinity',
    })),
    discovery_track: discoverySlice.map(({ anime, score }) => ({
      ...sanitizePublicAnime(anime),
      compatibility_score: score,
      origin_track: 'discovery',
    })),
    generated_at: new Date().toISOString(),
  };
}

/**
 * Checks whether the FastAPI backend is reachable.
 */
export async function checkBackendHealth(): Promise<HealthResponse & { online: boolean }> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${API_BASE_URL}/health`, { signal: controller.signal });
    clearTimeout(timer);
    if (res.ok) {
      const data = (await res.json()) as HealthResponse;
      isBackendOnlineCache = true;
      return { ...data, online: true };
    }
  } catch {
    // Offline fallback mode active
  }
  isBackendOnlineCache = false;
  return {
    status: 'offline_fallback',
    catalog_size: SAMPLE_ANIMES_FULL.length,
    cache_freshness: new Date().toISOString(),
    gemini_status: 'fixture_fallback',
    version: '1.0.0-fallback',
    online: false,
  };
}

/**
 * Fetches paginated catalog with optional search query and genre filter.
 */
export async function fetchCatalog(params?: {
  q?: string;
  genre?: string;
  page?: number;
  limit?: number;
}): Promise<AnimeCatalogResponse> {
  const page = params?.page ?? 1;
  const limit = params?.limit ?? 50;
  const q = params?.q?.trim() ?? '';
  const genre = params?.genre?.trim() ?? '';

  try {
    const searchParams = new URLSearchParams({
      page: String(page),
      limit: String(limit),
    });
    if (q) searchParams.set('q', q);
    if (genre) searchParams.set('genre', genre);

    const res = await fetch(`${API_BASE_URL}/catalog?${searchParams.toString()}`);
    if (res.ok) {
      const data = (await res.json()) as AnimeCatalogResponse;
      isBackendOnlineCache = true;
      return {
        ...data,
        items: data.items.map(sanitizePublicAnime),
      };
    }
  } catch {
    isBackendOnlineCache = false;
  }

  // Fallback to local fixtures
  let filtered = SAMPLE_ANIMES_FULL.map(sanitizePublicAnime);
  if (q) {
    const token = q.toLowerCase();
    filtered = filtered.filter(
      (a) =>
        a.title_romaji.toLowerCase().includes(token) ||
        (a.title_english && a.title_english.toLowerCase().includes(token)) ||
        a.genres.some((g) => g.toLowerCase().includes(token))
    );
  }
  if (genre) {
    const gToken = genre.toLowerCase();
    filtered = filtered.filter((a) => a.genres.some((g) => g.toLowerCase() === gToken));
  }

  return {
    items: filtered.slice((page - 1) * limit, page * limit),
    total: filtered.length,
    page,
    limit,
    total_pages: Math.max(1, Math.ceil(filtered.length / limit)),
  };
}

/**
 * Fetches dual-track (80% Affinity / 20% Discovery MMR) recommendations.
 */
export async function fetchRecommendations(topK = 10): Promise<RecommendationPayload> {
  try {
    const res = await fetch(`${API_BASE_URL}/recommendations?top_k=${topK}`);
    if (res.ok) {
      const data = (await res.json()) as RecommendationPayload;
      isBackendOnlineCache = true;
      return {
        ...data,
        affinity_track: data.affinity_track.map(sanitizePublicAnime),
        discovery_track: data.discovery_track.map(sanitizePublicAnime),
      };
    }
  } catch {
    isBackendOnlineCache = false;
  }

  return computeFallbackRecommendations(fallbackProfile, fallbackHistory);
}

/**
 * Fetches the current user profile vector and hyperparameters.
 */
export async function fetchUserProfile(): Promise<UserProfileData> {
  try {
    const res = await fetch(`${API_BASE_URL}/profile`);
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as UserProfileData;
    }
  } catch {
    isBackendOnlineCache = false;
  }
  return structuredClone(fallbackProfile);
}

/**
 * Fetches watch history and user lists (skipped, dropped, favorites, watchlist).
 */
export async function fetchWatchHistory(): Promise<HistoryResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/history`);
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as HistoryResponse;
    }
  } catch {
    isBackendOnlineCache = false;
  }
  return {
    items: structuredClone(fallbackHistory),
    skipped_anime_ids: [...(fallbackProfile.skipped_anime_ids ?? [])],
    dropped_anime_ids: [...(fallbackProfile.dropped_anime_ids ?? [])],
    favorite_anime_ids: [...(fallbackProfile.favorite_anime_ids ?? [])],
    watchlist_anime_ids: [...(fallbackProfile.watchlist_anime_ids ?? [])],
  };
}

/**
 * Submits a numerical rating [1.0, 10.0] and recalculates profile weights + recommendations.
 */
export async function submitRating(payload: RatingSubmission): Promise<RateResponse> {
  const clampedRating = Math.max(1.0, Math.min(10.0, payload.rating));

  try {
    const res = await fetch(`${API_BASE_URL}/rate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        anime_id: payload.anime_id,
        rating: clampedRating,
        origin_track: payload.origin_track,
        user_review: payload.user_review?.slice(0, 300),
      }),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      const data = (await res.json()) as RateResponse;
      if (data.updated_recommendations) {
        data.updated_recommendations = {
          ...data.updated_recommendations,
          affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
          discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
        };
      }
      return data;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  // Offline reactive update following exact mathematical model (gamma = 0.98, eta = 0.15, L2 unit norm)
  const targetAnime = SAMPLE_ANIMES_FULL.find((a) => a.id === payload.anime_id) ?? SAMPLE_ANIMES_FULL[0];
  const { eta, gamma } = fallbackProfile.hyperparameters;

  const nextWeights: Record<string, number> = {};
  for (const [k, v] of Object.entries(fallbackProfile.feature_weights)) {
    nextWeights[k] = v * gamma;
  }

  const rawAnimeFeatures: Record<string, number> = {};
  for (const g of targetAnime.genres) rawAnimeFeatures[g] = 1.0;
  for (const t of targetAnime.tags) rawAnimeFeatures[t.name] = t.rank / 100.0;
  const unitAnimeVec = normalizeWeightsL2(rawAnimeFeatures);

  const deltaScale = eta * (clampedRating - 5.5);
  for (const [feat, val] of Object.entries(unitAnimeVec)) {
    nextWeights[feat] = (nextWeights[feat] ?? 0) + deltaScale * val;
  }

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    total_ratings: fallbackProfile.total_ratings + 1,
    feature_weights: normalizeWeightsL2(nextWeights),
  };

  const entry: WatchHistoryEntry = {
    id: `wh-${Date.now()}`,
    anime_id: targetAnime.id,
    anime_title: targetAnime.title_english ?? targetAnime.title_romaji,
    rating: clampedRating,
    origin_track: payload.origin_track,
    interaction_status: 'watched',
    user_review_raw: payload.user_review?.slice(0, 300) ?? null,
    timestamp: new Date().toISOString(),
  };
  fallbackHistory = [...fallbackHistory, entry];

  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);

  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
    history_entry: entry,
    message: 'Avaliação registrada (Modo Local Fallback).',
  };
}

export interface EditRatingSubmission {
  anime_id: number;
  rating: number;
  user_review?: string;
}

/**
 * Edits an existing anime rating, updating watch history and recalibrating recommendation weights.
 */
export async function editRating(payload: EditRatingSubmission): Promise<RateResponse> {
  const clampedRating = Math.max(1.0, Math.min(10.0, payload.rating));

  try {
    const res = await fetch(`${API_BASE_URL}/rate/edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        anime_id: payload.anime_id,
        rating: clampedRating,
        user_review: payload.user_review?.slice(0, 300),
      }),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      const data = (await res.json()) as RateResponse;
      if (data.updated_recommendations) {
        data.updated_recommendations = {
          ...data.updated_recommendations,
          affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
          discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
        };
      }
      return data;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  const now = new Date().toISOString();
  let updatedEntry: WatchHistoryEntry | null = null;
  fallbackHistory = fallbackHistory.map((h) => {
    if (h.anime_id === payload.anime_id) {
      updatedEntry = {
        ...h,
        rating: clampedRating,
        user_review_raw: payload.user_review?.slice(0, 300) ?? h.user_review_raw,
        timestamp: now,
      };
      return updatedEntry;
    }
    return h;
  });

  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
    history_entry: updatedEntry ?? undefined,
    message: 'Avaliação atualizada.',
  };
}


/**
 * Submits qualitative feedback to Gemini for semantic vector calibration.
 */
export async function submitSemanticTuning(payload: TuneSubmission): Promise<TuneResponse> {
  const sanitizedFeedback = payload.user_feedback.trim().slice(0, 300);

  try {
    const res = await fetch(`${API_BASE_URL}/tune`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        anime_id: payload.anime_id,
        user_feedback: sanitizedFeedback,
      }),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as TuneResponse;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  // Offline reactive semantic tuning using alpha = 0.25 and L2 normalization
  const alpha = fallbackProfile.hyperparameters.alpha;
  const nextWeights: Record<string, number> = { ...fallbackProfile.feature_weights };

  for (const delta of MOCK_GEMINI_RESPONSE.identified_tags) {
    const clampedImpact = Math.max(-1.0, Math.min(1.0, delta.impact_multiplier));
    const clampedConf = Math.max(0.0, Math.min(1.0, delta.confidence));
    nextWeights[delta.tag_name] =
      (nextWeights[delta.tag_name] ?? 0) + clampedImpact * clampedConf * alpha;
  }

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    feature_weights: normalizeWeightsL2(nextWeights),
  };

  return {
    success: true,
    tuning_response: structuredClone(MOCK_GEMINI_RESPONSE),
    new_profile: structuredClone(fallbackProfile),
  };
}

/**
 * Skips (or unskips) an anime without altering the user's L2 preference weights.
 */
export async function skipAnime(animeId: number, unskip = false): Promise<SkipResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/skip`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ anime_id: animeId, unskip }),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      const data = (await res.json()) as SkipResponse;
      if (data.updated_recommendations) {
        data.updated_recommendations = {
          ...data.updated_recommendations,
          affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
          discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
        };
      }
      return data;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  const currentSkipped = new Set(fallbackProfile.skipped_anime_ids ?? []);
  if (unskip) {
    currentSkipped.delete(animeId);
  } else {
    currentSkipped.add(animeId);
  }
  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    skipped_anime_ids: Array.from(currentSkipped),
  };
  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);
  return {
    success: true,
    skipped_anime_ids: fallbackProfile.skipped_anime_ids ?? [],
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
  };
}

/**
 * Drops an anime with a written justification for Gemini to analyze and apply negative semantic tuning.
 */
export async function dropAnime(payload: DropSubmission): Promise<DropResponse> {
  const sanitizedReason = payload.drop_reason.trim().slice(0, 300);

  try {
    const res = await fetch(`${API_BASE_URL}/drop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        anime_id: payload.anime_id,
        drop_reason: sanitizedReason,
        origin_track: payload.origin_track ?? 'direct_search',
      }),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      const data = (await res.json()) as DropResponse;
      if (data.updated_recommendations) {
        data.updated_recommendations = {
          ...data.updated_recommendations,
          affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
          discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
        };
      }
      return data;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  // Offline fallback for drop analysis
  const targetAnime = SAMPLE_ANIMES_FULL.find((a) => a.id === payload.anime_id) ?? SAMPLE_ANIMES_FULL[0];
  const penalizedTags = targetAnime.genres.slice(0, 2);
  const deltas: SemanticTagDelta[] = penalizedTags.map((tag) => ({
    tag_name: tag,
    impact_multiplier: -0.8,
    confidence: 0.88,
    rationale: `Penalização semântica por drop associado a ${tag}.`,
  }));

  const alpha = fallbackProfile.hyperparameters.alpha;
  const nextWeights: Record<string, number> = { ...fallbackProfile.feature_weights };
  for (const d of deltas) {
    nextWeights[d.tag_name] = (nextWeights[d.tag_name] ?? 0) + d.impact_multiplier * d.confidence * alpha;
  }

  const droppedSet = new Set(fallbackProfile.dropped_anime_ids ?? []);
  droppedSet.add(payload.anime_id);

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    total_ratings: fallbackProfile.total_ratings + 1,
    feature_weights: normalizeWeightsL2(nextWeights),
    dropped_anime_ids: Array.from(droppedSet),
  };

  const tuningResp: SemanticTuningResponse = {
    identified_tags: deltas,
    public_safe_reason: `Nosso algoritmo analisou o motivo do drop e reduziu o peso de ${penalizedTags.join(', ')} no seu perfil.`,
  };

  const entry: WatchHistoryEntry = {
    id: `wh-${Date.now()}`,
    anime_id: targetAnime.id,
    anime_title: targetAnime.title_english ?? targetAnime.title_romaji,
    rating: 2.0,
    origin_track: payload.origin_track ?? 'direct_search',
    interaction_status: 'dropped',
    drop_reason: sanitizedReason,
    user_review_raw: sanitizedReason,
    qualitative_tuning: tuningResp,
    timestamp: new Date().toISOString(),
  };
  fallbackHistory = [...fallbackHistory, entry];

  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);

  return {
    success: true,
    tuning_response: tuningResp,
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
    history_entry: entry,
  };
}

/**
 * Toggles an anime in the user's Favorites list.
 */
export async function toggleFavorite(animeId: number): Promise<ToggleListResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/favorite`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ anime_id: animeId }),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as ToggleListResponse;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  const favs = new Set(fallbackProfile.favorite_anime_ids ?? []);
  let active = false;
  if (favs.has(animeId)) {
    favs.delete(animeId);
    active = false;
  } else {
    favs.add(animeId);
    active = true;
  }
  fallbackProfile = {
    ...fallbackProfile,
    favorite_anime_ids: Array.from(favs),
  };
  return {
    success: true,
    active,
    anime_ids: fallbackProfile.favorite_anime_ids ?? [],
    new_profile: structuredClone(fallbackProfile),
  };
}

/**
 * Toggles an anime in the user's Watchlist ('Minha Lista / Quero Assistir').
 */
export async function toggleWatchlist(animeId: number): Promise<ToggleListResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/watchlist`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ anime_id: animeId }),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as ToggleListResponse;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  const wl = new Set(fallbackProfile.watchlist_anime_ids ?? []);
  let active = false;
  if (wl.has(animeId)) {
    wl.delete(animeId);
    active = false;
  } else {
    wl.add(animeId);
    active = true;
  }
  fallbackProfile = {
    ...fallbackProfile,
    watchlist_anime_ids: Array.from(wl),
  };
  return {
    success: true,
    active,
    anime_ids: fallbackProfile.watchlist_anime_ids ?? [],
    new_profile: structuredClone(fallbackProfile),
  };
}

/**
 * Fetches all pages of the catalog so client-side lookup maps contain 100% of available titles.
 */
export async function fetchAllCatalogPages(): Promise<AnimePublic[]> {
  const firstPage = await fetchCatalog({ page: 1, limit: 100 });
  if (firstPage.total_pages <= 1) {
    return firstPage.items;
  }

  const pagePromises: Promise<AnimeCatalogResponse>[] = [];
  for (let p = 2; p <= firstPage.total_pages; p++) {
    pagePromises.push(fetchCatalog({ page: p, limit: 100 }));
  }
  const restPages = await Promise.all(pagePromises);
  const byId = new Map<number, AnimePublic>();
  for (const item of firstPage.items) {
    byId.set(item.id, item);
  }
  for (const pageRes of restPages) {
    for (const item of pageRes.items) {
      byId.set(item.id, item);
    }
  }
  return Array.from(byId.values());
}

/**
 * Updates format blocking (e.g. MOVIE, OVA, SPECIAL) and genre blocking (e.g. Mecha, Horror, Romance).
 */
export async function updateRecommendationFilters(
  payload: FilterPreferencesPayload
): Promise<FilterPreferencesResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/filters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      const data = (await res.json()) as FilterPreferencesResponse;
      if (data.updated_recommendations) {
        data.updated_recommendations = {
          ...data.updated_recommendations,
          affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
          discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
        };
      }
      return data;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    excluded_formats: payload.excluded_formats ?? fallbackProfile.excluded_formats ?? [],
    blocked_genres: payload.blocked_genres ?? fallbackProfile.blocked_genres ?? [],
  };
  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
    message: 'Filtros aplicados (Modo Local).',
  };
}

/**
 * Signs in with Google and saves/synchronizes user recommendation progress.
 */
export async function googleSignIn(payload: GoogleAuthPayload): Promise<GoogleAuthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      const data = (await res.json()) as GoogleAuthResponse;
      if (data.updated_recommendations) {
        data.updated_recommendations = {
          ...data.updated_recommendations,
          affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
          discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
        };
      }
      return data;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  const now = new Date().toISOString();
  const email = (payload.email || 'usuario@gmail.com').trim();
  const name = (payload.name || email.split('@')[0]).trim();
  fallbackProfile = {
    ...fallbackProfile,
    updated_at: now,
    google_account: {
      sub: payload.sub || `google-${email}`,
      email,
      name,
      picture: payload.picture || null,
      connected_at: now,
      last_synced_at: now,
    },
  };
  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    watch_history: structuredClone(fallbackHistory),
    updated_recommendations: updatedRecs,
    message: `Conectado como ${name} (${email}).`,
  };
}

/**
 * Explicitly synchronizes current user progress to the linked Google account backup.
 */
export async function syncGoogleProgress(): Promise<GoogleAuthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/auth/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as GoogleAuthResponse;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  if (fallbackProfile.google_account) {
    fallbackProfile = {
      ...fallbackProfile,
      google_account: {
        ...fallbackProfile.google_account,
        last_synced_at: new Date().toISOString(),
      },
    };
  }
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    watch_history: structuredClone(fallbackHistory),
    message: 'Progresso sincronizado!',
  };
}

/**
 * Logs out from the active Google session while keeping the cloud/local backup stored.
 */
export async function googleLogout(): Promise<GoogleAuthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/auth/logout`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as GoogleAuthResponse;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  fallbackProfile = {
    ...fallbackProfile,
    google_account: null,
  };
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    watch_history: structuredClone(fallbackHistory),
    message: 'Sessão Google desconectada.',
  };
}

/**
 * Resets user profile and watch history to cold-start state.
 */
export async function resetUserProfile(): Promise<UserProfileData> {
  try {
    const res = await fetch(`${API_BASE_URL}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirm: true }),
    });
    if (res.ok) {
      isBackendOnlineCache = true;
      const data = await res.json();
      if (data.profile) return data.profile as UserProfileData;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  // Fallback: Preserve existing google_account!
  const savedGoogle = fallbackProfile.google_account;
  fallbackProfile = {
    user_id: 'default_user',
    updated_at: new Date().toISOString(),
    total_ratings: 0,
    feature_weights: {},
    hyperparameters: { eta: 0.15, alpha: 0.25, gamma: 0.98, lambda_mmr: 0.5 },
    skipped_anime_ids: [],
    dropped_anime_ids: [],
    favorite_anime_ids: [],
    watchlist_anime_ids: [],
    excluded_formats: [],
    blocked_genres: [],
    google_account: savedGoogle,
  };
  fallbackHistory = [];
  return structuredClone(fallbackProfile);
}

/**
 * Fetches Portuguese synopsis and spoiler status for an anime.
 */
export async function getAnimeSynopsis(animeId: number): Promise<SynopsisResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/anime/${animeId}/synopsis`);
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as SynopsisResponse;
    }
  } catch {
    isBackendOnlineCache = false;
  }

  const found = SAMPLE_ANIMES_FULL.find((a) => a.id === animeId);
  return {
    anime_id: animeId,
    synopsis_pt: found?.description || null,
    synopsis_en: found?.description || null,
    has_spoiler_tags: Boolean(found?.tags?.some((t) => typeof t === 'object' && t.is_media_spoiler)),
  };
}

/**
 * Synchronizes user lists and ratings from AniList.
 */
export async function syncAniList(payload: AniListSyncRequest): Promise<AniListSyncResponse> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    const res = await fetch(`${API_BASE_URL}/auth/anilist/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as AniListSyncResponse;
    }
    const errData = await res.json().catch(() => null);
    throw new Error(errData?.detail || `Erro HTTP ${res.status} ao sincronizar com a AniList.`);
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error('Tempo limite de sincronização excedido (15s). O servidor ou o AniList demoraram para responder.');
    }
    if (isBackendOnlineCache !== false && err instanceof Error && !err.message.includes('Failed to fetch')) {
      throw err;
    }
    isBackendOnlineCache = false;
    throw new Error(err instanceof Error ? err.message : 'Falha na conexão com o servidor de sincronização.');
  }
}

/**
 * Returns current connection mode status.
 */
export function getBackendConnectionStatus(): boolean {
  return Boolean(isBackendOnlineCache);
}

export interface SeasonalAnimeItem {
  id: number;
  title_romaji: string;
  title_english: string;
  title_native?: string;
  cover_image: string;
  banner_image?: string;
  format?: string;
  episodes?: number | null;
  status?: string;
  season?: string;
  season_year?: number;
  average_score?: number | null;
  popularity?: number;
  genres: string[];
  description?: string;
  studio?: string | null;
  next_episode?: number | null;
  time_until_airing?: number | null;
}

export interface SeasonalResponse {
  season: string;
  year: number;
  updated_at?: string;
  items: SeasonalAnimeItem[];
  message?: string;
}

/**
 * Fetches current seasonal anime releases.
 */
export async function fetchSeasonalReleases(): Promise<SeasonalResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/season/releases`);
    if (res.ok) {
      isBackendOnlineCache = true;
      return (await res.json()) as SeasonalResponse;
    }
  } catch {
    isBackendOnlineCache = false;
  }
  return {
    season: 'ATUAL',
    year: new Date().getFullYear(),
    items: [],
  };
}
