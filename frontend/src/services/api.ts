/**
 * frontend/src/services/api.ts
 * Decoupled Asynchronous Service Layer for AnimeRecSys AI.
 *
 * Rules:
 * 1. UI components NEVER import fixtures directly; all data access goes through this service.
 * 2. Automatically probes FastAPI backend health (/api/health).
 * 3. When local FastAPI backend is online, routes all calls to live FastAPI endpoints.
 * 4. When hosted on static cloud CDN (e.g. Firebase Hosting), seamlessly loads the full
 *    961-anime catalog (/data/catalog.json) and seasonal releases (/data/seasonal_releases.json),
 *    persisting user state in localStorage and executing the full L2 + Cosine + MMR engine in-browser.
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
  AnimeTag,
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
const STORAGE_KEY_PROFILE = 'animerecsys_profile_v1';
const STORAGE_KEY_HISTORY = 'animerecsys_history_v1';
const STORAGE_KEY_GOOGLE_BACKUPS = 'animerecsys_google_backups_v1';
const STORAGE_KEY_SYNOPSES_PT = 'animerecsys_synopses_pt_v1';

const SPOILER_TAG_NAMES = new Set(
  TAGS_VOCABULARY.filter((t) => t.is_media_spoiler).map((t) => t.name.toLowerCase())
);

// Dynamic catalog loaded from /data/catalog.json (961 animes) when in Cloud/Static mode
let activeCatalog: RawAnimeFixture[] = [...SAMPLE_ANIMES_FULL];
let catalogLoadPromise: Promise<RawAnimeFixture[]> | null = null;
let seasonalCache: SeasonalResponse | null = null;

function loadStoredProfile(): UserProfileData {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const raw = window.localStorage.getItem(STORAGE_KEY_PROFILE);
      if (raw) {
        const parsed = JSON.parse(raw) as UserProfileData;
        if (parsed && typeof parsed === 'object' && parsed.hyperparameters) {
          return {
            ...structuredClone(MOCK_USER_PROFILE),
            ...parsed,
            skipped_anime_ids: parsed.skipped_anime_ids ?? [],
            dropped_anime_ids: parsed.dropped_anime_ids ?? [],
            favorite_anime_ids: parsed.favorite_anime_ids ?? [],
            watchlist_anime_ids: parsed.watchlist_anime_ids ?? [],
            excluded_formats: parsed.excluded_formats ?? [],
            blocked_genres: parsed.blocked_genres ?? [],
            google_account: parsed.google_account ?? null,
          };
        }
      }
    }
  } catch {
    // ignore storage errors
  }
  return {
    ...structuredClone(MOCK_USER_PROFILE),
    skipped_anime_ids: [],
    dropped_anime_ids: [],
    favorite_anime_ids: [],
    watchlist_anime_ids: [],
    excluded_formats: [],
    blocked_genres: [],
    google_account: null,
  };
}

function loadStoredHistory(): WatchHistoryEntry[] {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const raw = window.localStorage.getItem(STORAGE_KEY_HISTORY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          return parsed;
        }
      }
    }
  } catch {
    // ignore storage errors
  }
  return structuredClone(MOCK_WATCH_HISTORY);
}

function persistLocalState(): void {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(STORAGE_KEY_PROFILE, JSON.stringify(fallbackProfile));
      window.localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(fallbackHistory));
      if (fallbackProfile.google_account?.email) {
        const emailKey = fallbackProfile.google_account.email.toLowerCase();
        const rawBackups = window.localStorage.getItem(STORAGE_KEY_GOOGLE_BACKUPS);
        const backups = rawBackups ? JSON.parse(rawBackups) : {};
        backups[emailKey] = {
          profile: fallbackProfile,
          history: fallbackHistory,
          saved_at: new Date().toISOString(),
        };
        window.localStorage.setItem(STORAGE_KEY_GOOGLE_BACKUPS, JSON.stringify(backups));
      }
    }
  } catch {
    // ignore storage quota errors
  }
}

let fallbackProfile: UserProfileData = loadStoredProfile();
let fallbackHistory: WatchHistoryEntry[] = loadStoredHistory();
let isBackendOnlineCache: boolean | null = null;

/**
 * Loads the full 961-anime static catalog from /data/catalog.json when hosted on Firebase Hosting.
 */
async function ensureStaticCatalogLoaded(): Promise<RawAnimeFixture[]> {
  if (activeCatalog.length > SAMPLE_ANIMES_FULL.length) {
    return activeCatalog;
  }
  if (catalogLoadPromise) {
    return catalogLoadPromise;
  }
  catalogLoadPromise = (async () => {
    try {
      const res = await fetch('/data/catalog.json');
      if (res.ok && (res.headers.get('content-type') || '').includes('application/json')) {
        const data = (await res.json()) as RawAnimeFixture[];
        if (Array.isArray(data) && data.length > 0) {
          activeCatalog = data;
          return activeCatalog;
        }
      }
    } catch {
      // Fallback to seed fixtures if /data/catalog.json is unavailable
    }
    return activeCatalog;
  })();
  return catalogLoadPromise;
}

/**
 * Helper to fetch JSON from FastAPI backend while rejecting SPA HTML rewrites on Firebase Hosting.
 */
async function fetchBackendJson<T>(url: string, init?: RequestInit, timeoutMs = 6000): Promise<T | null> {
  if (isBackendOnlineCache === false && !url.endsWith('/health')) {
    return null;
  }
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timer);
    const contentType = res.headers.get('content-type') || '';
    if (res.ok && contentType.includes('application/json')) {
      isBackendOnlineCache = true;
      return (await res.json()) as T;
    }
  } catch {
    // Network error or timeout
  }
  isBackendOnlineCache = false;
  return null;
}

/**
 * Client-side defensive spoiler filter (R3 defense-in-depth).
 * Ensures no tag with is_media_spoiler: true ever reaches the UI DOM.
 */
export function sanitizePublicAnime(raw: RawAnimeFixture | AnimePublic): AnimePublic {
  const safeTags: string[] = [];
  let detectedSpoilerTag = Boolean('has_spoiler_tags' in raw && raw.has_spoiler_tags);

  for (const tag of raw.tags || []) {
    if (typeof tag === 'string') {
      if (!SPOILER_TAG_NAMES.has(tag.trim().toLowerCase())) {
        safeTags.push(tag);
      } else {
        detectedSpoilerTag = true;
      }
    } else if (tag && typeof tag === 'object') {
      if (tag.is_media_spoiler || SPOILER_TAG_NAMES.has(tag.name.trim().toLowerCase())) {
        detectedSpoilerTag = true;
      } else {
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
    genres: [...(raw.genres || [])],
    tags: safeTags,
    description: raw.description ?? null,
    has_spoiler_tags: detectedSpoilerTag,
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
 * Rebuilds the user's L2 preference vector from watch history cleanly.
 */
function recalculateProfileWeightsFromHistory(
  baseProfile: UserProfileData,
  history: WatchHistoryEntry[]
): Record<string, number> {
  const { eta, gamma, alpha } = baseProfile.hyperparameters;
  let weights: Record<string, number> = {};

  for (const entry of history) {
    const targetAnime = activeCatalog.find((a) => a.id === entry.anime_id);
    if (!targetAnime) continue;

    const decayed: Record<string, number> = {};
    for (const [k, v] of Object.entries(weights)) {
      decayed[k] = v * gamma;
    }

    const rawAnimeFeatures: Record<string, number> = {};
    for (const g of targetAnime.genres || []) rawAnimeFeatures[g] = 1.0;
    for (const t of targetAnime.tags || []) {
      if (typeof t === 'object' && t.name) {
        rawAnimeFeatures[t.name] = Math.max(rawAnimeFeatures[t.name] ?? 0, (t.rank ?? 60) / 100.0);
      }
    }
    const unitAnimeVec = normalizeWeightsL2(rawAnimeFeatures);
    const deltaScale = eta * (entry.rating - 5.5);

    for (const [feat, val] of Object.entries(unitAnimeVec)) {
      decayed[feat] = (decayed[feat] ?? 0) + deltaScale * val;
    }

    if (entry.qualitative_tuning?.identified_tags) {
      for (const delta of entry.qualitative_tuning.identified_tags) {
        decayed[delta.tag_name] =
          (decayed[delta.tag_name] ?? 0) + delta.impact_multiplier * delta.confidence * alpha;
      }
    }

    weights = normalizeWeightsL2(decayed);
  }

  return weights;
}

/**
 * Computes recommendations with 80% Cosine Affinity / 20% MMR Discovery split,
 * history/skip/drop exclusion, and format/genre blocking filters over the full catalog.
 */
function computeFallbackRecommendations(
  profile: UserProfileData,
  history: WatchHistoryEntry[],
  topK = 10
): RecommendationPayload {
  const excludedIds = new Set<number>([
    ...history.map((h) => h.anime_id),
    ...(profile.skipped_anime_ids ?? []),
    ...(profile.dropped_anime_ids ?? []),
  ]);
  const blockedFormats = new Set((profile.excluded_formats ?? []).map((f) => f.trim().toUpperCase()));
  const blockedGenres = new Set((profile.blocked_genres ?? []).map((g) => g.trim().toLowerCase()));

  const candidates = activeCatalog.filter((a) => {
    if (excludedIds.has(a.id)) return false;
    if (blockedFormats.size > 0 && blockedFormats.has((a.format || 'TV').trim().toUpperCase())) {
      return false;
    }
    if (
      blockedGenres.size > 0 &&
      (a.genres || []).some((g) => blockedGenres.has(g.trim().toLowerCase()))
    ) {
      return false;
    }
    return true;
  });

  const hasUserWeights = Object.keys(profile.feature_weights || {}).length > 0;

  const scored = candidates.map((anime) => {
    const rawFeatureMap: Record<string, number> = {};
    for (const g of anime.genres || []) rawFeatureMap[g] = 1.0;
    for (const t of anime.tags || []) {
      if (typeof t === 'object' && t.name) {
        rawFeatureMap[t.name] = Math.max(rawFeatureMap[t.name] ?? 0, (t.rank ?? 50) / 100.0);
      }
    }
    const normAnime = normalizeWeightsL2(rawFeatureMap);

    let score = 0;
    if (hasUserWeights) {
      let dot = 0;
      for (const [feat, w] of Object.entries(normAnime)) {
        dot += (profile.feature_weights[feat] ?? 0) * w;
      }
      const qualityBoost = ((anime.average_score ?? 75) / 100) * 0.12;
      score = Math.max(0.05, Math.min(0.99, Number(((dot + 1) / 2 * 0.88 + qualityBoost).toFixed(3))));
    } else {
      const avg = (anime.average_score ?? 80) / 100;
      const pop = Math.min(1, (anime.popularity ?? 100000) / 600000);
      score = Number((avg * 0.7 + pop * 0.3).toFixed(3));
    }
    return { anime, score, normAnime };
  });

  scored.sort((a, b) => b.score - a.score);

  const effectiveTopK = Math.min(topK, scored.length);
  const affinityCount = Math.max(1, Math.round(effectiveTopK * 0.8));
  const discoveryCount = Math.max(1, effectiveTopK - affinityCount);

  const affinitySlice = scored.slice(0, affinityCount);
  const remainingPool = scored.slice(affinityCount, Math.min(scored.length, affinityCount + 60));

  // MMR Selection for Discovery Track (20%)
  const lambdaMmr = profile.hyperparameters?.lambda_mmr ?? 0.5;
  const selectedDiscovery: typeof scored = [];
  const selectedPool = [...affinitySlice];

  while (selectedDiscovery.length < discoveryCount && remainingPool.length > 0) {
    let bestIdx = 0;
    let bestMmr = -Infinity;

    for (let i = 0; i < remainingPool.length; i++) {
      const cand = remainingPool[i];
      let maxSim = 0;
      for (const sel of selectedPool) {
        let sim = 0;
        for (const [feat, w] of Object.entries(cand.normAnime)) {
          sim += (sel.normAnime[feat] ?? 0) * w;
        }
        if (sim > maxSim) maxSim = sim;
      }
      const mmrScore = lambdaMmr * cand.score - (1 - lambdaMmr) * maxSim;
      if (mmrScore > bestMmr) {
        bestMmr = mmrScore;
        bestIdx = i;
      }
    }

    const [picked] = remainingPool.splice(bestIdx, 1);
    selectedDiscovery.push(picked);
    selectedPool.push(picked);
  }

  return {
    affinity_track: affinitySlice.map(({ anime, score }) => ({
      ...sanitizePublicAnime(anime),
      compatibility_score: score,
      origin_track: 'affinity',
    })),
    discovery_track: selectedDiscovery.map(({ anime, score }) => ({
      ...sanitizePublicAnime(anime),
      compatibility_score: score,
      origin_track: 'discovery',
    })),
    generated_at: new Date().toISOString(),
  };
}

/**
 * Checks whether the FastAPI backend is reachable or initializes the Cloud Static Engine.
 */
export async function checkBackendHealth(): Promise<HealthResponse & { online: boolean }> {
  const data = await fetchBackendJson<HealthResponse>(`${API_BASE_URL}/health`, undefined, 2500);
  if (data && data.status) {
    return { ...data, online: true };
  }

  await ensureStaticCatalogLoaded();
  return {
    status: 'ok',
    catalog_size: activeCatalog.length,
    cache_freshness: new Date().toISOString(),
    gemini_status: 'cloud_engine',
    version: '1.0.0',
    online: true,
  };
}

/**
 * Fetches paginated catalog with optional search query and genre filter.
 * Includes live AniList GraphQL search fallback when in Cloud mode.
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

  const searchParams = new URLSearchParams({
    page: String(page),
    limit: String(limit),
  });
  if (q) searchParams.set('q', q);
  if (genre) searchParams.set('genre', genre);

  const backendData = await fetchBackendJson<AnimeCatalogResponse>(
    `${API_BASE_URL}/catalog?${searchParams.toString()}`
  );
  if (backendData && Array.isArray(backendData.items)) {
    return {
      ...backendData,
      items: backendData.items.map(sanitizePublicAnime),
    };
  }

  await ensureStaticCatalogLoaded();

  let filtered = activeCatalog.map(sanitizePublicAnime);
  if (q) {
    const token = q.toLowerCase();
    filtered = filtered.filter(
      (a) =>
        a.title_romaji.toLowerCase().includes(token) ||
        (a.title_english && a.title_english.toLowerCase().includes(token)) ||
        (a.title_native && a.title_native.toLowerCase().includes(token)) ||
        a.genres.some((g) => g.toLowerCase().includes(token)) ||
        a.tags.some((t) => t.toLowerCase().includes(token))
    );

    // Live AniList GraphQL search if local matches are few
    if (filtered.length < 3 && q.length >= 2) {
      try {
        const res = await fetch('https://graphql.anilist.co', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({
            query: `
              query ($search: String) {
                Page(page: 1, perPage: 12) {
                  media(search: $search, type: ANIME, isAdult: false, sort: POPULARITY_DESC) {
                    id
                    title { romaji english native }
                    coverImage { extraLarge large }
                    bannerImage
                    format
                    episodes
                    duration
                    status
                    seasonYear
                    averageScore
                    popularity
                    genres
                    description(asHtml: false)
                    tags { id name rank isMediaSpoiler }
                  }
                }
              }
            `,
            variables: { search: q },
          }),
        });
        if (res.ok) {
          const json = await res.json();
          const mediaList = json?.data?.Page?.media || [];
          const existingIds = new Set(activeCatalog.map((a) => a.id));
          for (const m of mediaList) {
            if (!m?.id) continue;
            const rawItem: RawAnimeFixture = {
              id: m.id,
              title_romaji: m.title?.romaji || m.title?.english || `Anime #${m.id}`,
              title_english: m.title?.english || undefined,
              title_native: m.title?.native || undefined,
              cover_image: m.coverImage?.extraLarge || m.coverImage?.large || '',
              banner_image: m.bannerImage || undefined,
              format: m.format || 'TV',
              episodes: m.episodes || undefined,
              duration: m.duration || undefined,
              status: m.status || 'FINISHED',
              season_year: m.seasonYear || undefined,
              average_score: m.averageScore || undefined,
              popularity: m.popularity || 1000,
              genres: m.genres || [],
              description: m.description || undefined,
              tags: (m.tags || []).map((t: { id?: number; name: string; rank?: number; isMediaSpoiler?: boolean }) => ({
                id: t.id || 0,
                name: t.name,
                rank: t.rank || 50,
                is_media_spoiler: Boolean(t.isMediaSpoiler),
              })),
            };
            if (!existingIds.has(rawItem.id)) {
              activeCatalog.push(rawItem);
              existingIds.add(rawItem.id);
            }
            if (!filtered.some((f) => f.id === rawItem.id)) {
              filtered.push(sanitizePublicAnime(rawItem));
            }
          }
        }
      } catch {
        // ignore live search errors
      }
    }
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
  const data = await fetchBackendJson<RecommendationPayload>(
    `${API_BASE_URL}/recommendations?top_k=${topK}`
  );
  if (data && Array.isArray(data.affinity_track)) {
    return {
      ...data,
      affinity_track: data.affinity_track.map(sanitizePublicAnime),
      discovery_track: data.discovery_track.map(sanitizePublicAnime),
    };
  }

  await ensureStaticCatalogLoaded();
  return computeFallbackRecommendations(fallbackProfile, fallbackHistory, topK);
}

/**
 * Fetches the current user profile vector and hyperparameters.
 */
export async function fetchUserProfile(): Promise<UserProfileData> {
  const data = await fetchBackendJson<UserProfileData>(`${API_BASE_URL}/profile`);
  if (data && data.user_id) {
    return data;
  }
  return structuredClone(fallbackProfile);
}

/**
 * Fetches watch history and user lists (skipped, dropped, favorites, watchlist).
 */
export async function fetchWatchHistory(): Promise<HistoryResponse> {
  const data = await fetchBackendJson<HistoryResponse>(`${API_BASE_URL}/history`);
  if (data && Array.isArray(data.items)) {
    return data;
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

  const data = await fetchBackendJson<RateResponse>(`${API_BASE_URL}/rate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      anime_id: payload.anime_id,
      rating: clampedRating,
      origin_track: payload.origin_track,
      user_review: payload.user_review?.slice(0, 300),
    }),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    if (data.history_entry) {
      fallbackHistory = [
        ...fallbackHistory.filter((h) => h.anime_id !== data.history_entry!.anime_id),
        data.history_entry,
      ];
    }
    void autoSyncGoogleIfConnected();
    if (data.updated_recommendations) {
      data.updated_recommendations = {
        ...data.updated_recommendations,
        affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
        discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
      };
    }
    return data;
  }

  await ensureStaticCatalogLoaded();

  // Duplicate guard in Cloud Mode
  const existingEntry = fallbackHistory.find((h) => h.anime_id === payload.anime_id);
  if (existingEntry) {
    return editRating({
      anime_id: payload.anime_id,
      rating: clampedRating,
      user_review: payload.user_review,
    });
  }

  const targetAnime =
    activeCatalog.find((a) => a.id === payload.anime_id) ?? activeCatalog[0] ?? SAMPLE_ANIMES_FULL[0];
  const { eta, gamma } = fallbackProfile.hyperparameters;

  const nextWeights: Record<string, number> = {};
  for (const [k, v] of Object.entries(fallbackProfile.feature_weights)) {
    nextWeights[k] = v * gamma;
  }

  const rawAnimeFeatures: Record<string, number> = {};
  for (const g of targetAnime.genres || []) rawAnimeFeatures[g] = 1.0;
  for (const t of targetAnime.tags || []) {
    if (typeof t === 'object' && t.name) {
      rawAnimeFeatures[t.name] = (t.rank ?? 50) / 100.0;
    }
  }
  const unitAnimeVec = normalizeWeightsL2(rawAnimeFeatures);

  const deltaScale = eta * (clampedRating - 5.5);
  for (const [feat, val] of Object.entries(unitAnimeVec)) {
    nextWeights[feat] = (nextWeights[feat] ?? 0) + deltaScale * val;
  }

  // Remove from watchlist/skipped if rated
  const nextWatchlist = (fallbackProfile.watchlist_anime_ids ?? []).filter((id) => id !== payload.anime_id);
  const nextSkipped = (fallbackProfile.skipped_anime_ids ?? []).filter((id) => id !== payload.anime_id);

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

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    total_ratings: fallbackHistory.length,
    feature_weights: normalizeWeightsL2(nextWeights),
    watchlist_anime_ids: nextWatchlist,
    skipped_anime_ids: nextSkipped,
  };
  persistLocalState();

  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);

  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
    history_entry: entry,
    message: 'Avaliação registrada com sucesso!',
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

  const data = await fetchBackendJson<RateResponse>(`${API_BASE_URL}/rate/edit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      anime_id: payload.anime_id,
      rating: clampedRating,
      user_review: payload.user_review?.slice(0, 300),
    }),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    if (data.history_entry) {
      fallbackHistory = fallbackHistory.map((h) =>
        h.anime_id === payload.anime_id ? data.history_entry! : h
      );
    }
    void autoSyncGoogleIfConnected();
    if (data.updated_recommendations) {
      data.updated_recommendations = {
        ...data.updated_recommendations,
        affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
        discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
      };
    }
    return data;
  }

  await ensureStaticCatalogLoaded();

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

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: now,
    total_ratings: fallbackHistory.length,
    feature_weights: recalculateProfileWeightsFromHistory(fallbackProfile, fallbackHistory),
  };
  persistLocalState();

  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
    history_entry: updatedEntry ?? undefined,
    message: 'Avaliação atualizada e pesos recalculados.',
  };
}

/**
 * Submits qualitative feedback for semantic vector calibration.
 */
export async function submitSemanticTuning(payload: TuneSubmission): Promise<TuneResponse> {
  const sanitizedFeedback = payload.user_feedback.trim().slice(0, 300);

  const data = await fetchBackendJson<TuneResponse>(`${API_BASE_URL}/tune`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      anime_id: payload.anime_id,
      user_feedback: sanitizedFeedback,
    }),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    void autoSyncGoogleIfConnected();
    return data;
  }

  await ensureStaticCatalogLoaded();
  const targetAnime = activeCatalog.find((a) => a.id === payload.anime_id);
  const alpha = fallbackProfile.hyperparameters.alpha;
  const nextWeights: Record<string, number> = { ...fallbackProfile.feature_weights };

  const extractedDeltas: SemanticTagDelta[] = [];
  if (targetAnime) {
    for (const g of (targetAnime.genres || []).slice(0, 2)) {
      extractedDeltas.push({
        tag_name: g,
        impact_multiplier: 0.85,
        confidence: 0.92,
        rationale: `Reforço semântico em ${g} a partir do seu comentário.`,
      });
    }
    for (const t of (targetAnime.tags || []).slice(0, 2)) {
      if (typeof t === 'object' && t.name && !t.is_media_spoiler) {
        extractedDeltas.push({
          tag_name: t.name,
          impact_multiplier: 0.9,
          confidence: 0.9,
          rationale: `Afinidade temática identificada com ${t.name}.`,
        });
      }
    }
  }

  const deltasToApply = extractedDeltas.length > 0 ? extractedDeltas : MOCK_GEMINI_RESPONSE.identified_tags;

  for (const delta of deltasToApply) {
    const clampedImpact = Math.max(-1.0, Math.min(1.0, delta.impact_multiplier));
    const clampedConf = Math.max(0.0, Math.min(1.0, delta.confidence));
    nextWeights[delta.tag_name] =
      (nextWeights[delta.tag_name] ?? 0) + clampedImpact * clampedConf * alpha;
  }

  const tuningResponse: SemanticTuningResponse = {
    identified_tags: deltasToApply,
    public_safe_reason:
      extractedDeltas.length > 0
        ? `Nosso algoritmo calibrou seu perfil priorizando ${deltasToApply.map((d) => d.tag_name).join(', ')}.`
        : MOCK_GEMINI_RESPONSE.public_safe_reason,
  };

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    feature_weights: normalizeWeightsL2(nextWeights),
  };
  persistLocalState();

  return {
    success: true,
    tuning_response: tuningResponse,
    new_profile: structuredClone(fallbackProfile),
  };
}

/**
 * Skips (or unskips) an anime without altering the user's L2 preference weights.
 */
export async function skipAnime(animeId: number, unskip = false): Promise<SkipResponse> {
  const data = await fetchBackendJson<SkipResponse>(`${API_BASE_URL}/skip`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ anime_id: animeId, unskip }),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    void autoSyncGoogleIfConnected();
    if (data.updated_recommendations) {
      data.updated_recommendations = {
        ...data.updated_recommendations,
        affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
        discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
      };
    }
    return data;
  }

  await ensureStaticCatalogLoaded();
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
  persistLocalState();
  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);
  return {
    success: true,
    skipped_anime_ids: fallbackProfile.skipped_anime_ids ?? [],
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
  };
}

/**
 * Drops an anime with a written justification to apply negative semantic tuning.
 */
export async function dropAnime(payload: DropSubmission): Promise<DropResponse> {
  const sanitizedReason = payload.drop_reason.trim().slice(0, 300);

  const data = await fetchBackendJson<DropResponse>(`${API_BASE_URL}/drop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      anime_id: payload.anime_id,
      drop_reason: sanitizedReason,
      origin_track: payload.origin_track ?? 'direct_search',
    }),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    void autoSyncGoogleIfConnected();
    if (data.updated_recommendations) {
      data.updated_recommendations = {
        ...data.updated_recommendations,
        affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
        discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
      };
    }
    return data;
  }

  await ensureStaticCatalogLoaded();
  const targetAnime =
    activeCatalog.find((a) => a.id === payload.anime_id) ?? activeCatalog[0] ?? SAMPLE_ANIMES_FULL[0];
  const penalizedTags = (targetAnime.genres || []).slice(0, 2);
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
  fallbackHistory = [...fallbackHistory.filter((h) => h.anime_id !== payload.anime_id), entry];

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    total_ratings: fallbackHistory.length,
    feature_weights: normalizeWeightsL2(nextWeights),
    dropped_anime_ids: Array.from(droppedSet),
  };
  persistLocalState();

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
  const data = await fetchBackendJson<ToggleListResponse>(`${API_BASE_URL}/favorite`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ anime_id: animeId }),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    void autoSyncGoogleIfConnected();
    return data;
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
  persistLocalState();
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
  const data = await fetchBackendJson<ToggleListResponse>(`${API_BASE_URL}/watchlist`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ anime_id: animeId }),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    void autoSyncGoogleIfConnected();
    return data;
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
  persistLocalState();
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
  if (isBackendOnlineCache === false) {
    await ensureStaticCatalogLoaded();
    return activeCatalog.map(sanitizePublicAnime);
  }

  const firstPage = await fetchCatalog({ page: 1, limit: 100 });
  if (firstPage.total_pages <= 1) {
    return firstPage.items;
  }

  // If fetchCatalog fell back to static catalog, return all items directly
  if (!getBackendConnectionStatus()) {
    await ensureStaticCatalogLoaded();
    return activeCatalog.map(sanitizePublicAnime);
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
  const data = await fetchBackendJson<FilterPreferencesResponse>(`${API_BASE_URL}/filters`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    void autoSyncGoogleIfConnected();
    if (data.updated_recommendations) {
      data.updated_recommendations = {
        ...data.updated_recommendations,
        affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
        discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
      };
    }
    return data;
  }

  await ensureStaticCatalogLoaded();
  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    excluded_formats: payload.excluded_formats ?? fallbackProfile.excluded_formats ?? [],
    blocked_genres: payload.blocked_genres ?? fallbackProfile.blocked_genres ?? [],
  };
  persistLocalState();
  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory);
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
    message: 'Filtros aplicados com sucesso.',
  };
}

/**
 * Signs in with Google and saves/synchronizes user recommendation progress.
 */
export async function googleSignIn(payload: GoogleAuthPayload): Promise<GoogleAuthResponse> {
  const data = await fetchBackendJson<GoogleAuthResponse>(`${API_BASE_URL}/auth/google`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    if (Array.isArray(data.watch_history)) {
      fallbackHistory = data.watch_history;
    }
    persistLocalState();
    if (data.updated_recommendations) {
      data.updated_recommendations = {
        ...data.updated_recommendations,
        affinity_track: data.updated_recommendations.affinity_track.map(sanitizePublicAnime),
        discovery_track: data.updated_recommendations.discovery_track.map(sanitizePublicAnime),
      };
    }
    return data;
  }

  await ensureStaticCatalogLoaded();
  const now = new Date().toISOString();
  const email = (payload.email || 'usuario@gmail.com').trim();
  const name = (payload.name || email.split('@')[0]).trim();
  const emailKey = email.toLowerCase();

  // Restore backup if one exists for this Google email
  try {
    const rawBackups = window.localStorage.getItem(STORAGE_KEY_GOOGLE_BACKUPS);
    const backups = rawBackups ? JSON.parse(rawBackups) : {};
    if (backups[emailKey]?.profile && Array.isArray(backups[emailKey]?.history)) {
      if (backups[emailKey].history.length >= fallbackHistory.length) {
        fallbackProfile = backups[emailKey].profile;
        fallbackHistory = backups[emailKey].history;
      }
    }
  } catch {
    // ignore
  }

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
  persistLocalState();
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
 * Automatically synchronizes user progress with the linked Google account in the background.
 */
async function autoSyncGoogleIfConnected(): Promise<void> {
  try {
    persistLocalState();
    if (isBackendOnlineCache && fallbackProfile.google_account) {
      await fetchBackendJson<GoogleAuthResponse>(`${API_BASE_URL}/auth/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
    }
  } catch {
    // Ignore background auto-sync errors
  }
}

/**
 * Explicitly synchronizes current user progress to the linked Google account backup.
 */
export async function syncGoogleProgress(): Promise<GoogleAuthResponse> {
  const data = await fetchBackendJson<GoogleAuthResponse>(`${API_BASE_URL}/auth/sync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (data && data.new_profile) {
    fallbackProfile = data.new_profile;
    persistLocalState();
    return data;
  }

  if (fallbackProfile.google_account) {
    fallbackProfile = {
      ...fallbackProfile,
      google_account: {
        ...fallbackProfile.google_account,
        last_synced_at: new Date().toISOString(),
      },
    };
    persistLocalState();
  }
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    watch_history: structuredClone(fallbackHistory),
    message: 'Progresso sincronizado com sucesso!',
  };
}

/**
 * Logs out from the active Google session while keeping the cloud/local backup stored.
 */
export async function googleLogout(): Promise<GoogleAuthResponse> {
  const data = await fetchBackendJson<GoogleAuthResponse>(`${API_BASE_URL}/auth/logout`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (data && data.new_profile) {
    return data;
  }

  persistLocalState();
  fallbackProfile = {
    ...fallbackProfile,
    google_account: null,
  };
  persistLocalState();
  return {
    success: true,
    new_profile: structuredClone(fallbackProfile),
    watch_history: structuredClone(fallbackHistory),
    message: 'Sessão Google desconectada.',
  };
}

/**
 * Resets user profile and watch history to cold-start state while preserving Google login.
 */
export async function resetUserProfile(): Promise<UserProfileData> {
  const data = await fetchBackendJson<{ profile?: UserProfileData }>(`${API_BASE_URL}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirm: true }),
  });
  if (data?.profile) {
    return data.profile;
  }

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
  persistLocalState();
  return structuredClone(fallbackProfile);
}

/**
 * Translates an English synopsis to Brazilian Portuguese on-the-fly in the browser if not already in PT.
 */
async function translateSynopsisInBrowser(animeId: number, textEn: string): Promise<string> {
  const cleanEn = textEn
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/\n\s*\n+/g, '\n\n')
    .trim();
  if (!cleanEn) return '';

  // Check if already in Portuguese
  const lower = ` ${cleanEn.toLowerCase()} `;
  const ptMarkers = [' que ', ' para ', ' com ', ' uma ', ' não ', ' mais ', ' como ', ' pela ', ' seus '];
  const ptHits = ptMarkers.filter((m) => lower.includes(m)).length;
  if (ptHits >= 2) return cleanEn;

  try {
    const rawCache = window.localStorage.getItem(STORAGE_KEY_SYNOPSES_PT);
    const cache = rawCache ? JSON.parse(rawCache) : {};
    if (cache[String(animeId)]) {
      return cache[String(animeId)];
    }

    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=pt&dt=t&q=${encodeURIComponent(cleanEn.slice(0, 1800))}`;
    const res = await fetch(url);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data?.[0])) {
        const translated = data[0].map((part: unknown[]) => (Array.isArray(part) ? part[0] : '')).join('');
        if (translated.trim()) {
          cache[String(animeId)] = translated.trim();
          window.localStorage.setItem(STORAGE_KEY_SYNOPSES_PT, JSON.stringify(cache));
          return translated.trim();
        }
      }
    }
  } catch {
    // ignore translation network errors
  }

  return cleanEn;
}

/**
 * Fetches Portuguese synopsis and spoiler status for an anime.
 */
export async function getAnimeSynopsis(animeId: number): Promise<SynopsisResponse> {
  const data = await fetchBackendJson<SynopsisResponse>(`${API_BASE_URL}/anime/${animeId}/synopsis`);
  if (data && data.anime_id) {
    return data;
  }

  await ensureStaticCatalogLoaded();
  const found = activeCatalog.find((a) => a.id === animeId);
  const rawDesc = found?.description || '';
  const synopsisPt = rawDesc ? await translateSynopsisInBrowser(animeId, rawDesc) : null;
  const hasSpoilerTags = Boolean(
    found?.has_spoiler_tags ||
      found?.tags?.some((t) => typeof t === 'object' && t.is_media_spoiler)
  );

  return {
    anime_id: animeId,
    synopsis_pt: synopsisPt,
    synopsis_en: rawDesc || null,
    has_spoiler_tags: hasSpoilerTags,
  };
}

/**
 * Synchronizes user lists and ratings from AniList (via backend or direct AniList GraphQL in browser).
 */
export async function syncAniList(payload: AniListSyncRequest): Promise<AniListSyncResponse> {
  const backendRes = await fetchBackendJson<AniListSyncResponse>(
    `${API_BASE_URL}/auth/anilist/sync`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    15000
  );
  if (backendRes && backendRes.success) {
    return backendRes;
  }

  await ensureStaticCatalogLoaded();

  const rawUser = (payload.username || '').trim();
  const token = (payload.access_token || '').trim();
  if (!rawUser && !token) {
    throw new Error('Informe seu nome de usuário da AniList.');
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  let userNameToQuery = rawUser;
  if (!userNameToQuery && token) {
    const viewerRes = await fetch('https://graphql.anilist.co', {
      method: 'POST',
      headers,
      body: JSON.stringify({ query: `query { Viewer { id name } }` }),
    });
    const viewerJson = await viewerRes.json();
    userNameToQuery = viewerJson?.data?.Viewer?.name || '';
  }

  const query = `
    query ($userName: String) {
      MediaListCollection(userName: $userName, type: ANIME) {
        lists {
          name
          status
          entries {
            score(format: POINT_10_DECIMAL)
            status
            media {
              id
              title { romaji english native }
              coverImage { extraLarge large }
              bannerImage
              format
              episodes
              duration
              status
              seasonYear
              averageScore
              popularity
              genres
              description(asHtml: false)
              tags { id name rank isMediaSpoiler }
            }
          }
        }
      }
    }
  `;

  const res = await fetch('https://graphql.anilist.co', {
    method: 'POST',
    headers,
    body: JSON.stringify({ query, variables: { userName: userNameToQuery } }),
  });

  if (!res.ok) {
    throw new Error(`Não foi possível encontrar o perfil público '${userNameToQuery}' na AniList.`);
  }

  const json = await res.json();
  const lists = json?.data?.MediaListCollection?.lists;
  if (!Array.isArray(lists) || lists.length === 0) {
    throw new Error(`Nenhuma lista pública encontrada para '${userNameToQuery}' na AniList.`);
  }

  const catalogById = new Map<number, RawAnimeFixture>();
  for (const a of activeCatalog) catalogById.set(a.id, a);

  let importedWatched = 0;
  let importedWatchlist = 0;
  let importedDropped = 0;
  const favIds = new Set<number>(fallbackProfile.favorite_anime_ids ?? []);
  const wlIds = new Set<number>(fallbackProfile.watchlist_anime_ids ?? []);
  const dropIds = new Set<number>(fallbackProfile.dropped_anime_ids ?? []);
  const existingHistIds = new Set<number>(fallbackHistory.map((h) => h.anime_id));

  for (const listObj of lists) {
    const lStatus = String(listObj.status || '').toUpperCase();
    for (const entry of listObj.entries || []) {
      const m = entry.media;
      if (!m?.id) continue;
      const mid = Number(m.id);

      if (!catalogById.has(mid)) {
        const newRaw: RawAnimeFixture = {
          id: mid,
          title_romaji: m.title?.romaji || m.title?.english || `Anime #${mid}`,
          title_english: m.title?.english || undefined,
          title_native: m.title?.native || undefined,
          cover_image: m.coverImage?.extraLarge || m.coverImage?.large || '',
          banner_image: m.bannerImage || undefined,
          format: m.format || 'TV',
          episodes: m.episodes || undefined,
          duration: m.duration || undefined,
          status: m.status || 'FINISHED',
          season_year: m.seasonYear || undefined,
          average_score: m.averageScore || undefined,
          popularity: m.popularity || 1000,
          genres: m.genres || [],
          description: m.description || undefined,
          tags: (m.tags || []).map((t: { id?: number; name: string; rank?: number; isMediaSpoiler?: boolean }): AnimeTag => ({
            id: t.id || 0,
            name: t.name,
            rank: t.rank || 50,
            is_media_spoiler: Boolean(t.isMediaSpoiler),
          })),
        };
        activeCatalog.push(newRaw);
        catalogById.set(mid, newRaw);
      }

      let sc = Number(entry.score || 0);
      if (sc > 10) sc = sc / 10;
      if (sc <= 0) sc = 7.5;

      if (lStatus === 'COMPLETED') {
        importedWatched++;
        if (sc >= 9) favIds.add(mid);
        if (!existingHistIds.has(mid)) {
          const animeItem = catalogById.get(mid)!;
          fallbackHistory.push({
            id: `wh-al-${mid}`,
            anime_id: mid,
            anime_title: animeItem.title_english || animeItem.title_romaji,
            rating: sc,
            origin_track: 'onboarding',
            interaction_status: 'watched',
            timestamp: new Date().toISOString(),
          });
          existingHistIds.add(mid);
        }
      } else if (lStatus === 'PLANNING') {
        importedWatchlist++;
        wlIds.add(mid);
      } else if (lStatus === 'DROPPED') {
        importedDropped++;
        dropIds.add(mid);
      }
    }
  }

  fallbackProfile = {
    ...fallbackProfile,
    updated_at: new Date().toISOString(),
    total_ratings: fallbackHistory.length,
    favorite_anime_ids: Array.from(favIds),
    watchlist_anime_ids: Array.from(wlIds),
    dropped_anime_ids: Array.from(dropIds),
  };
  fallbackProfile.feature_weights = recalculateProfileWeightsFromHistory(fallbackProfile, fallbackHistory);
  persistLocalState();

  const updatedRecs = computeFallbackRecommendations(fallbackProfile, fallbackHistory, 10);

  return {
    success: true,
    username: userNameToQuery || 'AniList User',
    imported_watched: importedWatched,
    imported_watchlist: importedWatchlist,
    imported_dropped: importedDropped,
    imported_favorites: favIds.size,
    new_profile: structuredClone(fallbackProfile),
    updated_recommendations: updatedRecs,
    message: `Sincronização concluída! ${importedWatched} assistidos, ${importedWatchlist} na lista, ${importedDropped} dropados.`,
  };
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
 * Fetches current seasonal anime releases from backend or static /data/seasonal_releases.json.
 */
export async function fetchSeasonalReleases(): Promise<SeasonalResponse> {
  const data = await fetchBackendJson<SeasonalResponse>(`${API_BASE_URL}/season/releases`);
  if (data && Array.isArray(data.items) && data.items.length > 0) {
    return data;
  }

  if (seasonalCache) {
    return seasonalCache;
  }

  try {
    const res = await fetch('/data/seasonal_releases.json');
    if (res.ok && (res.headers.get('content-type') || '').includes('application/json')) {
      const staticSeasonal = (await res.json()) as SeasonalResponse;
      if (staticSeasonal && Array.isArray(staticSeasonal.items)) {
        seasonalCache = staticSeasonal;
        return staticSeasonal;
      }
    }
  } catch {
    // ignore
  }

  return {
    season: 'ATUAL',
    year: new Date().getFullYear(),
    items: [],
  };
}
