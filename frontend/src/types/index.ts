/**
 * frontend/src/types/index.ts
 * TypeScript Data Transfer Contracts mirrored with Backend Pydantic v2 Schemas.
 */

export interface AnimeTag {
  id: number;
  name: string;
  rank: number;
  is_media_spoiler: boolean;
}

export interface AnimePublic {
  id: number;
  title_romaji: string;
  title_english?: string | null;
  title_native?: string | null;
  cover_image: string;
  banner_image?: string | null;
  format: string;
  episodes?: number | null;
  duration?: number | null;
  status: string;
  season_year?: number | null;
  average_score?: number | null;
  popularity?: number;
  genres: string[];
  tags: string[]; // Strictly non-spoiler public tags
  description?: string | null;
  has_spoiler_tags?: boolean;
  prequel_ids?: number[];
  is_sequel?: boolean;
  compatibility_score?: number | null;
  origin_track?: 'affinity' | 'discovery' | null;
  reason?: string | null;
}

export interface AnimeCatalogResponse {
  items: AnimePublic[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

export interface SemanticTagDelta {
  tag_name: string;
  impact_multiplier: number; // [-1.0, 1.0]
  confidence: number;        // [0.0, 1.0]
  rationale: string;
}

export interface SemanticTuningResponse {
  identified_tags: SemanticTagDelta[];
  public_safe_reason: string;
}

export interface RecommendationPayload {
  affinity_track: AnimePublic[];
  discovery_track: AnimePublic[];
  generated_at: string;
}

export interface UserHyperparameters {
  eta: number;
  alpha: number;
  gamma: number;
  lambda_mmr: number;
}

export interface GoogleUserAccount {
  sub: string;
  email: string;
  name: string;
  picture?: string | null;
  connected_at: string;
  last_synced_at: string;
}

export interface UserProfileData {
  user_id: string;
  updated_at: string;
  total_ratings: number;
  feature_weights: Record<string, number>;
  hyperparameters: UserHyperparameters;
  skipped_anime_ids?: number[];
  dropped_anime_ids?: number[];
  favorite_anime_ids?: number[];
  watchlist_anime_ids?: number[];
  excluded_formats?: string[];
  blocked_genres?: string[];
  google_account?: GoogleUserAccount | null;
}

export interface WatchHistoryEntry {
  id: string;
  anime_id: number;
  anime_title: string;
  rating: number;
  origin_track: 'affinity' | 'discovery' | 'direct_search' | 'onboarding';
  interaction_status?: 'watched' | 'skipped' | 'dropped' | 'watchlist';
  is_favorite?: boolean;
  drop_reason?: string | null;
  user_review_raw?: string | null;
  qualitative_tuning?: SemanticTuningResponse | null;
  timestamp: string;
}

export interface RatingSubmission {
  anime_id: number;
  rating: number; // [1.0, 10.0]
  origin_track: 'affinity' | 'discovery' | 'direct_search' | 'onboarding';
  user_review?: string;
}

export interface RateResponse {
  success: boolean;
  new_profile: UserProfileData;
  updated_recommendations?: RecommendationPayload | null;
  history_entry?: WatchHistoryEntry | null;
  message?: string;
}

export interface TuneSubmission {
  anime_id: number;
  user_feedback: string; // max 300 chars
}

export interface TuneResponse {
  success: boolean;
  tuning_response: SemanticTuningResponse;
  new_profile: UserProfileData;
  history_entry?: WatchHistoryEntry | null;
}

export interface SkipResponse {
  success: boolean;
  skipped_anime_ids: number[];
  skipped_animes?: AnimePublic[];
  new_profile: UserProfileData;
  updated_recommendations?: RecommendationPayload | null;
  message?: string;
}

export interface DropSubmission {
  anime_id: number;
  drop_reason: string;
  origin_track?: 'affinity' | 'discovery' | 'direct_search' | 'onboarding';
}

export interface DropResponse {
  success: boolean;
  tuning_response: SemanticTuningResponse;
  new_profile: UserProfileData;
  updated_recommendations?: RecommendationPayload | null;
  history_entry?: WatchHistoryEntry | null;
  message?: string;
}

export interface ToggleListResponse {
  success: boolean;
  active: boolean;
  anime_ids: number[];
  new_profile: UserProfileData;
}

export interface HistoryResponse {
  items: WatchHistoryEntry[];
  skipped_anime_ids: number[];
  dropped_anime_ids: number[];
  favorite_anime_ids: number[];
  watchlist_anime_ids: number[];
  skipped_animes?: AnimePublic[];
  dropped_animes?: AnimePublic[];
  favorite_animes?: AnimePublic[];
  watchlist_animes?: AnimePublic[];
}

export interface FilterPreferencesPayload {
  excluded_formats?: string[];
  blocked_genres?: string[];
}

export interface FilterPreferencesResponse {
  success: boolean;
  new_profile: UserProfileData;
  updated_recommendations: RecommendationPayload;
  message?: string;
}

export interface GoogleAuthPayload {
  credential?: string;
  email?: string;
  name?: string;
  picture?: string;
  sub?: string;
}

export interface GoogleAuthResponse {
  success: boolean;
  new_profile: UserProfileData;
  watch_history: WatchHistoryEntry[];
  updated_recommendations?: RecommendationPayload | null;
  message?: string;
}

export interface HealthResponse {
  status: string;
  catalog_size: number;
  cache_freshness: string;
  gemini_status: string;
  version: string;
}

export interface SynopsisResponse {
  anime_id: number;
  synopsis_pt?: string | null;
  synopsis_en?: string | null;
  has_spoiler_tags: boolean;
}

export interface AniListSyncRequest {
  username?: string;
  access_token?: string;
}

export interface AniListSyncResponse {
  success: boolean;
  username: string;
  imported_watched: number;
  imported_watchlist: number;
  imported_dropped: number;
  imported_favorites: number;
  new_profile: UserProfileData;
  updated_recommendations?: RecommendationPayload | null;
  message: string;
}
