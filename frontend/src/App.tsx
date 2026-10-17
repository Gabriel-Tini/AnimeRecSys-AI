import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Bell,
  Flame,
  Play,
  Bookmark,
  Heart,
  Compass,
  EyeOff,
  Ban,
  CheckCircle2,
  RotateCcw,
  Sparkles,
  Filter,
  ArrowUpDown,
  Search,
  Tv,
  Film,
  Cloud,
  X,
  SlidersHorizontal,
  Eye,
  RefreshCw,
  Star,
  Edit3,
  Trash2,
} from 'lucide-react';
import { Navbar } from './components/Navbar';
import { SearchBar } from './components/SearchBar';
import { OnboardingShelf } from './components/OnboardingShelf';
import { TrackAffinity } from './components/TrackAffinity';
import { TrackDiscovery } from './components/TrackDiscovery';
import { QualitativeModal } from './components/QualitativeModal';
import { ProfileMetrics } from './components/ProfileMetrics';
import { AnimeCard } from './components/AnimeCard';
import { AnimeDetailModal } from './components/AnimeDetailModal';
import { SeasonalDrawer } from './components/SeasonalDrawer';
import {
  cancelRating,
  checkBackendHealth,
  dropAnime,
  editRating,
  fetchAllCatalogPages,
  fetchCatalog,
  fetchRecommendations,
  fetchUserProfile,
  fetchWatchHistory,
  googleLogout,
  googleSignIn,
  isAnimeEligibleForRecommendation,
  isReleasedAnimeWork,
  resetUserProfile,
  skipAnime,
  submitRating,
  submitSemanticTuning,
  syncAniList,
  toggleFavorite,
  toggleWatchlist,
  updateRecommendationFilters,
} from './services/api';
import {
  signInWithRealGooglePopup,
  signOutFromRealGoogle,
  subscribeToGoogleAuthChanges,
} from './services/firebaseAuth';
import type {
  AnimePublic,
  RecommendationPayload,
  SemanticTuningResponse,
  UserProfileData,
  WatchHistoryEntry,
} from './types';
import { handleCoverError, handleBannerError } from './utils/imageFallback';

const SPOILER_KEYWORDS = new Set([
  'plot twist',
  'betrayal',
  'tragedy',
  'conspiracy',
  'memory manipulation',
]);

const EXPLORE_GENRES = [
  'Action',
  'Adventure',
  'Comedy',
  'Drama',
  'Fantasy',
  'Sci-Fi',
  'Romance',
  'Suspense',
  'Mystery',
  'Psychological',
  'Supernatural',
  'Slice of Life',
  'Horror',
  'Mecha',
  'Sports',
  'Music',
  'Ecchi',
  'Mahou Shoujo',
];

const BLOCKABLE_GENRES: { value: string; label: string }[] = [
  { value: 'Mecha', label: 'Mecha (Robôs)' },
  { value: 'Horror', label: 'Terror (Horror)' },
  { value: 'Romance', label: 'Romance' },
  { value: 'Ecchi', label: 'Ecchi' },
  { value: 'Sports', label: 'Esportes (Sports)' },
  { value: 'Slice of Life', label: 'Slice of Life' },
  { value: 'Music', label: 'Musical (Music)' },
  { value: 'Mahou Shoujo', label: 'Mahou Shoujo' },
  { value: 'Comedy', label: 'Comédia' },
  { value: 'Drama', label: 'Drama' },
  { value: 'Sci-Fi', label: 'Sci-Fi' },
  { value: 'Fantasy', label: 'Fantasia' },
  { value: 'Action', label: 'Ação' },
  { value: 'Adventure', label: 'Aventura' },
  { value: 'Mystery', label: 'Mistério' },
  { value: 'Psychological', label: 'Psicológico' },
  { value: 'Supernatural', label: 'Sobrenatural' },
  { value: 'Thriller', label: 'Suspense / Thriller' },
];

const ALL_FORMATS: { value: string; label: string }[] = [
  { value: 'TV', label: 'Séries de TV / Seasons' },
  { value: 'TV_SHORT', label: 'TV Curto' },
  { value: 'ONA', label: 'ONA (Web Series)' },
  { value: 'MOVIE', label: 'Filmes (Movie)' },
  { value: 'OVA', label: 'OVA' },
  { value: 'SPECIAL', label: 'Especiais (Special)' },
];

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [listSubTab, setListSubTab] = useState<'watched' | 'watchlist' | 'dropped' | 'skipped'>('watched');
  const [isSeasonalDrawerOpen, setIsSeasonalDrawerOpen] = useState(false);
  const [isOnline, setIsOnline] = useState(false);
  const [fullCatalog, setFullCatalog] = useState<AnimePublic[]>([]);
  const [catalog, setCatalog] = useState<AnimePublic[]>([]);
  const [knownAnimesMap, setKnownAnimesMap] = useState<Record<number, AnimePublic>>({});
  const [recommendations, setRecommendations] = useState<RecommendationPayload>({
    affinity_track: [],
    discovery_track: [],
    generated_at: new Date().toISOString(),
  });
  const [profile, setProfile] = useState<UserProfileData>({
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
    google_account: null,
  });
  const [watchHistory, setWatchHistory] = useState<WatchHistoryEntry[]>([]);
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);
  const [ratedMap, setRatedMap] = useState<Record<number, number>>({});
  const [lastSafeReason, setLastSafeReason] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [showFilterPanel, setShowFilterPanel] = useState<boolean>(true);
  const [confirmResetOpen, setConfirmResetOpen] = useState<boolean>(false);

  // Real Google OAuth State
  const [isSyncingGoogle, setIsSyncingGoogle] = useState(false);

  // Edit Rating State (Minha Lista)
  const [editingHistoryEntry, setEditingHistoryEntry] = useState<WatchHistoryEntry | null>(null);
  const [editRatingValue, setEditRatingValue] = useState<number>(8);
  const [editReviewValue, setEditReviewValue] = useState<string>('');
  const [isSavingEditRating, setIsSavingEditRating] = useState<boolean>(false);

  // AniList Sync State
  const [anilistModalOpen, setAnilistModalOpen] = useState(false);
  const [anilistUsernameInput, setAnilistUsernameInput] = useState('');
  const [anilistTokenInput, setAnilistTokenInput] = useState('');
  const [isSyncingAniList, setIsSyncingAniList] = useState(false);
  const [anilistSyncSummary, setAnilistSyncSummary] = useState<{
    username: string;
    watched: number;
    watchlist: number;
    dropped: number;
    favorites: number;
  } | null>(null);

  // Explore Tab Filters & Infinite Scroll
  const [exploreQuery, setExploreQuery] = useState('');
  const [exploreGenre, setExploreGenre] = useState<string>('ALL');
  const [exploreFormat, setExploreFormat] = useState<string>('ALL');
  const [exploreSort, setExploreSort] = useState<'score' | 'popularity' | 'year' | 'title'>('popularity');
  const [hideInteractedInExplore, setHideInteractedInExplore] = useState(false);
  const [exploreVisibleCount, setExploreVisibleCount] = useState<number>(24);
  const exploreSentinelRef = useRef<HTMLDivElement | null>(null);

  // Qualitative / Drop Modal State
  const [modalOpen, setModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<'like' | 'drop'>('like');
  const [modalAnime, setModalAnime] = useState<AnimePublic | null>(null);
  const [modalRating, setModalRating] = useState<number>(9);
  const [modalOriginTrack, setModalOriginTrack] = useState<'affinity' | 'discovery' | 'onboarding' | 'direct_search'>('onboarding');

  // Anime Detail Modal State
  const [selectedDetailAnime, setSelectedDetailAnime] = useState<AnimePublic | null>(null);
  const [isTuning, setIsTuning] = useState(false);
  const [tuningResult, setTuningResult] = useState<SemanticTuningResponse | null>(null);

  const showToast = useCallback((msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage((prev) => (prev === msg ? null : prev));
    }, 3500);
  }, []);

  const registerKnownAnimes = useCallback((animes: (AnimePublic | undefined | null)[]) => {
    setKnownAnimesMap((prev) => {
      const next = { ...prev };
      for (const a of animes) {
        if (a && a.id) {
          next[a.id] = a;
        }
      }
      return next;
    });
  }, []);

  const loadInitialData = useCallback(async () => {
    try {
      const [health, allCatItems, recRes, profRes, histRes] = await Promise.all([
        checkBackendHealth(),
        fetchAllCatalogPages(),
        fetchRecommendations(10),
        fetchUserProfile(),
        fetchWatchHistory(),
      ]);
      setIsOnline(health.online);
      setFullCatalog(allCatItems || []);
      setCatalog(allCatItems || []);
      if (recRes) setRecommendations(recRes);
      if (profRes) setProfile(profRes);
      if (histRes?.items) setWatchHistory(histRes.items);

      registerKnownAnimes([
        ...(allCatItems || []),
        ...(recRes?.affinity_track ?? []),
        ...(recRes?.discovery_track ?? []),
        ...(histRes?.skipped_animes ?? []),
        ...(histRes?.dropped_animes ?? []),
        ...(histRes?.favorite_animes ?? []),
        ...(histRes?.watchlist_animes ?? []),
      ]);

      const initialRatings: Record<number, number> = {};
      for (const item of histRes?.items ?? []) {
        if (item.interaction_status !== 'dropped') {
          initialRatings[item.anime_id] = item.rating;
        }
      }
      setRatedMap(initialRatings);
    } catch (err) {
      console.error('Falha ao carregar dados iniciais no F5:', err);
    }
  }, [registerKnownAnimes]);

  useEffect(() => {
    void loadInitialData();
  }, [loadInitialData]);

  // Filter onboarding catalog by genre when selected
  useEffect(() => {
    const loadFilteredCatalog = async () => {
      if (!selectedGenre) {
        setCatalog(fullCatalog);
        return;
      }
      const catRes = await fetchCatalog({
        genre: selectedGenre,
        limit: 100,
      });
      setCatalog(catRes.items);
      registerKnownAnimes(catRes.items);
    };
    void loadFilteredCatalog();
  }, [selectedGenre, fullCatalog, registerKnownAnimes]);

  const skippedIds = useMemo(() => new Set(profile.skipped_anime_ids ?? []), [profile.skipped_anime_ids]);
  const droppedIds = useMemo(() => new Set(profile.dropped_anime_ids ?? []), [profile.dropped_anime_ids]);
  const favoriteIds = useMemo(() => new Set(profile.favorite_anime_ids ?? []), [profile.favorite_anime_ids]);
  const watchlistIds = useMemo(() => new Set(profile.watchlist_anime_ids ?? []), [profile.watchlist_anime_ids]);
  const excludedFormatsSet = useMemo(
    () => new Set((profile.excluded_formats ?? []).map((f) => f.trim().toUpperCase())),
    [profile.excluded_formats]
  );
  const blockedGenresSet = useMemo(
    () => new Set((profile.blocked_genres ?? []).map((g) => g.trim().toLowerCase())),
    [profile.blocked_genres]
  );

  const handleRateAnime = async (
    anime: AnimePublic,
    rating: number,
    track: 'affinity' | 'discovery' | 'onboarding' | 'direct_search' = 'onboarding'
  ) => {
    // Guard: Prevent duplicate evaluations
    if (ratedMap[anime.id] !== undefined) {
      showToast(
        `A obra "${anime.title_english || anime.title_romaji}" já foi avaliada (${ratedMap[anime.id]}/10). Para alterar a nota sem distorcer o algoritmo, utilize "Editar Nota" em Minha Lista.`
      );
      return;
    }

    registerKnownAnimes([anime]);
    setRatedMap((prev) => ({ ...prev, [anime.id]: rating }));
    const res = await submitRating({
      anime_id: anime.id,
      rating,
      origin_track: track,
    });
    setProfile(res.new_profile);
    if (res.history_entry) {
      setWatchHistory((prev) => [
        ...prev.filter((h) => h.anime_id !== res.history_entry!.anime_id),
        res.history_entry!,
      ]);
    }
    if (res.updated_recommendations) {
      setRecommendations(res.updated_recommendations);
      registerKnownAnimes([
        ...res.updated_recommendations.affinity_track,
        ...res.updated_recommendations.discovery_track,
      ]);
    } else {
      const freshRecs = await fetchRecommendations(10);
      setRecommendations(freshRecs);
      registerKnownAnimes([...freshRecs.affinity_track, ...freshRecs.discovery_track]);
    }

    // Open Qualitative Gemini Modal immediately after rating
    setModalMode('like');
    setModalAnime(anime);
    setModalRating(rating);
    setModalOriginTrack(track);
    setTuningResult(null);
    setModalOpen(true);
  };

  const handleOpenDropModal = (
    anime: AnimePublic,
    track: 'affinity' | 'discovery' | 'onboarding' | 'direct_search' = 'onboarding'
  ) => {
    registerKnownAnimes([anime]);
    setModalMode('drop');
    setModalAnime(anime);
    setModalOriginTrack(track);
    setTuningResult(null);
    setModalOpen(true);
  };

  const handleSkipAnime = async (anime: AnimePublic, unskip = false) => {
    registerKnownAnimes([anime]);
    const res = await skipAnime(anime.id, unskip);
    setProfile(res.new_profile);
    if (res.skipped_animes) {
      registerKnownAnimes(res.skipped_animes);
    }
    if (res.updated_recommendations) {
      setRecommendations(res.updated_recommendations);
      registerKnownAnimes([
        ...res.updated_recommendations.affinity_track,
        ...res.updated_recommendations.discovery_track,
      ]);
    } else {
      const freshRecs = await fetchRecommendations(10);
      setRecommendations(freshRecs);
      registerKnownAnimes([...freshRecs.affinity_track, ...freshRecs.discovery_track]);
    }
    const title = anime.title_english || anime.title_romaji;
    showToast(
      unskip
        ? `"${title}" restaurado para as recomendações.`
        : `"${title}" pulado e movido para Minha Lista → Pulados.`
    );
  };

  const handleRestoreAllSkipped = async () => {
    const ids = [...(profile.skipped_anime_ids ?? [])];
    for (const id of ids) {
      const res = await skipAnime(id, true);
      setProfile(res.new_profile);
      if (res.updated_recommendations) {
        setRecommendations(res.updated_recommendations);
      }
    }
    showToast('Todos os animes pulados foram restaurados para as recomendações!');
  };

  const handleToggleFavorite = async (anime: AnimePublic) => {
    registerKnownAnimes([anime]);
    const res = await toggleFavorite(anime.id);
    setProfile(res.new_profile);
    const title = anime.title_english || anime.title_romaji;
    showToast(res.active ? `"${title}" adicionado aos Favoritos!` : `"${title}" removido dos Favoritos.`);
  };

  const handleToggleWatchlist = async (anime: AnimePublic) => {
    registerKnownAnimes([anime]);
    const res = await toggleWatchlist(anime.id);
    setProfile(res.new_profile);
    const title = anime.title_english || anime.title_romaji;
    showToast(
      res.active
        ? `"${title}" salvo em Minha Lista (Quero Assistir).`
        : `"${title}" removido de Minha Lista.`
    );
  };

  // Format & Genre Blocking Filter Handlers
  const handleToggleExcludedFormat = async (formatCode: string) => {
    const upper = formatCode.trim().toUpperCase();
    const current = new Set((profile.excluded_formats ?? []).map((f) => f.trim().toUpperCase()));
    if (current.has(upper)) {
      current.delete(upper);
    } else {
      current.add(upper);
    }
    const nextFormats = Array.from(current);
    const res = await updateRecommendationFilters({
      excluded_formats: nextFormats,
      blocked_genres: profile.blocked_genres ?? [],
    });
    setProfile(res.new_profile);
    setRecommendations(res.updated_recommendations);
    registerKnownAnimes([
      ...res.updated_recommendations.affinity_track,
      ...res.updated_recommendations.discovery_track,
    ]);
    showToast('Filtro de formatos atualizado nas recomendações!');
  };

  const handleSetOnlyTvSeriesPreset = async () => {
    const nonTvFormats = ['MOVIE', 'OVA', 'SPECIAL'];
    const isAlreadyTvOnly = nonTvFormats.every((f) => excludedFormatsSet.has(f));
    const nextFormats = isAlreadyTvOnly ? [] : nonTvFormats;
    const res = await updateRecommendationFilters({
      excluded_formats: nextFormats,
      blocked_genres: profile.blocked_genres ?? [],
    });
    setProfile(res.new_profile);
    setRecommendations(res.updated_recommendations);
    registerKnownAnimes([
      ...res.updated_recommendations.affinity_track,
      ...res.updated_recommendations.discovery_track,
    ]);
    showToast(
      isAlreadyTvOnly
        ? 'Todos os formatos (Filmes, OVAs, Especiais e TV) liberados!'
        : 'Modo "Apenas Séries / TV" ativado! Filmes, OVAs e Especiais ocultados das recomendações.'
    );
  };

  const handleToggleBlockedGenre = async (genreName: string) => {
    const current = [...(profile.blocked_genres ?? [])];
    const existsIdx = current.findIndex((g) => g.trim().toLowerCase() === genreName.trim().toLowerCase());
    let nextGenres: string[];
    let blockedNow = false;
    if (existsIdx >= 0) {
      nextGenres = current.filter((_, idx) => idx !== existsIdx);
      blockedNow = false;
    } else {
      nextGenres = [...current, genreName];
      blockedNow = true;
    }
    const res = await updateRecommendationFilters({
      excluded_formats: profile.excluded_formats ?? [],
      blocked_genres: nextGenres,
    });
    setProfile(res.new_profile);
    setRecommendations(res.updated_recommendations);
    registerKnownAnimes([
      ...res.updated_recommendations.affinity_track,
      ...res.updated_recommendations.discovery_track,
    ]);
    showToast(
      blockedNow
        ? `Gênero "${genreName}" bloqueado! O algoritmo não recomendará obras desse gênero.`
        : `Gênero "${genreName}" desbloqueado nas recomendações.`
    );
  };

  const handleClearAllFilters = async () => {
    const res = await updateRecommendationFilters({
      excluded_formats: [],
      blocked_genres: [],
    });
    setProfile(res.new_profile);
    setRecommendations(res.updated_recommendations);
    showToast('Todos os bloqueios de formato e gênero foram removidos.');
  };

  // Real Google OAuth Popup Login & Automatic Sync Handlers
  const handleRealGoogleLogin = async () => {
    setIsSyncingGoogle(true);
    try {
      const googleUser = await signInWithRealGooglePopup();
      const res = await googleSignIn({
        sub: googleUser.uid,
        email: googleUser.email,
        name: googleUser.displayName,
        picture: googleUser.photoURL ?? undefined,
      });

      setProfile(res.new_profile);
      if (res.watch_history) {
        setWatchHistory(res.watch_history);
        const updatedRatings: Record<number, number> = {};
        for (const item of res.watch_history) {
          if (item.interaction_status !== 'dropped') {
            updatedRatings[item.anime_id] = item.rating;
          }
        }
        setRatedMap(updatedRatings);
      }
      if (res.updated_recommendations) {
        setRecommendations(res.updated_recommendations);
      }
      showToast(
        `Conta Google (${googleUser.email}) vinculada! Sincronização automática ativada.`
      );
    } catch (err: unknown) {
      const code = (err as { code?: string })?.code || '';
      if (code === 'auth/popup-closed-by-user' || code === 'auth/cancelled-popup-request') {
        return;
      }
      showToast(
        'Não foi possível concluir o login com o Google. Verifique se pop-ups estão permitidos no navegador.'
      );
    } finally {
      setIsSyncingGoogle(false);
    }
  };

  useEffect(() => {
    const unsubscribe = subscribeToGoogleAuthChanges((googleUser) => {
      if (!googleUser) return;
      void googleSignIn({
        sub: googleUser.uid,
        email: googleUser.email,
        name: googleUser.displayName,
        picture: googleUser.photoURL ?? undefined,
      }).then((res) => {
        setProfile(res.new_profile);
        if (res.watch_history) {
          setWatchHistory(res.watch_history);
          const updatedRatings: Record<number, number> = {};
          for (const item of res.watch_history) {
            if (item.interaction_status !== 'dropped') {
              updatedRatings[item.anime_id] = item.rating;
            }
          }
          setRatedMap(updatedRatings);
        }
        if (res.updated_recommendations) {
          setRecommendations(res.updated_recommendations);
        }
      });
    });
    return () => unsubscribe();
  }, []);

  const handleGoogleSignOut = async () => {
    await signOutFromRealGoogle();
    const res = await googleLogout();
    setProfile(res.new_profile);
    showToast('Conta Google desconectada. Seu backup permanece salvo.');
  };

  const handleAniListSyncSubmit = async (username: string, token?: string) => {
    if (!username.trim() && !token?.trim()) {
      showToast('Por favor, informe seu nome de usuário da AniList.');
      return;
    }
    setIsSyncingAniList(true);
    try {
      const res = await syncAniList({
        username: username.trim() || undefined,
        access_token: token?.trim() || undefined,
      });

      setProfile(res.new_profile);
      if (res.updated_recommendations) {
        setRecommendations(res.updated_recommendations);
        registerKnownAnimes([
          ...res.updated_recommendations.affinity_track,
          ...res.updated_recommendations.discovery_track,
        ]);
      }

      setAnilistSyncSummary({
        username: res.username,
        watched: res.imported_watched,
        watchlist: res.imported_watchlist,
        dropped: res.imported_dropped,
        favorites: res.imported_favorites,
      });

      // Reload watch history
      const hist = await fetchWatchHistory();
      setWatchHistory(hist.items);
      const updatedRatings: Record<number, number> = {};
      for (const item of hist.items) {
        if (item.interaction_status !== 'dropped') {
          updatedRatings[item.anime_id] = item.rating;
        }
      }
      setRatedMap(updatedRatings);

      // Reload full catalog to register newly imported animes
      const cat = await fetchAllCatalogPages();
      setFullCatalog(cat);
      setCatalog(cat);
      registerKnownAnimes(cat);

      setAnilistModalOpen(false);
      showToast(res.message);
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : 'Falha na sincronização com AniList.');
    } finally {
      setIsSyncingAniList(false);
    }
  };

  const handleSubmitModalFeedback = async (animeId: number, feedback: string) => {
    setIsTuning(true);
    try {
      if (modalMode === 'drop') {
        const dropRes = await dropAnime({
          anime_id: animeId,
          drop_reason: feedback,
          origin_track: modalOriginTrack,
        });
        setTuningResult(dropRes.tuning_response);
        setLastSafeReason(dropRes.tuning_response.public_safe_reason);
        setProfile(dropRes.new_profile);
        if (dropRes.history_entry) {
          setWatchHistory((prev) => [
            ...prev.filter((h) => h.anime_id !== dropRes.history_entry!.anime_id),
            dropRes.history_entry!,
          ]);
        }
        if (dropRes.updated_recommendations) {
          setRecommendations(dropRes.updated_recommendations);
          registerKnownAnimes([
            ...dropRes.updated_recommendations.affinity_track,
            ...dropRes.updated_recommendations.discovery_track,
          ]);
        } else {
          const refreshedRecs = await fetchRecommendations(10);
          setRecommendations(refreshedRecs);
        }
        showToast('Drop analisado pelo nosso algoritmo! Elementos negativos penalizados.');
      } else {
        const res = await submitSemanticTuning({
          anime_id: animeId,
          user_feedback: feedback,
        });
        setTuningResult(res.tuning_response);
        setLastSafeReason(res.tuning_response.public_safe_reason);
        setProfile(res.new_profile);
        if (res.history_entry) {
          setWatchHistory((prev) =>
            prev.map((h) => (h.id === res.history_entry!.id ? res.history_entry! : h))
          );
        }
        const refreshedRecs = await fetchRecommendations(10);
        setRecommendations(refreshedRecs);
        showToast('Preferências atualizadas pelo nosso algoritmo!');
      }
    } finally {
      setIsTuning(false);
    }
  };

  const handleSaveEditRating = async () => {
    if (!editingHistoryEntry) return;
    setIsSavingEditRating(true);
    try {
      const res = await editRating({
        anime_id: editingHistoryEntry.anime_id,
        rating: editRatingValue,
        user_review: editReviewValue.trim() || undefined,
      });

      setProfile(res.new_profile);
      setRatedMap((prev) => ({ ...prev, [editingHistoryEntry.anime_id]: editRatingValue }));
      if (res.updated_recommendations) {
        setRecommendations(res.updated_recommendations);
        registerKnownAnimes([
          ...res.updated_recommendations.affinity_track,
          ...res.updated_recommendations.discovery_track,
        ]);
      }
      if (res.history_entry) {
        setWatchHistory((prev) =>
          prev.map((h) => (h.anime_id === editingHistoryEntry.anime_id ? res.history_entry! : h))
        );
      }
      setEditingHistoryEntry(null);
      showToast(res.message || 'Avaliação atualizada e pesos recalibrados com sucesso!');
    } catch {
      showToast('Falha ao atualizar avaliação.');
    } finally {
      setIsSavingEditRating(false);
    }
  };

  const handleCancelRating = async (animeId: number) => {
    try {
      const res = await cancelRating(animeId);
      setProfile(res.new_profile);
      setRatedMap((prev) => {
        const next = { ...prev };
        delete next[animeId];
        return next;
      });
      setWatchHistory((prev) => prev.filter((h) => h.anime_id !== animeId));
      if (res.updated_recommendations) {
        setRecommendations(res.updated_recommendations);
        registerKnownAnimes([
          ...res.updated_recommendations.affinity_track,
          ...res.updated_recommendations.discovery_track,
        ]);
      } else {
        const freshRecs = await fetchRecommendations(10);
        setRecommendations(freshRecs);
      }
      if (editingHistoryEntry?.anime_id === animeId) {
        setEditingHistoryEntry(null);
      }
      showToast('Avaliação cancelada! A nota foi removida e os pesos do algoritmo foram restaurados.');
    } catch {
      showToast('Não foi possível cancelar a avaliação.');
    }
  };

  const handleResetProfile = async () => {
    const freshProfile = await resetUserProfile();
    setProfile(freshProfile);
    setRatedMap({});
    setWatchHistory([]);
    setLastSafeReason(null);
    setConfirmResetOpen(false);
    const freshRecs = await fetchRecommendations(10);
    setRecommendations(freshRecs);
    showToast('Preferências e histórico resetados para o estado inicial.');
  };

  const positivelyRatedIds = useMemo(
    () =>
      new Set<number>(
        Object.entries(ratedMap)
          .filter(([, score]) => Number(score) >= 5.5)
          .map(([id]) => Number(id))
      ),
    [ratedMap]
  );

  // Visible Onboarding Catalog (excludes evaluated, skipped, dropped, unrated sequels, unreleased, MVs, blocked formats & blocked genres)
  const visibleOnboardingCatalog = useMemo(
    () =>
      catalog.filter((a) => {
        if (ratedMap[a.id] !== undefined) return false;
        if (skippedIds.has(a.id) || droppedIds.has(a.id)) return false;
        if (!isAnimeEligibleForRecommendation(a, positivelyRatedIds)) return false;
        if (
          excludedFormatsSet.size > 0 &&
          excludedFormatsSet.has((a.format || 'TV').trim().toUpperCase())
        ) {
          return false;
        }
        if (
          blockedGenresSet.size > 0 &&
          a.genres.some((g) => blockedGenresSet.has(g.trim().toLowerCase()))
        ) {
          return false;
        }
        return true;
      }),
    [catalog, ratedMap, skippedIds, droppedIds, positivelyRatedIds, excludedFormatsSet, blockedGenresSet]
  );

  // Filtered & Sorted Catalog for Explorar Tab
  const exploreAnimes = useMemo(() => {
    let list = fullCatalog.filter(isReleasedAnimeWork);
    if (exploreQuery.trim()) {
      const q = exploreQuery.trim().toLowerCase();
      list = list.filter(
        (a) =>
          a.title_romaji.toLowerCase().includes(q) ||
          (a.title_english && a.title_english.toLowerCase().includes(q)) ||
          a.genres.some((g) => g.toLowerCase().includes(q)) ||
          a.tags.some((t) => t.toLowerCase().includes(q))
      );
    } else {
      // When browsing without a specific search query, hide subsequent seasons whose previous season hasn't been evaluated yet
      list = list.filter((a) => isAnimeEligibleForRecommendation(a, positivelyRatedIds));
    }
    if (exploreGenre !== 'ALL') {
      list = list.filter((a) =>
        a.genres.some((g) => g.toLowerCase() === exploreGenre.toLowerCase())
      );
    }
    if (exploreFormat !== 'ALL') {
      list = list.filter((a) => a.format.toUpperCase() === exploreFormat);
    }
    if (!exploreQuery.trim() && excludedFormatsSet.size > 0) {
      list = list.filter((a) => !excludedFormatsSet.has((a.format || 'TV').trim().toUpperCase()));
    }
    if (!exploreQuery.trim() && blockedGenresSet.size > 0) {
      list = list.filter((a) => !a.genres.some((g) => blockedGenresSet.has(g.trim().toLowerCase())));
    }
    if (hideInteractedInExplore) {
      list = list.filter(
        (a) =>
          ratedMap[a.id] == null &&
          !skippedIds.has(a.id) &&
          !droppedIds.has(a.id)
      );
    }
    list.sort((a, b) => {
      if (exploreSort === 'score') return (b.average_score ?? 0) - (a.average_score ?? 0);
      if (exploreSort === 'year') return (b.season_year ?? 0) - (a.season_year ?? 0);
      if (exploreSort === 'title') {
        return (a.title_english || a.title_romaji).localeCompare(b.title_english || b.title_romaji);
      }
      return (b.popularity ?? 0) - (a.popularity ?? 0);
    });
    return list;
  }, [
    fullCatalog,
    exploreQuery,
    exploreGenre,
    exploreFormat,
    exploreSort,
    hideInteractedInExplore,
    ratedMap,
    skippedIds,
    droppedIds,
    positivelyRatedIds,
    excludedFormatsSet,
    blockedGenresSet,
  ]);

  // Reset Infinite Scroll batch count whenever Explore filters change
  useEffect(() => {
    setExploreVisibleCount(24);
  }, [exploreQuery, exploreGenre, exploreFormat, exploreSort, hideInteractedInExplore, activeTab]);

  // Infinite Scroll IntersectionObserver for Explorar Tab
  useEffect(() => {
    if (activeTab !== 'explore') return;
    if (exploreVisibleCount >= exploreAnimes.length) return;
    const node = exploreSentinelRef.current;
    if (!node || typeof IntersectionObserver === 'undefined') return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setExploreVisibleCount((prev) => Math.min(prev + 24, exploreAnimes.length));
        }
      },
      { rootMargin: '320px' }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [activeTab, exploreVisibleCount, exploreAnimes.length]);

  // Helper to resolve any anime ID from knownAnimesMap or fullCatalog with guaranteed fallback
  const resolveAnimeById = useCallback(
    (id: number): AnimePublic => {
      if (knownAnimesMap[id]) return knownAnimesMap[id];
      const fromCat = fullCatalog.find((a) => a.id === id);
      if (fromCat) return fromCat;
      return {
        id,
        title_romaji: `Anime #${id}`,
        title_english: `Anime #${id}`,
        cover_image: 'https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/default.jpg',
        format: 'TV',
        status: 'FINISHED',
        genres: [],
        tags: [],
      };
    },
    [knownAnimesMap, fullCatalog]
  );

  // Derived lists for Minha Lista & Favoritos (guaranteed 100% resolution for every ID and zero duplicates)
  const watchedEntries = useMemo(() => {
    const map = new Map<number, typeof watchHistory[0]>();
    for (const h of watchHistory) {
      if (h.interaction_status !== 'dropped') {
        map.set(h.anime_id, h);
      }
    }
    return Array.from(map.values()).reverse();
  }, [watchHistory]);

  const droppedEntries = useMemo(() => {
    const map = new Map<number, typeof watchHistory[0]>();
    for (const h of watchHistory) {
      if (h.interaction_status === 'dropped') {
        map.set(h.anime_id, h);
      }
    }
    return Array.from(map.values()).reverse();
  }, [watchHistory]);
  const watchlistAnimes = useMemo(
    () => (profile.watchlist_anime_ids ?? []).map(resolveAnimeById),
    [profile.watchlist_anime_ids, resolveAnimeById]
  );
  const skippedAnimes = useMemo(
    () => [...(profile.skipped_anime_ids ?? [])].reverse().map(resolveAnimeById),
    [profile.skipped_anime_ids, resolveAnimeById]
  );
  const favoriteAnimes = useMemo(() => {
    const combinedIds = new Set<number>(profile.favorite_anime_ids ?? []);
    for (const [aidStr, score] of Object.entries(ratedMap)) {
      if (score >= 9) combinedIds.add(Number(aidStr));
    }
    return Array.from(combinedIds).map(resolveAnimeById);
  }, [profile.favorite_anime_ids, ratedMap, resolveAnimeById]);

  const isOnlyTvSeriesActive = useMemo(
    () => ['MOVIE', 'OVA', 'SPECIAL'].every((f) => excludedFormatsSet.has(f)),
    [excludedFormatsSet]
  );

  // Hero Spotlight Anime: dynamic rotation over real trending anime
  // Excludes evaluated, skipped, and dropped titles; updates immediately upon evaluation
  const heroAnime = useMemo(() => {
    if (visibleOnboardingCatalog.length === 0) return null;
    // Sort unrated, unskipped, undropped catalog by popularity descending
    const trending = [...visibleOnboardingCatalog].sort(
      (a, b) => (b.popularity || 0) - (a.popularity || 0)
    );
    // Find the highest trending title with a banner image within top 25 for visual appeal, or top overall
    const withBanner = trending.slice(0, 25).find((a) => Boolean(a.banner_image && a.banner_image.trim()));
    return withBanner || trending[0] || null;
  }, [visibleOnboardingCatalog]);

  return (
    <div className="min-h-screen bg-[#0a0c10] text-slate-100 flex">
      {/* Left Sidebar Navigation */}
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isOnline={isOnline}
        totalRatings={profile.total_ratings}
        onResetProfile={() => setConfirmResetOpen(true)}
        googleAccount={profile.google_account}
        onOpenGoogleModal={() => void handleRealGoogleLogin()}
        onGoogleLogout={() => void handleGoogleSignOut()}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar with Raycast-Style Search & Google Auth */}
        <header className="sticky top-0 z-30 bg-[#0a0c10]/85 backdrop-blur-md border-b border-[#232936] px-3 sm:px-6 py-2.5 sm:py-3.5 flex items-center justify-between gap-2.5 sm:gap-4">
          <SearchBar
            onQuickRate={(anime, score) => {
              void handleRateAnime(anime, score, 'direct_search');
            }}
            onSelectAnime={(anime) => setSelectedDetailAnime(anime)}
            ratedMap={ratedMap}
          />

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            {/* Google Sign-In / Automatic Sync Status in Header */}
            {profile.google_account ? (
              <div
                onClick={() => setActiveTab('profile')}
                title="Sincronização automática ativa com sua Conta Google"
                className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1.5 rounded-xl bg-emerald-950/40 hover:bg-emerald-900/50 border border-emerald-500/30 text-xs text-emerald-200 transition-colors cursor-pointer"
              >
                <Cloud className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <span className="hidden sm:inline font-medium">
                  Auto-Sync: {profile.google_account.name}
                </span>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => void handleRealGoogleLogin()}
                disabled={isSyncingGoogle}
                data-testid="header-google-login-btn"
                className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1.5 rounded-xl bg-white hover:bg-slate-100 disabled:opacity-60 text-slate-900 text-xs font-semibold shadow transition-colors"
              >
                <svg className="w-3.5 h-3.5 shrink-0" viewBox="0 0 24 24">
                  <path
                    fill="#4285F4"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  />
                </svg>
                <span className="hidden sm:inline">
                  {isSyncingGoogle ? 'Conectando...' : 'Entrar com Google'}
                </span>
              </button>
            )}

            {/* Novidades e Lançamentos da Temporada */}
            <button
              type="button"
              onClick={() => setIsSeasonalDrawerOpen(true)}
              aria-label="Novidades e Lançamentos da Temporada"
              title="Novidades e Lançamentos da Temporada"
              className="p-2 rounded-xl bg-[#171b26] border border-amber-500/20 hover:border-amber-500/50 text-amber-400 hover:text-amber-300 relative transition-all group shadow-sm hover:shadow-amber-500/10"
            >
              <Flame className="w-4 h-4 group-hover:scale-110 transition-transform fill-amber-400/20 group-hover:fill-amber-400/40" />
              <span className="absolute -top-1 -right-1 flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
              </span>
            </button>
            <div
              onClick={() => setActiveTab('profile')}
              className="flex items-center gap-2 pl-2 border-l border-[#232936] cursor-pointer"
            >
              {profile.google_account?.picture ? (
                <img
                  src={profile.google_account.picture}
                  alt={profile.google_account.name}
                  referrerPolicy="no-referrer"
                  className="w-8 h-8 rounded-full object-cover border border-indigo-400/50 shadow"
                />
              ) : (
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-600 to-purple-500 flex items-center justify-center text-xs font-bold text-white shadow">
                  {profile.google_account ? profile.google_account.name.charAt(0).toUpperCase() : 'U'}
                </div>
              )}
              <div className="hidden md:block text-xs">
                <span className="font-semibold text-white block leading-none">
                  {profile.google_account ? profile.google_account.name : 'Meu Perfil'}
                </span>
                <span className="text-[10px] text-slate-400 font-mono">
                  {profile.total_ratings} interações
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Workspace Canvas */}
        <main className="p-3.5 sm:p-6 pb-24 md:pb-8 space-y-6 sm:space-y-8 max-w-7xl w-full mx-auto">
          {/* ============================================================== */}
          {/* TAB 1: INÍCIO (HOME)                                           */}
          {/* ============================================================== */}
          {activeTab === 'home' && (
            <>
              {heroAnime && (
                <div className="relative rounded-2xl overflow-hidden border border-white/10 bg-[#11141c] shadow-2xl min-h-[260px] flex items-end">
                  {heroAnime.banner_image || heroAnime.cover_image ? (
                    <img
                      src={heroAnime.banner_image || heroAnime.cover_image}
                      alt={heroAnime.title_english || heroAnime.title_romaji}
                      onError={(e) => handleBannerError(e, heroAnime.id)}
                      className="absolute inset-0 w-full h-full object-cover object-center opacity-35"
                    />
                  ) : null}
                  <div className="absolute inset-0 bg-gradient-to-r from-[#0a0c10] via-[#0a0c10]/85 to-transparent" />

                  <div className="relative z-10 p-6 sm:p-8 max-w-2xl space-y-3">
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30">
                      <Flame className="w-3.5 h-3.5 text-amber-400" />
                      Destaque no Catálogo
                    </span>
                    <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                      {heroAnime.title_english || heroAnime.title_romaji}
                    </h1>
                    <p className="text-xs text-slate-300 font-medium">
                      {heroAnime.genres.join(' • ')}
                    </p>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Avalie esta obra para atualizar suas recomendações, pule caso não tenha interesse em assistir, ou registre um drop com justificativa para o nosso algoritmo ajustar suas preferências.
                    </p>
                    <div className="flex flex-wrap items-center gap-2.5 pt-2">
                      <button
                        type="button"
                        onClick={() => setSelectedDetailAnime(heroAnime)}
                        className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-indigo-600/30 hover:bg-indigo-600/40 text-indigo-200 border border-indigo-500/40 flex items-center gap-1.5 transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Ver Detalhes</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleRateAnime(heroAnime, 9, 'onboarding')}
                        className="px-4 py-2 rounded-xl text-xs font-semibold bg-white text-slate-950 hover:bg-slate-200 flex items-center gap-2 transition-colors shadow"
                      >
                        <Play className="w-3.5 h-3.5 fill-slate-950" />
                        <span>Avaliar Agora (9/10)</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleToggleWatchlist(heroAnime)}
                        className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171b26]/90 hover:bg-[#1f2433] text-white border border-white/10 flex items-center gap-1.5 transition-colors"
                      >
                        <Bookmark className={`w-3.5 h-3.5 ${watchlistIds.has(heroAnime.id) ? 'fill-indigo-400 text-indigo-400' : ''}`} />
                        <span>{watchlistIds.has(heroAnime.id) ? 'Na Minha Lista' : '+ Minha Lista'}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleSkipAnime(heroAnime)}
                        className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171b26]/90 hover:bg-slate-800 text-slate-300 border border-white/10 flex items-center gap-1.5 transition-colors"
                      >
                        <EyeOff className="w-3.5 h-3.5" />
                        <span>Pular (Sem Interesse)</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleOpenDropModal(heroAnime, 'onboarding')}
                        className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-rose-950/50 hover:bg-rose-900/60 text-rose-300 border border-rose-500/30 flex items-center gap-1.5 transition-colors"
                      >
                        <Ban className="w-3.5 h-3.5" />
                        <span>Dropar</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* ============================================================ */}
              {/* RECOMMENDATION BLOCKING FILTERS (FORMATOS & GÊNEROS)         */}
              {/* ============================================================ */}
              <section
                data-testid="recommendation-filters-bar"
                className="p-5 rounded-2xl bg-[#11141c] border border-[#232936] space-y-4"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                      <SlidersHorizontal className="w-4 h-4" />
                    </div>
                    <div>
                      <h2 className="text-sm sm:text-base font-bold text-white">
                        Filtros de Recomendação & Bloqueio de Gêneros/Formatos
                      </h2>
                      <p className="text-xs text-slate-400">
                        Evite recomendações de formatos que não busca agora (ex.: Filmes/OVAs) ou bloqueie gêneros que você não gosta (ex.: Mecha, Terror, Romance).
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 self-start sm:self-auto">
                    {(excludedFormatsSet.size > 0 || blockedGenresSet.size > 0) && (
                      <button
                        type="button"
                        onClick={() => void handleClearAllFilters()}
                        className="px-3 py-1.5 rounded-xl text-xs font-medium bg-rose-950/40 hover:bg-rose-900/50 text-rose-200 border border-rose-500/30 transition-colors"
                      >
                        Limpar Bloqueios ({excludedFormatsSet.size + blockedGenresSet.size})
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => setShowFilterPanel((prev) => !prev)}
                      className="px-3 py-1.5 rounded-xl text-xs font-medium bg-[#171b26] hover:bg-[#1f2433] text-slate-300 border border-white/10 transition-colors"
                    >
                      {showFilterPanel ? 'Recolher Filtros' : 'Expandir Filtros'}
                    </button>
                  </div>
                </div>

                {showFilterPanel && (
                  <div className="space-y-4 pt-3 border-t border-white/5">
                    {/* Format Filter Row */}
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                          <Tv className="w-3.5 h-3.5 text-indigo-400" />
                          <span>O que você quer receber nas recomendações? (Formatos):</span>
                        </span>
                        <button
                          type="button"
                          onClick={() => void handleSetOnlyTvSeriesPreset()}
                          data-testid="preset-only-tv-btn"
                          className={`px-3 py-1 rounded-lg text-xs font-semibold border transition-all flex items-center gap-1.5 ${
                            isOnlyTvSeriesActive
                              ? 'bg-indigo-600 text-white border-indigo-400 shadow'
                              : 'bg-[#171b26] text-indigo-300 border-indigo-500/30 hover:bg-indigo-600/20'
                          }`}
                        >
                          <Tv className="w-3.5 h-3.5" />
                          <span>
                            {isOnlyTvSeriesActive
                              ? '✓ Modo Apenas Séries / TV Ativo (Clique p/ liberar Filmes/OVAs)'
                              : 'Ativar "Apenas Séries / TV" (Ocultar Filmes, OVAs e Especiais)'}
                          </span>
                        </button>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        {ALL_FORMATS.map((fmt) => {
                          const isBlocked = excludedFormatsSet.has(fmt.value);
                          return (
                            <button
                              key={fmt.value}
                              type="button"
                              onClick={() => void handleToggleExcludedFormat(fmt.value)}
                              className={`px-3 py-1.5 rounded-xl text-xs font-medium border transition-all flex items-center gap-1.5 ${
                                isBlocked
                                  ? 'bg-rose-950/50 text-rose-300 border-rose-500/40 line-through opacity-85'
                                  : 'bg-[#171b26] text-emerald-300 border-emerald-500/30 hover:bg-[#1f2433]'
                              }`}
                              title={
                                isBlocked
                                  ? `Formato ${fmt.label} bloqueado. Clique para permitir novamente.`
                                  : `Formato ${fmt.label} permitido. Clique para bloquear.`
                              }
                            >
                              {isBlocked ? (
                                <Ban className="w-3 h-3 text-rose-400 shrink-0" />
                              ) : (
                                <Film className="w-3 h-3 text-emerald-400 shrink-0" />
                              )}
                              <span>{fmt.label}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    {/* Genre Blocking Row */}
                    <div className="space-y-2 pt-2 border-t border-white/5">
                      <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                        <Ban className="w-3.5 h-3.5 text-rose-400" />
                        <span>
                          Bloquear Gêneros Indesejados (Clique nos gêneros que você NÃO quer receber):
                        </span>
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {BLOCKABLE_GENRES.map((g) => {
                          const isBlocked = blockedGenresSet.has(g.value.toLowerCase());
                          return (
                            <button
                              key={g.value}
                              type="button"
                              onClick={() => void handleToggleBlockedGenre(g.value)}
                              data-testid={`block-genre-${g.value.toLowerCase().replace(/\s+/g, '-')}`}
                              className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-all flex items-center gap-1 ${
                                isBlocked
                                  ? 'bg-rose-600 text-white border-rose-400 shadow-sm'
                                  : 'bg-[#171b26] text-slate-300 border-white/10 hover:border-rose-500/40 hover:text-white'
                              }`}
                            >
                              {isBlocked && <Ban className="w-3 h-3 shrink-0" />}
                              <span>{g.label}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}
              </section>

              {/* Dual Recommendation Tracks (80% Affinity vs 20% Discovery MMR) */}
              <div className="space-y-6">
                <TrackAffinity
                  animes={recommendations.affinity_track}
                  onRate={(a, r, t) => void handleRateAnime(a, r, t)}
                  onSkip={(a) => void handleSkipAnime(a)}
                  onDrop={(a, t) => handleOpenDropModal(a, t)}
                  onSelect={(a) => setSelectedDetailAnime(a)}
                  onToggleFavorite={(a) => void handleToggleFavorite(a)}
                  onToggleWatchlist={(a) => void handleToggleWatchlist(a)}
                  ratedMap={ratedMap}
                  favoriteIds={favoriteIds}
                  watchlistIds={watchlistIds}
                  droppedIds={droppedIds}
                  lastSafeReason={lastSafeReason}
                />

                <TrackDiscovery
                  animes={recommendations.discovery_track}
                  onRate={(a, r, t) => void handleRateAnime(a, r, t)}
                  onSelect={(a) => setSelectedDetailAnime(a)}
                  onSkip={(a) => void handleSkipAnime(a)}
                  onDrop={(a, t) => handleOpenDropModal(a, t)}
                  onToggleFavorite={(a) => void handleToggleFavorite(a)}
                  onToggleWatchlist={(a) => void handleToggleWatchlist(a)}
                  ratedMap={ratedMap}
                  favoriteIds={favoriteIds}
                  watchlistIds={watchlistIds}
                  droppedIds={droppedIds}
                />
              </div>

              {/* Cold-Start Onboarding & Popular Genres Shelf */}
              <OnboardingShelf
                animes={visibleOnboardingCatalog}
                selectedGenre={selectedGenre}
                onSelectGenre={setSelectedGenre}
                onRate={(a, r, t) => void handleRateAnime(a, r, t)}
                onSelect={(a) => setSelectedDetailAnime(a)}
                onSkip={(a) => void handleSkipAnime(a)}
                onDrop={(a, t) => handleOpenDropModal(a, t)}
                onToggleFavorite={(a) => void handleToggleFavorite(a)}
                onToggleWatchlist={(a) => void handleToggleWatchlist(a)}
                onExploreAll={() => setActiveTab('explore')}
                ratedMap={ratedMap}
                favoriteIds={favoriteIds}
                watchlistIds={watchlistIds}
                droppedIds={droppedIds}
              />

              {/* Simplified Genre Preferences + Advanced Vector Toggle */}
              <ProfileMetrics profile={profile} spoilerTagsSet={SPOILER_KEYWORDS} />
            </>
          )}

          {/* ============================================================== */}
          {/* TAB 2: EXPLORAR CATÁLOGO (EXPLORE)                             */}
          {/* ============================================================== */}
          {activeTab === 'explore' && (
            <section className="space-y-6">
              <div className="p-5 rounded-2xl bg-[#11141c] border border-[#232936] space-y-4">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <Compass className="w-5 h-5 text-indigo-400" />
                      <h1 className="text-xl font-bold text-white tracking-tight">
                        Explorar Catálogo Completo
                      </h1>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">
                      Filtre por gênero, formato ou popularidade. Avalie obras assistidas, pule as que não tem interesse ou registre drops com justificativa.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                      <input
                        type="text"
                        value={exploreQuery}
                        onChange={(e) => setExploreQuery(e.target.value)}
                        placeholder="Filtrar por nome ou tag..."
                        className="pl-8 pr-3 py-1.5 rounded-xl bg-[#0a0c10] border border-white/10 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                      />
                    </div>

                    <div className="flex items-center gap-1.5 bg-[#171b26] px-2.5 py-1.5 rounded-xl border border-white/10 text-xs">
                      <Filter className="w-3.5 h-3.5 text-slate-400" />
                      <select
                        value={exploreFormat}
                        onChange={(e) => setExploreFormat(e.target.value)}
                        className="bg-transparent text-slate-200 focus:outline-none"
                      >
                        <option value="ALL" className="bg-[#11141c]">Todos Formatos</option>
                        <option value="TV" className="bg-[#11141c]">Séries (TV)</option>
                        <option value="MOVIE" className="bg-[#11141c]">Filme (MOVIE)</option>
                        <option value="OVA" className="bg-[#11141c]">OVA</option>
                        <option value="ONA" className="bg-[#11141c]">ONA</option>
                        <option value="SPECIAL" className="bg-[#11141c]">Especial (SPECIAL)</option>
                      </select>
                    </div>

                    <div className="flex items-center gap-1.5 bg-[#171b26] px-2.5 py-1.5 rounded-xl border border-white/10 text-xs">
                      <ArrowUpDown className="w-3.5 h-3.5 text-slate-400" />
                      <select
                        value={exploreSort}
                        onChange={(e) => setExploreSort(e.target.value as 'score' | 'popularity' | 'year' | 'title')}
                        className="bg-transparent text-slate-200 focus:outline-none"
                      >
                        <option value="popularity" className="bg-[#11141c]">Mais Populares</option>
                        <option value="score" className="bg-[#11141c]">Maior Nota Média</option>
                        <option value="year" className="bg-[#11141c]">Mais Recentes</option>
                        <option value="title" className="bg-[#11141c]">Ordem Alfabética (A-Z)</option>
                      </select>
                    </div>

                    <label className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#171b26] border border-white/10 text-xs text-slate-300 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={hideInteractedInExplore}
                        onChange={(e) => setHideInteractedInExplore(e.target.checked)}
                        className="accent-indigo-500 rounded"
                      />
                      <span>Ocultar já avaliados/pulados</span>
                    </label>
                  </div>
                </div>

                {/* Genre Filter Pills */}
                <div className="flex items-center gap-2 overflow-x-auto pb-1">
                  <button
                    type="button"
                    onClick={() => setExploreGenre('ALL')}
                    className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors shrink-0 ${
                      exploreGenre === 'ALL'
                        ? 'bg-indigo-600 text-white border-indigo-500'
                        : 'bg-[#171b26] text-slate-300 border-white/10 hover:text-white'
                    }`}
                  >
                    Todos ({fullCatalog.length})
                  </button>
                  {EXPLORE_GENRES.map((g) => (
                    <button
                      key={g}
                      type="button"
                      onClick={() => setExploreGenre(exploreGenre === g ? 'ALL' : g)}
                      className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors shrink-0 ${
                        exploreGenre === g
                          ? 'bg-indigo-600 text-white border-indigo-500'
                          : 'bg-[#171b26] text-slate-300 border-white/10 hover:text-white'
                      }`}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              </div>

              {exploreAnimes.length === 0 ? (
                <div className="p-12 rounded-2xl bg-[#11141c] border border-[#232936] text-center text-sm text-slate-400">
                  Nenhuma obra encontrada para os filtros selecionados.
                </div>
              ) : (
                <>
                  <div className="flex items-center justify-between px-1 text-xs text-slate-400">
                    <span>
                      Exibindo <strong className="text-slate-200">{Math.min(exploreVisibleCount, exploreAnimes.length)}</strong> de{' '}
                      <strong className="text-slate-200">{exploreAnimes.length}</strong> obras encontradas
                    </span>
                    <span className="font-mono text-[11px] text-indigo-400">
                      Rolagem infinita ativa
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                    {exploreAnimes.slice(0, exploreVisibleCount).map((anime) => (
                      <AnimeCard
                        key={anime.id}
                        anime={anime}
                        trackType="onboarding"
                        onRate={(a, r, t) => void handleRateAnime(a, r, t)}
                        onSelect={(a) => setSelectedDetailAnime(a)}
                        onSkip={(a) => void handleSkipAnime(a)}
                        onDrop={(a, t) => handleOpenDropModal(a, t)}
                        onToggleFavorite={(a) => void handleToggleFavorite(a)}
                        onToggleWatchlist={(a) => void handleToggleWatchlist(a)}
                        userRating={ratedMap[anime.id]}
                        isFavorite={favoriteIds.has(anime.id)}
                        isInWatchlist={watchlistIds.has(anime.id)}
                        isDropped={droppedIds.has(anime.id)}
                      />
                    ))}
                  </div>

                  {/* Infinite Scroll Sentinel */}
                  {exploreVisibleCount < exploreAnimes.length && (
                    <div
                      ref={exploreSentinelRef}
                      className="py-6 flex flex-col items-center justify-center gap-2 text-xs text-slate-400"
                    >
                      <button
                        type="button"
                        onClick={() =>
                          setExploreVisibleCount((prev) => Math.min(prev + 24, exploreAnimes.length))
                        }
                        className="px-4 py-2 rounded-xl bg-[#171b26] hover:bg-[#1f2433] text-indigo-300 border border-indigo-500/30 font-semibold transition-colors"
                      >
                        Carregar mais obras ({exploreAnimes.length - exploreVisibleCount} restantes)
                      </button>
                    </div>
                  )}
                </>
              )}
            </section>
          )}

          {/* ============================================================== */}
          {/* TAB 3: MINHA LISTA (WATCHED, WATCHLIST, DROPPED, SKIPPED)      */}
          {/* ============================================================== */}
          {activeTab === 'list' && (
            <section className="space-y-6">
              <div className="p-5 rounded-2xl bg-[#11141c] border border-[#232936] space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <Bookmark className="w-5 h-5 text-indigo-400" />
                      <h1 className="text-xl font-bold text-white tracking-tight">
                        Minha Lista & Histórico de Interações
                      </h1>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">
                      Gerencie animes assistidos, salvos para ver depois, dropados e pulados.
                    </p>
                  </div>
                </div>

                {/* Sub-tabs */}
                <div className="flex flex-wrap gap-2 pt-1 border-t border-white/5">
                  <button
                    type="button"
                    onClick={() => setListSubTab('watched')}
                    className={`px-3.5 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 border transition-colors ${
                      listSubTab === 'watched'
                        ? 'bg-indigo-600 text-white border-indigo-500'
                        : 'bg-[#171b26] text-slate-300 border-white/10 hover:text-white'
                    }`}
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Assistidos & Avaliados ({watchedEntries.length})</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setListSubTab('watchlist')}
                    className={`px-3.5 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 border transition-colors ${
                      listSubTab === 'watchlist'
                        ? 'bg-indigo-600 text-white border-indigo-500'
                        : 'bg-[#171b26] text-slate-300 border-white/10 hover:text-white'
                    }`}
                  >
                    <Bookmark className="w-3.5 h-3.5" />
                    <span>Quero Assistir ({watchlistAnimes.length})</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setListSubTab('dropped')}
                    className={`px-3.5 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 border transition-colors ${
                      listSubTab === 'dropped'
                        ? 'bg-rose-600 text-white border-rose-500'
                        : 'bg-[#171b26] text-slate-300 border-white/10 hover:text-white'
                    }`}
                  >
                    <Ban className="w-3.5 h-3.5" />
                    <span>Dropados ({droppedEntries.length})</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setListSubTab('skipped')}
                    data-testid="subtab-skipped-btn"
                    className={`px-3.5 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 border transition-colors ${
                      listSubTab === 'skipped'
                        ? 'bg-slate-700 text-white border-slate-500'
                        : 'bg-[#171b26] text-slate-300 border-white/10 hover:text-white'
                    }`}
                  >
                    <EyeOff className="w-3.5 h-3.5" />
                    <span>Pulados / Sem Interesse ({skippedAnimes.length})</span>
                  </button>
                </div>
              </div>

              {/* Sub-tab 1: Assistidos & Avaliados */}
              {listSubTab === 'watched' && (
                <div className="space-y-3">
                  {watchedEntries.length === 0 ? (
                    <div className="p-10 rounded-2xl bg-[#11141c] border border-[#232936] text-center text-sm text-slate-400">
                      Você ainda não avaliou nenhum anime. Avalie títulos na tela inicial ou em Explorar!
                    </div>
                  ) : (
                    watchedEntries.map((entry) => {
                      const animeObj = resolveAnimeById(entry.anime_id);
                      return (
                        <div
                          key={entry.id}
                          className="p-4 rounded-2xl bg-[#11141c] border border-[#232936] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
                        >
                          <div
                            onClick={() => setSelectedDetailAnime(animeObj)}
                            className="flex items-center gap-4 cursor-pointer group"
                          >
                            <img
                              src={animeObj.cover_image}
                              alt={entry.anime_title}
                              onError={(e) => handleCoverError(e, animeObj.id)}
                              className="w-14 h-20 object-cover rounded-lg border border-white/10 shrink-0 group-hover:scale-105 transition-transform"
                            />
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <h3 className="text-sm font-bold text-white group-hover:text-indigo-400 transition-colors">
                                  {entry.anime_title}
                                </h3>
                                <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                                  Nota: {entry.rating}/10
                                </span>
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-400">
                                  Trilha: {entry.origin_track}
                                </span>
                              </div>
                              {entry.user_review_raw && (
                                <p className="text-xs text-slate-300 italic">
                                  "{entry.user_review_raw}"
                                </p>
                              )}
                              {entry.qualitative_tuning && (
                                <p className="text-xs text-emerald-300 flex items-center gap-1.5">
                                  <Sparkles className="w-3.5 h-3.5 shrink-0" />
                                  <span>{entry.qualitative_tuning.public_safe_reason}</span>
                                </p>
                              )}
                            </div>
                          </div>

                          <div className="flex items-center gap-2 shrink-0">
                            <button
                              type="button"
                              onClick={() => setSelectedDetailAnime(animeObj)}
                              className="px-3 py-1.5 rounded-xl text-xs font-medium bg-[#171b26] hover:bg-[#1f2433] text-slate-300 border border-white/10"
                            >
                              Ver Detalhes
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setEditingHistoryEntry(entry);
                                setEditRatingValue(entry.rating);
                                setEditReviewValue(entry.user_review_raw || '');
                              }}
                              className="px-3 py-1.5 rounded-xl text-xs font-medium bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/30 flex items-center gap-1.5 transition-colors"
                            >
                              <Star className="w-3.5 h-3.5 fill-amber-400 text-amber-400" />
                              <span>Editar Nota</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setModalMode('like');
                                setModalAnime(animeObj);
                                setModalRating(entry.rating);
                                setTuningResult(null);
                                setModalOpen(true);
                              }}
                              className="px-3 py-1.5 rounded-xl text-xs font-medium bg-[#171b26] hover:bg-indigo-600/20 text-indigo-300 border border-indigo-500/30"
                            >
                              Refinar Avaliação
                            </button>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              )}

              {/* Sub-tab 2: Quero Assistir (Watchlist) */}
              {listSubTab === 'watchlist' && (
                <div>
                  {watchlistAnimes.length === 0 ? (
                    <div className="p-10 rounded-2xl bg-[#11141c] border border-[#232936] text-center text-sm text-slate-400">
                      Sua lista "Quero Assistir" está vazia. Clique no ícone de marcador (Bookmark) em qualquer anime para salvá-lo aqui!
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                      {watchlistAnimes.map((anime) => (
                        <AnimeCard
                          key={anime.id}
                          anime={anime}
                          trackType="onboarding"
                          onRate={(a, r, t) => void handleRateAnime(a, r, t)}
                          onSelect={(a) => setSelectedDetailAnime(a)}
                          onSkip={(a) => void handleSkipAnime(a)}
                          onDrop={(a, t) => handleOpenDropModal(a, t)}
                          onToggleFavorite={(a) => void handleToggleFavorite(a)}
                          onToggleWatchlist={(a) => void handleToggleWatchlist(a)}
                          userRating={ratedMap[anime.id]}
                          isFavorite={favoriteIds.has(anime.id)}
                          isInWatchlist={true}
                          isDropped={droppedIds.has(anime.id)}
                        />
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Sub-tab 3: Dropados */}
              {listSubTab === 'dropped' && (
                <div className="space-y-3">
                  {droppedEntries.length === 0 ? (
                    <div className="p-10 rounded-2xl bg-[#11141c] border border-[#232936] text-center text-sm text-slate-400">
                      Nenhum anime dropado até o momento. Quando você dropar uma obra e explicar o motivo, o diagnóstico do nosso algoritmo aparecerá aqui.
                    </div>
                  ) : (
                    droppedEntries.map((entry) => {
                      const animeObj = resolveAnimeById(entry.anime_id);
                      return (
                        <div
                          key={entry.id}
                          className="p-4 rounded-2xl bg-[#11141c] border border-rose-500/30 space-y-3"
                        >
                          <div className="flex items-start gap-4">
                            <img
                              src={animeObj.cover_image}
                              alt={entry.anime_title}
                              onClick={() => setSelectedDetailAnime(animeObj)}
                              onError={(e) => handleCoverError(e, animeObj.id)}
                              className="w-14 h-20 object-cover rounded-lg border border-white/10 shrink-0 cursor-pointer hover:scale-105 transition-transform"
                            />
                            <div className="space-y-1.5 flex-1">
                              <div className="flex items-center gap-2">
                                <h3
                                  onClick={() => setSelectedDetailAnime(animeObj)}
                                  className="text-sm font-bold text-white cursor-pointer hover:text-indigo-400 transition-colors"
                                >
                                  {entry.anime_title}
                                </h3>
                                <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30 flex items-center gap-1">
                                  <Ban className="w-3 h-3" /> Dropado
                                </span>
                              </div>
                              {(entry.drop_reason || entry.user_review_raw) && (
                                <p className="text-xs text-slate-200">
                                  <strong>Sua justificativa de drop:</strong> "{entry.drop_reason || entry.user_review_raw}"
                                </p>
                              )}
                              {entry.qualitative_tuning && (
                                <div className="p-3 rounded-xl bg-rose-950/30 border border-rose-500/20 space-y-2">
                                  <p className="text-xs text-rose-200 flex items-center gap-1.5">
                                    <Sparkles className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                                    <span>
                                      <strong>Diagnóstico do Algoritmo:</strong> {entry.qualitative_tuning.public_safe_reason}
                                    </span>
                                  </p>
                                  <div className="flex flex-wrap gap-1.5">
                                    {entry.qualitative_tuning.identified_tags.map((t) => (
                                      <span
                                        key={t.tag_name}
                                        className="px-2 py-0.5 rounded text-[11px] font-mono bg-rose-950/70 text-rose-300 border border-rose-500/30"
                                      >
                                        {t.tag_name}: {t.impact_multiplier}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              )}

              {/* Sub-tab 4: Pulados / Sem Interesse */}
              {listSubTab === 'skipped' && (
                <div className="space-y-4">
                  {skippedAnimes.length > 0 && (
                    <div className="flex items-center justify-between px-1">
                      <span className="text-xs text-slate-400">
                        Exibindo {skippedAnimes.length} obras ocultadas das suas recomendações
                      </span>
                      <button
                        type="button"
                        onClick={() => void handleRestoreAllSkipped()}
                        className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-indigo-600/20 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/30 flex items-center gap-1.5 transition-colors"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        <span>Restaurar Todos ({skippedAnimes.length})</span>
                      </button>
                    </div>
                  )}

                  {skippedAnimes.length === 0 ? (
                    <div className="p-10 rounded-2xl bg-[#11141c] border border-[#232936] text-center text-sm text-slate-400">
                      Nenhum anime pulado no momento. Use o botão "Pular" nos cards para ocultar animes que você não assistiu ou não tem interesse sem afetar suas preferências.
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                      {skippedAnimes.map((anime) => (
                        <div
                          key={anime.id}
                          className="p-3.5 rounded-xl bg-[#11141c] border border-[#232936] flex items-center justify-between gap-3"
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <img
                              src={anime.cover_image}
                              alt={anime.title_romaji}
                              onError={(e) => handleCoverError(e, anime.id)}
                              className="w-12 h-16 object-cover rounded-lg border border-white/10 shrink-0"
                            />
                            <div className="min-w-0 space-y-0.5">
                              <h3 className="text-xs font-bold text-white truncate">
                                {anime.title_english || anime.title_romaji}
                              </h3>
                              <p className="text-[11px] text-slate-400">
                                {anime.format} • {anime.genres.slice(0, 2).join(', ') || 'Sem impacto no perfil'}
                              </p>
                              <span className="inline-block text-[10px] text-slate-500">
                                Pulado (Sem interesse agora)
                              </span>
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => void handleSkipAnime(anime, true)}
                            className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-[#171b26] hover:bg-indigo-600 text-slate-200 hover:text-white border border-white/10 flex items-center gap-1 shrink-0 transition-colors"
                          >
                            <RotateCcw className="w-3 h-3" />
                            <span>Restaurar</span>
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          {/* ============================================================== */}
          {/* TAB 4: FAVORITOS (FAVORITES)                                   */}
          {/* ============================================================== */}
          {activeTab === 'favorites' && (
            <section className="space-y-6">
              <div className="p-5 rounded-2xl bg-[#11141c] border border-rose-500/20 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <Heart className="w-5 h-5 text-rose-400 fill-rose-400" />
                    <h1 className="text-xl font-bold text-white tracking-tight">
                      Meus Animes Favoritos & Obras Nota 9+
                    </h1>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Obras marcadas com o coração ou avaliadas com nota 9 ou 10 que servem de referência principal para suas recomendações.
                  </p>
                </div>
                <span className="text-xs font-mono text-rose-300 bg-rose-500/10 border border-rose-500/20 px-3 py-1 rounded-full">
                  {favoriteAnimes.length} obras favoritas
                </span>
              </div>

              {favoriteAnimes.length === 0 ? (
                <div className="p-12 rounded-2xl bg-[#11141c] border border-[#232936] text-center text-sm text-slate-400">
                  Você ainda não marcou favoritos nem avaliou obras com nota 9+. Clique no ícone de coração nos cards ou avalie seus animes preferidos!
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {favoriteAnimes.map((anime) => (
                    <AnimeCard
                      key={anime.id}
                      anime={anime}
                      trackType="affinity"
                      onRate={(a, r, t) => void handleRateAnime(a, r, t)}
                      onSelect={(a) => setSelectedDetailAnime(a)}
                      onSkip={(a) => void handleSkipAnime(a)}
                      onDrop={(a, t) => handleOpenDropModal(a, t)}
                      onToggleFavorite={(a) => void handleToggleFavorite(a)}
                      onToggleWatchlist={(a) => void handleToggleWatchlist(a)}
                      userRating={ratedMap[anime.id]}
                      isFavorite={favoriteIds.has(anime.id)}
                      isInWatchlist={watchlistIds.has(anime.id)}
                      isDropped={droppedIds.has(anime.id)}
                    />
                  ))}
                </div>
              )}
            </section>
          )}

          {/* ============================================================== */}
          {/* TAB 5: MEU PERFIL (SIMPLIFIED BY DEFAULT + GOOGLE SYNC)        */}
          {/* ============================================================== */}
          {activeTab === 'profile' && (
            <section className="space-y-6">
              {/* Google Account Sync Card in Profile */}
              <div className="p-5 rounded-2xl bg-[#11141c] border border-[#232936] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3.5">
                  {profile.google_account?.picture ? (
                    <img
                      src={profile.google_account.picture}
                      alt={profile.google_account.name}
                      referrerPolicy="no-referrer"
                      className="w-12 h-12 rounded-2xl object-cover border border-indigo-400/50 shadow shrink-0"
                    />
                  ) : (
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-indigo-600 to-purple-500 flex items-center justify-center text-lg font-bold text-white shadow shrink-0">
                      {profile.google_account ? profile.google_account.name.charAt(0).toUpperCase() : 'U'}
                    </div>
                  )}
                  <div>
                    <h1 className="text-lg font-bold text-white">
                      {profile.google_account
                        ? profile.google_account.name
                        : 'Perfil Local (Sem conta vinculada)'}
                    </h1>
                    <p className="text-xs text-slate-400">
                      {profile.google_account
                        ? `Conectado ao Google (${profile.google_account.email}) • Sincronização automática em tempo real ativa`
                        : 'Vincule sua conta Google real para sincronizar seu histórico, notas, favoritos e lista automaticamente.'}
                    </p>
                  </div>
                </div>

                {profile.google_account ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="px-3 py-2 rounded-xl text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      <span>Sincronizado Automaticamente</span>
                    </span>
                    <button
                      type="button"
                      onClick={() => void handleGoogleSignOut()}
                      className="px-3 py-2 rounded-xl text-xs font-medium bg-[#171b26] hover:bg-rose-950/40 text-slate-300 hover:text-rose-200 border border-white/10 transition-colors"
                    >
                      Desconectar
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => void handleRealGoogleLogin()}
                    disabled={isSyncingGoogle}
                    className="px-4 py-2.5 rounded-xl text-xs font-semibold bg-white hover:bg-slate-100 disabled:opacity-60 text-slate-900 flex items-center gap-2 shadow transition-colors self-start sm:self-auto"
                  >
                    <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24">
                      <path
                        fill="#4285F4"
                        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                      />
                      <path
                        fill="#34A853"
                        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                      />
                      <path
                        fill="#FBBC05"
                        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                      />
                      <path
                        fill="#EA4335"
                        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                      />
                    </svg>
                    <span>{isSyncingGoogle ? 'Abrindo Google...' : 'Vincular Conta Google'}</span>
                  </button>
                )}
              </div>

              {/* AniList Integration Card */}
              <div className="p-5 rounded-2xl bg-[#11141c] border border-sky-500/30 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3.5">
                  <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-sky-600 to-blue-500 flex items-center justify-center text-lg font-bold text-white shadow shrink-0">
                    AL
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h1 className="text-lg font-bold text-white">
                        Sincronização com AniList
                      </h1>
                      {anilistSyncSummary && (
                        <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-sky-500/20 text-sky-300 border border-sky-500/30">
                          @{anilistSyncSummary.username}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">
                      {anilistSyncSummary
                        ? `Sincronizado: ${anilistSyncSummary.watched} assistidos, ${anilistSyncSummary.watchlist} na lista, ${anilistSyncSummary.dropped} dropados e ${anilistSyncSummary.favorites} favoritos integrados.`
                        : 'Importe sua lista do AniList para calibrar os pesos das recomendações com seus animes assistidos, notas e favoritos.'}
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => setAnilistModalOpen(true)}
                  className="px-4 py-2.5 rounded-xl text-xs font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 flex items-center gap-2 shadow transition-colors self-start sm:self-auto shrink-0"
                >
                  <RefreshCw className={`w-4 h-4 ${isSyncingAniList ? 'animate-spin' : ''}`} />
                  <span>{anilistSyncSummary ? 'Atualizar do AniList' : 'Conectar com AniList'}</span>
                </button>
              </div>

              {/* Simple Activity Counters */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-4 rounded-2xl bg-[#11141c] border border-[#232936]">
                  <span className="text-xs text-slate-400 block">Assistidos & Avaliados</span>
                  <span className="text-2xl font-mono font-bold text-indigo-400">
                    {watchedEntries.length}
                  </span>
                </div>
                <div className="p-4 rounded-2xl bg-[#11141c] border border-[#232936]">
                  <span className="text-xs text-slate-400 block">Favoritos (❤️ / 9+)</span>
                  <span className="text-2xl font-mono font-bold text-rose-400">
                    {favoriteAnimes.length}
                  </span>
                </div>
                <div className="p-4 rounded-2xl bg-[#11141c] border border-[#232936]">
                  <span className="text-xs text-slate-400 block">Dropados</span>
                  <span className="text-2xl font-mono font-bold text-amber-400">
                    {droppedEntries.length}
                  </span>
                </div>
                <div className="p-4 rounded-2xl bg-[#11141c] border border-[#232936]">
                  <span className="text-xs text-slate-400 block">Pulados (Sem interesse)</span>
                  <span className="text-2xl font-mono font-bold text-slate-300">
                    {skippedAnimes.length}
                  </span>
                </div>
              </div>

              {/* Simplified Genres View + Optional Advanced Vector Insights */}
              <ProfileMetrics profile={profile} spoilerTagsSet={SPOILER_KEYWORDS} />

              {/* Mobile & Desktop Profile Reset Action Card */}
              <div className="p-5 rounded-2xl bg-[#11141c] border border-rose-500/20 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    <RotateCcw className="w-4 h-4 text-rose-400" />
                    <span>Gerenciamento de Histórico e Preferências</span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Deseja recomeçar a calibração do zero? Você pode limpar suas avaliações, listas e pesos sem desconectar sua conta Google.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setConfirmResetOpen(true)}
                  className="px-4 py-2.5 rounded-xl text-xs font-semibold bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 border border-rose-500/30 flex items-center gap-2 transition-colors self-start sm:self-auto shrink-0"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Resetar Preferências</span>
                </button>
              </div>
            </section>
          )}
        </main>
      </div>

      {/* ==================================================================== */}
      {/* ANILIST INTEGRATION MODAL                                            */}
      {/* ==================================================================== */}
      {anilistModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-2xl bg-[#11141c] border border-sky-500/30 p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-sky-500 flex items-center justify-center font-black text-slate-950 text-sm">
                  AL
                </div>
                <div>
                  <h2 className="text-base font-bold text-white">
                    Conectar Conta AniList
                  </h2>
                  <span className="text-[11px] text-sky-400">Sincronização de Listas & Algoritmo</span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setAnilistModalOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-3.5 rounded-xl bg-sky-950/30 border border-sky-500/20 text-xs text-slate-300 space-y-1.5">
              <p className="font-semibold text-sky-300">💡 Como sincronizar sem complicação:</p>
              <p className="text-slate-300 leading-relaxed">
                Se sua conta no AniList for pública, <strong>basta digitar seu Nome de Usuário</strong> abaixo! Não é necessário criar aplicativo nem gerar token.
              </p>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                void handleAniListSyncSubmit(anilistUsernameInput, anilistTokenInput);
              }}
              className="space-y-3.5"
            >
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Nome de Usuário no AniList
                </label>
                <input
                  type="text"
                  required
                  value={anilistUsernameInput}
                  onChange={(e) => setAnilistUsernameInput(e.target.value)}
                  placeholder="Ex.: seu_usuario_anilist"
                  className="w-full px-3.5 py-2 rounded-xl bg-[#0a0c10] border border-white/10 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-sky-500"
                />
              </div>

              <div>
                <details className="text-xs text-slate-400 space-y-2">
                  <summary className="cursor-pointer text-sky-400 hover:text-sky-300 font-medium">
                    Opções Avançadas (Token OAuth / Portal AniList Developers)
                  </summary>
                  <div className="pt-2 space-y-2.5 bg-black/30 p-3 rounded-xl border border-white/5">
                    <div className="text-[11px] text-slate-300 space-y-1 leading-relaxed">
                      <p className="font-semibold text-sky-300">📌 O que preencher ao criar cliente na AniList:</p>
                      <ul className="list-disc pl-4 space-y-0.5 text-slate-400">
                        <li><strong>Name:</strong> AnimeRecSys AI</li>
                        <li><strong>Redirect URL:</strong> <code className="text-sky-300 bg-sky-950/40 px-1 py-0.5 rounded">http://localhost:5173</code></li>
                      </ul>
                    </div>
                    <div>
                      <label className="block text-[11px] font-medium text-slate-300 mb-1">
                        Token de Acesso OAuth (Opcional - apenas para contas privadas)
                      </label>
                      <input
                        type="text"
                        value={anilistTokenInput}
                        onChange={(e) => setAnilistTokenInput(e.target.value)}
                        placeholder="Cole seu token de acesso aqui"
                        className="w-full px-3.5 py-2 rounded-xl bg-[#0a0c10] border border-white/10 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-sky-500"
                      />
                    </div>
                  </div>
                </details>
              </div>

              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setAnilistModalOpen(false)}
                  className="px-3.5 py-2 rounded-xl text-xs font-medium bg-[#171b26] text-slate-300 hover:text-white"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={isSyncingAniList || !anilistUsernameInput.trim()}
                  className="px-4 py-2 rounded-xl text-xs font-semibold bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-slate-950 shadow transition-colors flex items-center gap-1.5 cursor-pointer"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isSyncingAniList ? 'animate-spin' : ''}`} />
                  <span>{isSyncingAniList ? 'Sincronizando...' : 'Sincronizar Listas'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ==================================================================== */}
      {/* CONFIRMATION MODAL FOR RESETTING PROFILE PREFERENCES                 */}
      {/* ==================================================================== */}
      {confirmResetOpen && (
        <div className="fixed inset-0 z-[95] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in">
          <div className="w-full max-w-md rounded-2xl bg-[#11141c] border border-rose-500/30 p-6 shadow-2xl space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-rose-500/15 text-rose-400 border border-rose-500/30">
                  <RotateCcw className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-white">
                    Resetar Preferências e Histórico?
                  </h3>
                  <p className="text-xs text-slate-400">
                    Confirmação de segurança
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setConfirmResetOpen(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Esta ação apagará suas <strong>{profile.total_ratings} avaliações</strong>, favoritos, lista "Quero Assistir", drops e recalibrará seu perfil para o estado inicial. Deseja realmente continuar?
            </p>

            <div className="flex items-center justify-end gap-2.5 pt-2">
              <button
                type="button"
                onClick={() => setConfirmResetOpen(false)}
                className="px-4 py-2 rounded-xl text-xs font-medium bg-[#171b26] text-slate-300 hover:text-white border border-white/10"
              >
                Manter Meus Dados
              </button>
              <button
                type="button"
                onClick={() => void handleResetProfile()}
                data-testid="confirm-reset-profile-btn"
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/20 flex items-center gap-1.5 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Sim, Resetar Tudo</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Rating & Recalibrate Modal */}
      {editingHistoryEntry && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in">
          <div className="relative w-full max-w-md max-h-[90vh] overflow-y-auto rounded-2xl bg-[#11141c] border border-amber-500/30 p-5 sm:p-6 shadow-2xl space-y-5">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[10px] font-mono uppercase tracking-wider text-amber-400 font-bold px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20">
                  Recalibração do Perfil
                </span>
                <h3 className="text-lg font-bold text-white mt-1">
                  Editar Avaliação
                </h3>
                <p className="text-xs text-slate-400">
                  {editingHistoryEntry.anime_title}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setEditingHistoryEntry(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold text-slate-300">
                    Sua Nova Nota:
                  </label>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-bold font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    {editRatingValue}/10
                  </span>
                </div>
                <div className="grid grid-cols-5 sm:grid-cols-10 gap-1.5 pt-1">
                  {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((score) => (
                    <button
                      key={score}
                      type="button"
                      onClick={() => setEditRatingValue(score)}
                      className={`py-2.5 sm:py-2 rounded-lg text-xs font-bold font-mono transition-all ${
                        editRatingValue === score
                          ? 'bg-amber-500 text-slate-950 shadow-md scale-105'
                          : 'bg-[#171b26] text-slate-300 hover:bg-[#232936] hover:text-white border border-white/5'
                      }`}
                    >
                      {score}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Comentário / Opinião (Opcional):
                </label>
                <textarea
                  value={editReviewValue}
                  onChange={(e) => setEditReviewValue(e.target.value)}
                  placeholder="Por que mudou de ideia sobre esta obra? (Ex: gostei mais do final, ou o ritmo piorou...)"
                  rows={3}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-[#0a0c10] border border-white/10 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-amber-500/60 resize-none"
                />
              </div>

              <div className="p-3 rounded-xl bg-amber-500/5 border border-amber-500/20 text-[11px] text-amber-200/90 leading-relaxed">
                💡 Ao salvar, a nota e o comentário recalibram imediatamente os pesos do seu perfil no algoritmo.
              </div>
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pt-2 border-t border-white/5">
              <button
                type="button"
                onClick={() => void handleCancelRating(editingHistoryEntry.anime_id)}
                className="px-3 py-2 rounded-xl text-xs font-semibold text-rose-300 hover:text-white bg-rose-950/40 hover:bg-rose-600/80 border border-rose-500/30 flex items-center justify-center gap-1.5 transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Remover Avaliação</span>
              </button>

              <div className="flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setEditingHistoryEntry(null)}
                  className="px-4 py-2 rounded-xl text-xs font-medium bg-[#171b26] text-slate-300 hover:text-white border border-white/5"
                >
                  Fechar
                </button>
                <button
                  type="button"
                  onClick={() => void handleSaveEditRating()}
                  disabled={isSavingEditRating}
                  className="px-4 py-2 rounded-xl text-xs font-semibold bg-amber-500 hover:bg-amber-400 text-slate-950 shadow-lg shadow-amber-500/20 transition-all flex items-center gap-1.5 disabled:opacity-50 cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{isSavingEditRating ? 'Recalibrando...' : 'Salvar Alteração'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Qualitative & Drop Analysis Modal */}
      <QualitativeModal
        anime={modalAnime}
        rating={modalRating}
        mode={modalMode}
        isOpen={modalOpen}
        isSubmitting={isTuning}
        tuningResult={tuningResult}
        onClose={() => setModalOpen(false)}
        onSubmitFeedback={(animeId, feedback) => {
          void handleSubmitModalFeedback(animeId, feedback);
        }}
        onCancelRating={(animeId) => {
          void handleCancelRating(animeId);
        }}
      />

      {/* Anime Detail Modal (Centered with Backdrop Blur and Spoiler Protection) */}
      <AnimeDetailModal
        anime={selectedDetailAnime}
        onClose={() => setSelectedDetailAnime(null)}
        onRate={(anime, score) => {
          void handleRateAnime(anime, score, 'direct_search');
        }}
        onToggleWatchlist={(anime) => {
          void handleToggleWatchlist(anime);
        }}
        onToggleFavorite={(anime) => {
          void handleToggleFavorite(anime);
        }}
        onSkip={(anime) => {
          void handleSkipAnime(anime);
        }}
        onDrop={(anime) => {
          handleOpenDropModal(anime, 'affinity');
        }}
        isInWatchlist={selectedDetailAnime ? watchlistIds.has(selectedDetailAnime.id) : false}
        isFavorite={selectedDetailAnime ? favoriteIds.has(selectedDetailAnime.id) : false}
        isDropped={selectedDetailAnime ? droppedIds.has(selectedDetailAnime.id) : false}
        isSkipped={selectedDetailAnime ? skippedIds.has(selectedDetailAnime.id) : false}
        userRating={selectedDetailAnime ? ratedMap[selectedDetailAnime.id] ?? null : null}
        onEditRating={(anime, currentRating) => {
          const entry = watchHistory.find((h) => h.anime_id === anime.id);
          if (entry) {
            setEditingHistoryEntry(entry);
            setEditRatingValue(entry.rating);
            setEditReviewValue(entry.user_review_raw || '');
          } else {
            setEditingHistoryEntry({
              id: `wh-${anime.id}`,
              anime_id: anime.id,
              anime_title: anime.title_english || anime.title_romaji,
              rating: currentRating,
              origin_track: 'direct_search',
              interaction_status: 'watched',
              timestamp: new Date().toISOString(),
            });
            setEditRatingValue(currentRating);
            setEditReviewValue('');
          }
        }}
      />

      {/* Barra Lateral Direita: Lançamentos da Temporada Atual com Blur na Página Principal */}
      <SeasonalDrawer
        isOpen={isSeasonalDrawerOpen}
        onClose={() => setIsSeasonalDrawerOpen(false)}
        onSelectAnime={(anime) => {
          setIsSeasonalDrawerOpen(false);
          setSelectedDetailAnime(anime);
        }}
        onToggleWatchlist={(anime) => {
          void handleToggleWatchlist(anime);
        }}
        onToggleFavorite={(anime) => {
          void handleToggleFavorite(anime);
        }}
        watchlistIds={watchlistIds}
        favoriteIds={favoriteIds}
        ratedMap={ratedMap}
        knownAnimesMap={knownAnimesMap}
      />

      {/* Global Floating Toast Notification (Visible on Mobile, Tablet, Desktop & TV) */}
      {toastMessage && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-20 md:bottom-6 right-3 left-3 sm:left-auto sm:max-w-md z-[100] flex items-center justify-between gap-2.5 px-4 py-3 rounded-2xl bg-[#131724]/95 backdrop-blur-md border border-indigo-500/40 text-xs sm:text-sm text-indigo-100 shadow-2xl animate-in fade-in slide-in-from-bottom-3 duration-200"
        >
          <div className="flex items-center gap-2.5 min-w-0">
            <Sparkles className="w-4 h-4 text-indigo-400 shrink-0" />
            <span className="leading-snug">{toastMessage}</span>
          </div>
          <button
            type="button"
            onClick={() => setToastMessage(null)}
            aria-label="Fechar aviso"
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-white/5 shrink-0"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
};

export default App;
