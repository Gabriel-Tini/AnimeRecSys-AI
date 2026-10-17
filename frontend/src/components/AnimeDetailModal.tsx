import React, { useState, useEffect } from 'react';
import {
  X,
  Star,
  Bookmark,
  Heart,
  Eye,
  EyeOff,
  AlertTriangle,
  Compass,
  Sparkles,
  Calendar,
  Clock,
  Tv,
  CheckCircle,
  ExternalLink,
  Ban,
  Trash2,
} from 'lucide-react';
import type { AnimePublic } from '../types';
import { getAnimeSynopsis } from '../services/api';
import {
  FALLBACK_ANIME_POSTER,
  FALLBACK_ANIME_BANNER,
  handleCoverError,
  handleBannerError,
} from '../utils/imageFallback';

interface AnimeDetailModalProps {
  anime: AnimePublic | null;
  onClose: () => void;
  onRate?: (anime: AnimePublic, score: number) => void;
  onToggleWatchlist?: (anime: AnimePublic) => void;
  onToggleFavorite?: (anime: AnimePublic) => void;
  onSkip?: (anime: AnimePublic) => void;
  onDrop?: (anime: AnimePublic) => void;
  isInWatchlist?: boolean;
  isFavorite?: boolean;
  isDropped?: boolean;
  isSkipped?: boolean;
  userRating?: number | null;
  onEditRating?: (anime: AnimePublic, currentRating: number) => void;
}

export const AnimeDetailModal: React.FC<AnimeDetailModalProps> = ({
  anime,
  onClose,
  onRate,
  onToggleWatchlist,
  onToggleFavorite,
  onSkip,
  onDrop,
  isInWatchlist = false,
  isFavorite = false,
  isDropped = false,
  isSkipped = false,
  userRating = null,
  onEditRating,
}) => {
  // If anime has spoiler tags, start with synopsis protected/hidden; otherwise start revealed
  const [showSynopsis, setShowSynopsis] = useState<boolean>(!anime?.has_spoiler_tags);
  const [selectedScore, setSelectedScore] = useState<number>(userRating ?? 9);
  const [descriptionPt, setDescriptionPt] = useState<string>(anime?.description || '');
  const [hasSpoilers, setHasSpoilers] = useState<boolean>(Boolean(anime?.has_spoiler_tags));
  const [isLoadingSynopsis, setIsLoadingSynopsis] = useState<boolean>(false);

  // Sync state when anime changes
  useEffect(() => {
    if (anime) {
      setDescriptionPt(anime.description || '');
      setHasSpoilers(Boolean(anime.has_spoiler_tags));
      setShowSynopsis(!anime.has_spoiler_tags);
      if (userRating) setSelectedScore(userRating);

      // Always fetch official Portuguese synopsis and spoiler status from backend
      setIsLoadingSynopsis(true);
      getAnimeSynopsis(anime.id)
        .then((res) => {
          if (res.synopsis_pt) {
            setDescriptionPt(res.synopsis_pt);
          }
          if (res.has_spoiler_tags != null) {
            setHasSpoilers(Boolean(res.has_spoiler_tags));
            if (res.has_spoiler_tags) {
              setShowSynopsis(false);
            }
          }
        })
        .catch(() => {})
        .finally(() => setIsLoadingSynopsis(false));
    }
  }, [anime, userRating]);

  // Handle escape key to close modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!anime) return null;

  const displayTitle = anime.title_english || anime.title_romaji;
  const secondaryTitle = anime.title_english ? anime.title_romaji : null;

  // Clean html tags from description if present
  const cleanDescription = (descriptionPt || anime.description || '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/\n\s*\n+/g, '\n\n')
    .trim();

  const matchPct =
    anime.compatibility_score != null
      ? Math.round(anime.compatibility_score * 100)
      : null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="anime-detail-title"
      data-testid="anime-detail-modal"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      className="fixed inset-0 z-[80] flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-md overflow-y-auto animate-in fade-in duration-200"
    >
      <div className="relative w-full max-w-2xl bg-[#11141c] border border-[#232936] rounded-2xl shadow-2xl overflow-hidden flex flex-col my-auto max-h-[92vh]">
        {/* Banner with Ambient Gradient */}
        <div className="relative h-44 sm:h-52 w-full bg-slate-900 shrink-0 overflow-hidden">
          {anime.banner_image ? (
            <img
              src={anime.banner_image}
              alt=""
              onError={(e) => handleBannerError(e, anime.id)}
              className="w-full h-full object-cover object-center"
            />
          ) : (
            <div
              className="w-full h-full bg-cover bg-center opacity-40 blur-sm scale-110"
              style={{ backgroundImage: `url(${anime.cover_image || FALLBACK_ANIME_POSTER})` }}
            />
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-[#11141c] via-[#11141c]/60 to-black/30" />

          {/* Close button */}
          <button
            type="button"
            onClick={onClose}
            aria-label="Fechar detalhes"
            className="absolute top-3.5 right-3.5 p-2 rounded-xl bg-black/60 hover:bg-black/90 text-slate-300 hover:text-white border border-white/10 backdrop-blur-md transition-all z-20"
          >
            <X className="w-5 h-5" />
          </button>

          {/* Poster Floating Over Banner */}
          <div className="absolute -bottom-2 left-6 flex items-end gap-4 z-10">
            <div className="w-24 sm:w-28 aspect-[3/4] rounded-xl overflow-hidden border-2 border-[#232936] shadow-2xl bg-slate-800 shrink-0">
              <img
                src={anime.cover_image || FALLBACK_ANIME_POSTER}
                alt={displayTitle}
                onError={(e) => handleCoverError(e, anime.id)}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="pb-4 space-y-1 pr-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  {anime.format || 'TV'}
                </span>
                {anime.season_year && (
                  <span className="flex items-center gap-1 text-xs text-slate-300 font-mono">
                    <Calendar className="w-3 h-3 text-slate-400" />
                    {anime.season_year}
                  </span>
                )}
                {anime.status && (
                  <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-white/5 border border-white/10 text-slate-300">
                    {anime.status === 'FINISHED'
                      ? 'Finalizado'
                      : anime.status === 'RELEASING'
                      ? 'Em exibição'
                      : anime.status === 'NOT_YET_RELEASED'
                      ? 'Não Lançado'
                      : anime.status === 'CANCELLED'
                      ? 'Cancelado'
                      : anime.status === 'HIATUS'
                      ? 'Em Hiato'
                      : anime.status}
                  </span>
                )}
                {matchPct !== null && (
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                    <Sparkles className="w-3 h-3" />
                    {matchPct}% Match
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Scrollable Body Content */}
        <div className="p-6 pt-5 space-y-5 overflow-y-auto">
          {/* Title Header */}
          <div className="space-y-1">
            <div className="flex items-start justify-between gap-3">
              <h2
                id="anime-detail-title"
                className="text-xl sm:text-2xl font-bold text-white tracking-tight"
              >
                {displayTitle}
              </h2>
              <a
                href={`https://anilist.co/anime/${anime.id}`}
                target="_blank"
                rel="noreferrer"
                title="Ver página oficial na AniList"
                className="text-slate-400 hover:text-indigo-400 p-1.5 rounded-lg hover:bg-white/5 transition-colors shrink-0"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
            {secondaryTitle && (
              <p className="text-xs text-slate-400 font-mono">{secondaryTitle}</p>
            )}
          </div>

          {/* Quick Metrics Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-3 rounded-xl bg-[#171b26] border border-white/5 text-xs">
            <div className="space-y-0.5">
              <span className="text-[11px] text-slate-400 block">Avaliação Média</span>
              <div className="flex items-center gap-1 font-mono font-semibold text-amber-400">
                <Star className="w-3.5 h-3.5 fill-amber-400" />
                <span>{anime.average_score ? `${anime.average_score}%` : 'N/A'}</span>
              </div>
            </div>

            <div className="space-y-0.5">
              <span className="text-[11px] text-slate-400 block">Episódios</span>
              <div className="flex items-center gap-1 font-mono font-semibold text-slate-200">
                <Tv className="w-3.5 h-3.5 text-slate-400" />
                <span>{anime.episodes ? `${anime.episodes} eps` : 'Desconhecido'}</span>
              </div>
            </div>

            <div className="space-y-0.5">
              <span className="text-[11px] text-slate-400 block">Duração</span>
              <div className="flex items-center gap-1 font-mono font-semibold text-slate-200">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                <span>{anime.duration ? `${anime.duration} min/ep` : '~24 min'}</span>
              </div>
            </div>

            <div className="space-y-0.5">
              <span className="text-[11px] text-slate-400 block">Popularidade</span>
              <div className="flex items-center gap-1 font-mono font-semibold text-indigo-300">
                <Compass className="w-3.5 h-3.5 text-indigo-400" />
                <span>{anime.popularity ? `${(anime.popularity / 1000).toFixed(0)}k fãs` : 'Popular'}</span>
              </div>
            </div>
          </div>

          {/* Genres and Safe Tags */}
          <div className="space-y-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              Gêneros & Elementos Principais
            </span>
            <div className="flex flex-wrap gap-1.5">
              {anime.genres.map((genre) => (
                <span
                  key={genre}
                  className="px-2.5 py-1 rounded-lg text-xs font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
                >
                  {genre}
                </span>
              ))}
              {anime.tags.slice(0, 6).map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-slate-800 text-slate-300 border border-white/5"
                >
                  #{tag}
                </span>
              ))}
            </div>
          </div>

          {/* Synopsis with Spoiler Shielding */}
          <div className="space-y-2 pt-1 border-t border-white/5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Sinopse da Obra
              </span>
              <button
                type="button"
                onClick={() => setShowSynopsis((prev) => !prev)}
                className="flex items-center gap-1.5 text-xs font-medium text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                {showSynopsis ? (
                  <>
                    <EyeOff className="w-3.5 h-3.5" />
                    <span>Ocultar Sinopse</span>
                  </>
                ) : (
                  <>
                    <Eye className="w-3.5 h-3.5" />
                    <span>Revelar Sinopse</span>
                  </>
                )}
              </button>
            </div>

            {/* Spoiler Warning Shield if tags contain spoilers */}
            {hasSpoilers && !showSynopsis && (
              <div className="p-3.5 rounded-xl bg-amber-950/30 border border-amber-500/30 text-amber-200 text-xs space-y-2">
                <div className="flex items-center gap-2 font-semibold text-amber-400">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>Proteção contra Spoilers Ativa</span>
                </div>
                <p className="text-slate-300 leading-relaxed">
                  Esta obra possui reviravoltas ou revelações cruciais no enredo. Para preservar sua experiência de surpresa, a sinopse detalhada foi ocultada.
                </p>
                <button
                  type="button"
                  onClick={() => setShowSynopsis(true)}
                  className="px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 font-medium transition-colors inline-flex items-center gap-1.5"
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>Desejo ler a sinopse mesmo assim</span>
                </button>
              </div>
            )}

            {/* Synopsis Content */}
            {showSynopsis ? (
              <div className="p-4 rounded-xl bg-[#171b26]/70 border border-white/5 text-sm text-slate-300 leading-relaxed whitespace-pre-line max-h-48 overflow-y-auto">
                {isLoadingSynopsis ? (
                  <span className="text-xs text-indigo-300 italic animate-pulse">
                    Carregando sinopse em português...
                  </span>
                ) : (
                  cleanDescription || 'Nenhuma sinopse disponível no momento para esta obra.'
                )}
              </div>
            ) : !hasSpoilers ? (
              <div
                onClick={() => setShowSynopsis(true)}
                className="p-4 rounded-xl bg-[#171b26]/40 border border-white/5 text-xs text-slate-400 cursor-pointer hover:bg-[#171b26]/70 transition-colors text-center"
              >
                Sinopse recolhida. Clique para visualizar o resumo da história.
              </div>
            ) : null}
          </div>

          {/* Quick Decision Action Panel */}
          <div className="pt-3 border-t border-white/5 space-y-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              Ações Rápidas & Decisão
            </span>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              {/* Watchlist Toggle */}
              {onToggleWatchlist && (
                <button
                  type="button"
                  onClick={() => onToggleWatchlist(anime)}
                  className={`px-3 py-2.5 rounded-xl text-xs font-medium flex items-center justify-center gap-1.5 border transition-all ${
                    isInWatchlist
                      ? 'bg-indigo-600 text-white border-indigo-400 shadow-lg shadow-indigo-600/30'
                      : 'bg-[#171b26] hover:bg-[#1f2433] text-slate-200 border-white/10'
                  }`}
                >
                  <Bookmark className={`w-4 h-4 ${isInWatchlist ? 'fill-white' : ''}`} />
                  <span>{isInWatchlist ? 'Na Minha Lista' : '+ Minha Lista'}</span>
                </button>
              )}

              {/* Favorite Toggle */}
              {onToggleFavorite && (
                <button
                  type="button"
                  onClick={() => onToggleFavorite(anime)}
                  className={`px-3 py-2.5 rounded-xl text-xs font-medium flex items-center justify-center gap-1.5 border transition-all ${
                    isFavorite
                      ? 'bg-rose-600 text-white border-rose-400 shadow-lg shadow-rose-600/30'
                      : 'bg-[#171b26] hover:bg-[#1f2433] text-slate-200 border-white/10'
                  }`}
                >
                  <Heart className={`w-4 h-4 ${isFavorite ? 'fill-white' : ''}`} />
                  <span>{isFavorite ? 'Favorito' : 'Favoritar'}</span>
                </button>
              )}

              {/* Skip Toggle */}
              {onSkip && (
                <button
                  type="button"
                  onClick={() => {
                    onSkip(anime);
                    onClose();
                  }}
                  className={`px-3 py-2.5 rounded-xl text-xs font-medium flex items-center justify-center gap-1.5 border transition-all ${
                    isSkipped
                      ? 'bg-amber-600/30 text-amber-300 border-amber-500/50'
                      : 'bg-[#171b26] hover:bg-slate-800 text-slate-300 hover:text-white border-white/10'
                  }`}
                >
                  <Trash2 className="w-4 h-4" />
                  <span>{isSkipped ? 'Despular' : 'Pular (Sem Interesse)'}</span>
                </button>
              )}

              {/* Drop with Justification */}
              {onDrop && (
                <button
                  type="button"
                  onClick={() => {
                    onClose();
                    onDrop(anime);
                  }}
                  className={`px-3 py-2.5 rounded-xl text-xs font-medium flex items-center justify-center gap-1.5 border transition-all ${
                    isDropped
                      ? 'bg-rose-950/60 text-rose-300 border-rose-500/40'
                      : 'bg-rose-950/30 hover:bg-rose-900/40 text-rose-300 border-rose-500/30'
                  }`}
                >
                  <Ban className="w-4 h-4" />
                  <span>{isDropped ? 'Dropado' : 'Dropar Obra'}</span>
                </button>
              )}
            </div>

            {/* Rating Selector */}
            {onRate && (
              <div className="p-3.5 rounded-xl bg-[#171b26] border border-white/5 flex flex-col gap-3">
                {anime.status === 'NOT_YET_RELEASED' ? (
                  <div className="flex items-center gap-2.5 text-xs text-amber-300">
                    <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
                    <span>
                      Esta obra ainda não foi lançada. A avaliação estará disponível assim que a exibição começar.
                    </span>
                  </div>
                ) : userRating != null ? (
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 w-full">
                    <div className="flex items-center gap-2.5">
                      <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shrink-0">
                        <CheckCircle className="w-4 h-4" />
                      </span>
                      <div>
                        <span className="text-xs text-emerald-300 font-bold block">
                          Obra já avaliada com nota {userRating}/10
                        </span>
                        <span className="text-[11px] text-slate-400 block">
                          Novas avaliações diretas foram bloqueadas para proteger os pesos do seu perfil.
                        </span>
                      </div>
                    </div>
                    {onEditRating ? (
                      <button
                        type="button"
                        onClick={() => {
                          onClose();
                          onEditRating(anime, userRating);
                        }}
                        className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/30 flex items-center justify-center gap-1.5 transition-colors shrink-0"
                      >
                        <Star className="w-3.5 h-3.5 fill-amber-400 text-amber-400" />
                        <span>Editar Nota</span>
                      </button>
                    ) : null}
                  </div>
                ) : (
                  <div className="w-full space-y-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <Star className="w-4 h-4 text-amber-400 shrink-0" />
                        <span className="text-xs text-slate-200 font-medium">
                          Já assistiu? Toque na nota para avaliar:
                        </span>
                      </div>
                      <span className="text-[11px] font-mono text-amber-400 font-semibold">
                        1 a 10 ★
                      </span>
                    </div>
                    <div className="grid grid-cols-5 sm:grid-cols-10 gap-1.5 w-full">
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((score) => (
                        <button
                          key={score}
                          type="button"
                          onClick={() => {
                            setSelectedScore(score);
                            onClose();
                            onRate(anime, score);
                          }}
                          className={`py-2 px-1 rounded-lg text-xs font-mono font-bold transition-all flex items-center justify-center ${
                            selectedScore === score
                              ? 'bg-amber-500 text-black shadow-md'
                              : 'bg-white/5 hover:bg-amber-500/20 hover:text-amber-300 text-slate-200 border border-white/5'
                          }`}
                        >
                          {score}★
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
