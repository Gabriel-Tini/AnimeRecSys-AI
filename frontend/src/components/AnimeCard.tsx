import React, { useState } from 'react';
import {
  Star,
  Sparkles,
  Compass,
  CheckCircle2,
  Heart,
  Bookmark,
  EyeOff,
  Ban,
} from 'lucide-react';
import type { AnimePublic } from '../types';
import { FALLBACK_ANIME_POSTER, handleCoverError } from '../utils/imageFallback';

interface AnimeCardProps {
  anime: AnimePublic;
  trackType?: 'affinity' | 'discovery' | 'onboarding';
  onRate: (anime: AnimePublic, rating: number, track: 'affinity' | 'discovery' | 'onboarding') => void;
  onSelect?: (anime: AnimePublic) => void;
  onSkip?: (anime: AnimePublic) => void;
  onDrop?: (anime: AnimePublic, track: 'affinity' | 'discovery' | 'onboarding') => void;
  onToggleFavorite?: (anime: AnimePublic) => void;
  onToggleWatchlist?: (anime: AnimePublic) => void;
  userRating?: number;
  isFavorite?: boolean;
  isInWatchlist?: boolean;
  isDropped?: boolean;
}

export const AnimeCard: React.FC<AnimeCardProps> = ({
  anime,
  trackType = 'affinity',
  onRate,
  onSelect,
  onSkip,
  onDrop,
  onToggleFavorite,
  onToggleWatchlist,
  userRating,
  isFavorite = false,
  isInWatchlist = false,
  isDropped = false,
}) => {
  const [selectedScore, setSelectedScore] = useState<number>(userRating ?? 9);
  const [showRatingBar, setShowRatingBar] = useState(false);

  const matchPct =
    anime.compatibility_score != null
      ? Math.round(anime.compatibility_score * 100)
      : anime.average_score ?? 88;

  const displayTitle = anime.title_english || anime.title_romaji;

  return (
    <div
      tabIndex={0}
      role="button"
      aria-label={`Ver detalhes de ${displayTitle}`}
      data-testid={`anime-card-${anime.id}`}
      onMouseEnter={() => setShowRatingBar(true)}
      onMouseLeave={() => setShowRatingBar(false)}
      onFocus={() => setShowRatingBar(true)}
      onBlur={() => setShowRatingBar(false)}
      onClick={() => onSelect?.(anime)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          if (e.target === e.currentTarget) {
            e.preventDefault();
            onSelect?.(anime);
          }
        }
      }}
      className="group relative bg-[#171b26] rounded-xl border border-white/5 hover:border-indigo-500/40 focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-500/80 focus-visible:border-indigo-400 focus-visible:scale-105 transition-all duration-200 hover:scale-[1.02] flex flex-col overflow-hidden shadow-lg cursor-pointer select-none"
    >
      {/* 3:4 Aspect Ratio Poster Container */}
      <div className="relative aspect-[3/4] w-full bg-slate-900 overflow-hidden">
        <img
          src={anime.cover_image || FALLBACK_ANIME_POSTER}
          alt={displayTitle}
          loading="lazy"
          onError={(e) => handleCoverError(e, anime.id)}
          className="w-full h-full object-cover object-center group-hover:scale-105 transition-transform duration-300"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[#0a0c10] via-transparent to-black/30" />

        {/* Track Badge (Top Left) */}
        <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5">
          {trackType === 'affinity' && (
            <span className="px-2 py-0.5 rounded-md text-[11px] font-mono font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 backdrop-blur-md flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              {matchPct}% Match
            </span>
          )}
          {trackType === 'discovery' && (
            <span className="px-2 py-0.5 rounded-md text-[11px] font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30 backdrop-blur-md flex items-center gap-1">
              <Compass className="w-3 h-3" />
              Exploração MMR
            </span>
          )}
          {trackType === 'onboarding' && (
            <span className="px-2 py-0.5 rounded-md text-[11px] font-mono font-medium bg-slate-900/80 text-slate-200 border border-white/10 backdrop-blur-md">
              Popular
            </span>
          )}
        </div>

        {/* Top Right Quick Actions: Favorite & Watchlist */}
        <div className="absolute top-2.5 right-2.5 flex flex-col items-end gap-1.5 z-10">
          <div className="flex items-center gap-1">
            {onToggleWatchlist && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleWatchlist(anime);
                }}
                title={isInWatchlist ? 'Remover da Minha Lista' : 'Salvar em Quero Assistir (Minha Lista)'}
                aria-label={`Minha Lista ${displayTitle}`}
                className={`p-1.5 rounded-lg backdrop-blur-md border transition-colors ${
                  isInWatchlist
                    ? 'bg-indigo-600/90 text-white border-indigo-400/50'
                    : 'bg-black/60 text-slate-300 hover:text-white border-white/10 hover:bg-black/80'
                }`}
              >
                <Bookmark className={`w-3.5 h-3.5 ${isInWatchlist ? 'fill-white' : ''}`} />
              </button>
            )}
            {onToggleFavorite && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleFavorite(anime);
                }}
                title={isFavorite ? 'Remover dos Favoritos' : 'Adicionar aos Favoritos'}
                aria-label={`Favoritar ${displayTitle}`}
                className={`p-1.5 rounded-lg backdrop-blur-md border transition-colors ${
                  isFavorite
                    ? 'bg-rose-600/90 text-white border-rose-400/50'
                    : 'bg-black/60 text-slate-300 hover:text-rose-400 border-white/10 hover:bg-black/80'
                }`}
              >
                <Heart className={`w-3.5 h-3.5 ${isFavorite ? 'fill-white' : ''}`} />
              </button>
            )}
          </div>

          {/* Rated / Dropped Badge */}
          {isDropped ? (
            <div className="px-2 py-0.5 rounded-md text-[11px] font-mono font-bold bg-rose-600/90 text-white flex items-center gap-1 shadow">
              <Ban className="w-3 h-3" />
              Dropado
            </div>
          ) : (
            userRating != null && (
              <div className="px-2 py-0.5 rounded-md text-[11px] font-mono font-bold bg-indigo-600/90 text-white flex items-center gap-1 shadow">
                <CheckCircle2 className="w-3 h-3" />
                Nota: {userRating}
              </div>
            )
          )}
        </div>

        {/* Community Score (Bottom Left over Image) */}
        <div className="absolute bottom-2.5 left-2.5 flex items-center gap-1 text-xs font-mono font-semibold text-amber-400 bg-black/60 backdrop-blur-sm px-2 py-0.5 rounded">
          <Star className="w-3.5 h-3.5 fill-amber-400" />
          <span>{anime.average_score ? (anime.average_score / 10).toFixed(1) : '8.5'}</span>
        </div>
      </div>

      {/* Card Metadata Content */}
      <div className="p-3.5 flex-1 flex flex-col justify-between space-y-2.5">
        <div>
          <h3
            className="text-sm font-semibold text-white tracking-tight line-clamp-1"
            title={displayTitle}
          >
            {displayTitle}
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            {anime.season_year ?? 2023} • {anime.episodes ? `${anime.episodes} episódios` : anime.format}
          </p>

          {/* Genres & Public Safe Tags (Strictly Spoiler-Free) */}
          <div className="flex flex-wrap gap-1 mt-2">
            {anime.genres.slice(0, 2).map((genre) => (
              <span
                key={genre}
                className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-slate-800/90 text-slate-300"
              >
                {genre}
              </span>
            ))}
            {anime.tags.slice(0, 2).map((tag) => (
              <span
                key={tag}
                className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-indigo-950/60 text-indigo-300 border border-indigo-500/20"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>

        {/* Quick 1-Click Rating + Skip + Drop Controls */}
        <div
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          onPointerDown={(e) => e.stopPropagation()}
          className="pt-2 border-t border-white/5 space-y-1.5"
        >
          {userRating != null ? (
            <div className="w-full py-1.5 px-2.5 rounded-lg text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center justify-between gap-1 shadow-sm">
              <span className="flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <span>Avaliado: {userRating}/10</span>
              </span>
              <span className="text-[10px] text-emerald-400/80 font-mono">Em Minha Lista</span>
            </div>
          ) : showRatingBar ? (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[11px] text-slate-300">
                <span>Avaliar obra:</span>
                <span className="font-mono font-bold text-indigo-400">{selectedScore}/10</span>
              </div>
              <div
                className="flex items-center gap-1"
                onClick={(e) => e.stopPropagation()}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
              >
                <input
                  type="range"
                  min={1}
                  max={10}
                  step={1}
                  value={selectedScore}
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  onPointerDown={(e) => e.stopPropagation()}
                  onTouchStart={(e) => e.stopPropagation()}
                  onChange={(e) => {
                    e.stopPropagation();
                    setSelectedScore(Number(e.target.value));
                  }}
                  aria-label={`Nota para ${displayTitle}`}
                  className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRate(anime, selectedScore, trackType);
                  }}
                  className="px-2.5 py-1 rounded-md text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-colors shrink-0"
                >
                  Confirmar
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onRate(anime, selectedScore, trackType);
              }}
              className="w-full py-1.5 px-3 rounded-lg text-xs font-medium bg-[#1f2433] hover:bg-indigo-600/20 text-slate-200 hover:text-indigo-300 border border-white/5 hover:border-indigo-500/30 transition-all flex items-center justify-center gap-1.5"
            >
              <Star className="w-3.5 h-3.5 text-amber-400" />
              <span>Marcar como Visto ({selectedScore}/10)</span>
            </button>
          )}

          {/* Secondary Action Row: Pular (Sem Interesse) & Dropar (Com Justificativa) */}
          {userRating == null && (onSkip || onDrop) && (
            <div className="grid grid-cols-2 gap-1.5">
              {onSkip && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSkip(anime);
                  }}
                  title="Pular anime que você não assistiu ou não tem interesse (não altera seu vetor L2)"
                  className="py-1 px-2 rounded-lg text-[11px] font-medium bg-[#131722] hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-white/5 flex items-center justify-center gap-1 transition-colors"
                >
                  <EyeOff className="w-3 h-3 shrink-0" />
                  <span>Pular</span>
                </button>
              )}
              {onDrop && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDrop(anime, trackType);
                  }}
                  title="Dropar obra e explicar o motivo para o nosso algoritmo ajustar as recomendações"
                  className="py-1 px-2 rounded-lg text-[11px] font-medium bg-[#131722] hover:bg-rose-950/40 text-slate-400 hover:text-rose-300 border border-white/5 hover:border-rose-500/30 flex items-center justify-center gap-1 transition-colors"
                >
                  <Ban className="w-3 h-3 shrink-0" />
                  <span>Dropar</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
