import React, { useState } from 'react';
import { Flame, Sword, Compass, Smile, Theater, Sparkles, Heart, Ghost, Cpu, ChevronDown } from 'lucide-react';
import { AnimeCard } from './AnimeCard';
import type { AnimePublic } from '../types';

interface OnboardingShelfProps {
  animes: AnimePublic[];
  selectedGenre: string | null;
  onSelectGenre: (genre: string | null) => void;
  onRate: (anime: AnimePublic, rating: number, track: 'affinity' | 'discovery' | 'onboarding') => void;
  onSelect?: (anime: AnimePublic) => void;
  onSkip?: (anime: AnimePublic) => void;
  onDrop?: (anime: AnimePublic, track: 'affinity' | 'discovery' | 'onboarding') => void;
  onToggleFavorite?: (anime: AnimePublic) => void;
  onToggleWatchlist?: (anime: AnimePublic) => void;
  onExploreAll?: () => void;
  ratedMap: Record<number, number>;
  favoriteIds?: Set<number>;
  watchlistIds?: Set<number>;
  droppedIds?: Set<number>;
}

const POPULAR_GENRES = [
  { label: 'Ação', value: 'Action', icon: Sword },
  { label: 'Aventura', value: 'Adventure', icon: Compass },
  { label: 'Comédia', value: 'Comedy', icon: Smile },
  { label: 'Drama', value: 'Drama', icon: Theater },
  { label: 'Fantasia', value: 'Fantasy', icon: Sparkles },
  { label: 'Sci-Fi', value: 'Sci-Fi', icon: Cpu },
  { label: 'Romance', value: 'Romance', icon: Heart },
  { label: 'Suspense', value: 'Suspense', icon: Ghost },
];

export const OnboardingShelf: React.FC<OnboardingShelfProps> = ({
  animes,
  selectedGenre,
  onSelectGenre,
  onRate,
  onSelect,
  onSkip,
  onDrop,
  onToggleFavorite,
  onToggleWatchlist,
  onExploreAll,
  ratedMap,
  favoriteIds,
  watchlistIds,
  droppedIds,
}) => {
  const [visibleCount, setVisibleCount] = useState(6);
  const displayedAnimes = animes.slice(0, visibleCount);

  return (
    <section className="space-y-6">
      {/* Popular Genres Pills inspired by AnimeNest reference */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-white tracking-tight">
            Gêneros populares
          </h2>
          {selectedGenre && (
            <button
              type="button"
              onClick={() => onSelectGenre(null)}
              className="text-xs text-indigo-400 hover:text-indigo-300"
            >
              Limpar filtro ({selectedGenre})
            </button>
          )}
        </div>
        <div className="flex items-center gap-2.5 overflow-x-auto pb-1">
          {POPULAR_GENRES.map((g) => {
            const Icon = g.icon;
            const isSelected = selectedGenre === g.value;
            return (
              <button
                key={g.value}
                type="button"
                onClick={() => {
                  onSelectGenre(isSelected ? null : g.value);
                  setVisibleCount(6);
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium border transition-all shrink-0 ${
                  isSelected
                    ? 'bg-indigo-600 text-white border-indigo-500 shadow-md shadow-indigo-950/50'
                    : 'bg-[#171b26] text-slate-300 border-white/10 hover:bg-[#1f2433] hover:text-white'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{g.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Onboarding Popular Grid */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-amber-400" />
              <h2 className="text-lg font-semibold text-white tracking-tight">
                Catálogo Popular & Cold-Start Rápido
              </h2>
            </div>
            <p className="text-xs text-slate-400">
              Marque obras que você já assistiu para calibrar instantaneamente seu vetor de preferências, ou pule as que não tem interesse.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400 shrink-0">
            Exibindo {displayedAnimes.length} de {animes.length} títulos
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {displayedAnimes.map((anime) => (
            <AnimeCard
              key={anime.id}
              anime={anime}
              trackType="onboarding"
              onRate={onRate}
              onSelect={onSelect}
              onSkip={onSkip}
              onDrop={onDrop}
              onToggleFavorite={onToggleFavorite}
              onToggleWatchlist={onToggleWatchlist}
              userRating={ratedMap[anime.id]}
              isFavorite={favoriteIds?.has(anime.id)}
              isInWatchlist={watchlistIds?.has(anime.id)}
              isDropped={droppedIds?.has(anime.id)}
            />
          ))}
        </div>

        {animes.length > 6 && (
          <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
            {visibleCount < Math.min(animes.length, 24) && (
              <button
                type="button"
                onClick={() => setVisibleCount((prev) => Math.min(prev + 6, animes.length))}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#171b26] hover:bg-[#1f2433] text-slate-200 border border-white/10 flex items-center gap-1.5 transition-colors"
              >
                <ChevronDown className="w-3.5 h-3.5 text-indigo-400" />
                <span>Ver mais títulos (+6)</span>
              </button>
            )}
            {onExploreAll && (
              <button
                type="button"
                onClick={onExploreAll}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 flex items-center gap-1.5 transition-colors"
              >
                <Compass className="w-3.5 h-3.5" />
                <span>Explorar Catálogo Completo ({animes.length})</span>
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  );
};
