import React from 'react';
import { Compass } from 'lucide-react';
import { AnimeCard } from './AnimeCard';
import type { AnimePublic } from '../types';

interface TrackDiscoveryProps {
  animes: AnimePublic[];
  onRate: (anime: AnimePublic, rating: number, track: 'affinity' | 'discovery' | 'onboarding') => void;
  onSelect?: (anime: AnimePublic) => void;
  onSkip?: (anime: AnimePublic) => void;
  onDrop?: (anime: AnimePublic, track: 'affinity' | 'discovery' | 'onboarding') => void;
  onToggleFavorite?: (anime: AnimePublic) => void;
  onToggleWatchlist?: (anime: AnimePublic) => void;
  ratedMap: Record<number, number>;
  favoriteIds?: Set<number>;
  watchlistIds?: Set<number>;
  droppedIds?: Set<number>;
}

export const TrackDiscovery: React.FC<TrackDiscoveryProps> = ({
  animes,
  onRate,
  onSelect,
  onSkip,
  onDrop,
  onToggleFavorite,
  onToggleWatchlist,
  ratedMap,
  favoriteIds,
  watchlistIds,
  droppedIds,
}) => {
  return (
    <section
      data-testid="track-discovery"
      className="p-5 rounded-2xl bg-[#11141c]/90 border border-purple-500/20 shadow-xl space-y-4"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/5 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20">
              <Compass className="w-4 h-4" />
            </span>
            <h2 className="text-lg font-semibold text-white tracking-tight">
              Radar de Descoberta (20% Exploração MMR)
            </h2>
            <span className="px-2 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/20">
              Exploration • λ = 0.50
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Obras selecionadas via Maximal Marginal Relevance para romper bolhas de filtro mantendo qualidade.
          </p>
        </div>

        <span className="text-xs font-mono text-purple-400 shrink-0">
          {animes.length} descobertas diversas
        </span>
      </div>

      {animes.length === 0 ? (
        <div className="py-6 text-center text-sm text-slate-400">
          Nenhum item adicional na trilha de descoberta no momento.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 sm:gap-4">
          {animes.map((anime) => (
            <AnimeCard
              key={anime.id}
              anime={anime}
              trackType="discovery"
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
      )}
    </section>
  );
};
