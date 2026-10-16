import React from 'react';
import { Target, Sparkles } from 'lucide-react';
import { AnimeCard } from './AnimeCard';
import type { AnimePublic } from '../types';

interface TrackAffinityProps {
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
  lastSafeReason?: string | null;
}

export const TrackAffinity: React.FC<TrackAffinityProps> = ({
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
  lastSafeReason,
}) => {
  return (
    <section
      data-testid="track-affinity"
      className="p-5 rounded-2xl bg-[#11141c]/90 border border-emerald-500/20 shadow-xl space-y-4"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/5 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Target className="w-4 h-4" />
            </span>
            <h2 className="text-lg font-semibold text-white tracking-tight">
              Para o Seu Perfil (80% Afinidade)
            </h2>
            <span className="px-2 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Exploitation • Cosseno
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Alta compatibilidade direta com o seu vetor de preferências normalizado L2.
          </p>
        </div>

        <span className="text-xs font-mono text-emerald-400 shrink-0">
          {animes.length} recomendações diretas
        </span>
      </div>

      {/* Non-spoiler Semantic Justification Banner if available */}
      {lastSafeReason && (
        <div className="px-3.5 py-2.5 rounded-xl bg-emerald-950/30 border border-emerald-500/20 flex items-center gap-2.5 text-xs text-emerald-200">
          <Sparkles className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>
            <strong>Insight do Algoritmo:</strong> {lastSafeReason}
          </span>
        </div>
      )}

      {animes.length === 0 ? (
        <div className="py-8 text-center text-sm text-slate-400">
          Avalie alguns animes abaixo para gerar recomendações personalizadas de afinidade.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 sm:gap-4">
          {animes.map((anime) => (
            <AnimeCard
              key={anime.id}
              anime={anime}
              trackType="affinity"
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
