import React, { useState, useEffect } from 'react';
import {
  X,
  Flame,
  Star,
  Sparkles,
  Calendar,
  Tv,
  Bookmark,
  Heart,
  ExternalLink,
  Clock,
  Layers,
  ChevronRight,
} from 'lucide-react';
import type { AnimePublic } from '../types';
import { fetchSeasonalReleases, type SeasonalAnimeItem } from '../services/api';
import { FALLBACK_ANIME_POSTER, FALLBACK_ANIME_BANNER, handleCoverError } from '../utils/imageFallback';

interface SeasonalDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectAnime: (anime: AnimePublic) => void;
  onToggleWatchlist?: (anime: AnimePublic) => void;
  onToggleFavorite?: (anime: AnimePublic) => void;
  watchlistIds?: Set<number>;
  favoriteIds?: Set<number>;
}

const SEASON_TRANSLATIONS: Record<string, string> = {
  WINTER: 'Inverno',
  SPRING: 'Primavera',
  SUMMER: 'Verão',
  FALL: 'Outono',
};

export const SeasonalDrawer: React.FC<SeasonalDrawerProps> = ({
  isOpen,
  onClose,
  onSelectAnime,
  onToggleWatchlist,
  onToggleFavorite,
  watchlistIds = new Set(),
  favoriteIds = new Set(),
}) => {
  const [seasonalItems, setSeasonalItems] = useState<SeasonalAnimeItem[]>([]);
  const [seasonName, setSeasonName] = useState<string>('Temporada Atual');
  const [seasonYear, setSeasonYear] = useState<number>(new Date().getFullYear());
  const [loading, setLoading] = useState<boolean>(true);
  const [searchFilter, setSearchFilter] = useState<string>('');

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetchSeasonalReleases()
        .then((res) => {
          if (res.items && res.items.length > 0) {
            setSeasonalItems(res.items);
            const translatedSeason = SEASON_TRANSLATIONS[res.season?.toUpperCase()] || res.season || 'Temporada';
            setSeasonName(`${translatedSeason} ${res.year}`);
            setSeasonYear(res.year);
          }
        })
        .finally(() => setLoading(false));
    }
  }, [isOpen]);

  // Handle Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const filteredItems = seasonalItems.filter((item) => {
    if (!searchFilter.trim()) return true;
    const term = searchFilter.toLowerCase();
    return (
      item.title_romaji?.toLowerCase().includes(term) ||
      item.title_english?.toLowerCase().includes(term) ||
      item.genres?.some((g) => g.toLowerCase().includes(term)) ||
      item.studio?.toLowerCase().includes(term)
    );
  });

  const convertToAnimePublic = (item: SeasonalAnimeItem): AnimePublic => ({
    id: item.id,
    title_romaji: item.title_romaji,
    title_english: item.title_english,
    title_native: item.title_native,
    cover_image: item.cover_image,
    banner_image: item.banner_image,
    format: item.format || 'TV',
    episodes: item.episodes,
    status: item.status || 'RELEASING',
    season_year: item.season_year,
    genres: item.genres || [],
    tags: [],
    description: item.description,
    average_score: item.average_score,
    has_spoiler_tags: false,
  });

  return (
    <>
      {/* Semi-transparent Backdrop with Blur over Main Page */}
      <div
        className="fixed inset-0 z-[55] bg-black/60 backdrop-blur-md transition-opacity animate-in fade-in duration-200"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Right Drawer Panel */}
      <aside
        role="dialog"
        aria-label="Novidades e Lançamentos da Temporada"
        aria-modal="true"
        className="fixed inset-y-0 right-0 z-[60] w-full max-w-md sm:max-w-lg bg-[#0e1118]/95 border-l border-white/10 backdrop-blur-2xl shadow-2xl flex flex-col transform transition-transform animate-in slide-in-from-right duration-300"
      >
        {/* Header */}
        <div className="p-4 sm:p-5 border-b border-white/10 flex items-center justify-between shrink-0 bg-[#131722]/80">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-rose-600 flex items-center justify-center text-white shadow-lg shadow-amber-500/20">
              <Flame className="w-5 h-5 fill-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base sm:text-lg font-bold text-white tracking-tight">
                  Novidades da Temporada
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  AO VIVO
                </span>
              </div>
              <p className="text-xs text-slate-400">
                {seasonName} • {seasonalItems.length} lançamentos em exibição
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Fechar novidades"
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-all border border-white/5"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Quick Search in Season */}
        <div className="px-4 py-3 border-b border-white/5 bg-[#11141c]/50">
          <input
            type="text"
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            placeholder="Filtrar lançamentos por título, estúdio ou gênero..."
            className="w-full px-3 py-2 text-xs rounded-lg bg-[#171b26] border border-white/10 text-white placeholder-slate-500 focus:outline-none focus:border-amber-500/50 transition-colors"
          />
        </div>

        {/* Anime Releases List */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-3.5 scrollbar-thin scrollbar-thumb-white/10">
          {loading ? (
            <div className="space-y-3 py-8">
              {[1, 2, 3, 4].map((n) => (
                <div
                  key={n}
                  className="h-28 rounded-xl bg-slate-800/40 border border-white/5 animate-pulse flex items-center p-3 gap-3"
                >
                  <div className="w-16 h-22 rounded-lg bg-slate-700/50 shrink-0" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-slate-700/50 rounded w-3/4" />
                    <div className="h-3 bg-slate-700/40 rounded w-1/2" />
                    <div className="h-3 bg-slate-700/30 rounded w-full" />
                  </div>
                </div>
              ))}
            </div>
          ) : filteredItems.length === 0 ? (
            <div className="text-center py-12 text-slate-400 space-y-2">
              <Sparkles className="w-8 h-8 text-slate-600 mx-auto" />
              <p className="text-sm">Nenhum lançamento encontrado para a busca.</p>
            </div>
          ) : (
            filteredItems.map((item) => {
              const displayTitle = item.title_english || item.title_romaji;
              const isInWatchlist = watchlistIds.has(item.id);
              const isFavorite = favoriteIds.has(item.id);
              const scorePct = item.average_score != null ? item.average_score : null;

              return (
                <article
                  key={item.id}
                  onClick={() => onSelectAnime(convertToAnimePublic(item))}
                  className="group relative bg-[#151924] hover:bg-[#1b202e] border border-white/5 hover:border-amber-500/40 rounded-xl p-3 flex gap-3.5 transition-all cursor-pointer shadow-md hover:shadow-xl hover:scale-[1.01]"
                >
                  {/* Poster Thumbnail */}
                  <div className="relative w-20 sm:w-24 aspect-[3/4] rounded-lg overflow-hidden bg-slate-900 shrink-0 border border-white/10">
                    <img
                      src={item.cover_image || FALLBACK_ANIME_POSTER}
                      alt={displayTitle}
                      loading="lazy"
                      onError={(e) => handleCoverError(e, item.id)}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    {scorePct !== null && (
                      <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-black/80 text-amber-400 border border-amber-500/30 flex items-center gap-0.5 shadow">
                        <Star className="w-2.5 h-2.5 fill-amber-400" />
                        {scorePct}%
                      </div>
                    )}
                  </div>

                  {/* Metadata & Synopsis Preview */}
                  <div className="flex-1 flex flex-col justify-between min-w-0">
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="text-sm font-semibold text-white group-hover:text-amber-400 transition-colors line-clamp-1">
                          {displayTitle}
                        </h3>

                        {/* Quick action buttons */}
                        <div className="flex items-center gap-1 shrink-0">
                          {onToggleWatchlist && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                onToggleWatchlist(convertToAnimePublic(item));
                              }}
                              title={isInWatchlist ? 'Remover da Minha Lista' : 'Salvar na Minha Lista'}
                              aria-label="Salvar na Minha Lista"
                              className={`p-1.5 rounded-lg border transition-all ${
                                isInWatchlist
                                  ? 'bg-indigo-600 text-white border-indigo-400'
                                  : 'bg-black/40 text-slate-400 hover:text-white border-white/10 hover:bg-black/60'
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
                                onToggleFavorite(convertToAnimePublic(item));
                              }}
                              title={isFavorite ? 'Remover dos Favoritos' : 'Adicionar aos Favoritos'}
                              aria-label="Favoritar"
                              className={`p-1.5 rounded-lg border transition-all ${
                                isFavorite
                                  ? 'bg-rose-600 text-white border-rose-400'
                                  : 'bg-black/40 text-slate-400 hover:text-rose-400 border-white/10 hover:bg-black/60'
                              }`}
                            >
                              <Heart className={`w-3.5 h-3.5 ${isFavorite ? 'fill-white' : ''}`} />
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Badges: Format & Studio */}
                      <div className="flex flex-wrap items-center gap-1.5 mt-1">
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-white/5 border border-white/10 text-slate-300">
                          {item.format || 'TV'}
                        </span>
                        {item.studio && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 font-medium truncate max-w-[140px]">
                            {item.studio}
                          </span>
                        )}
                        {item.episodes && (
                          <span className="text-[10px] font-mono text-slate-400">
                            {item.episodes} eps
                          </span>
                        )}
                      </div>

                      {/* Portuguese Synopsis Preview */}
                      {item.description && (
                        <p className="text-xs text-slate-400 line-clamp-2 mt-2 leading-relaxed">
                          {item.description.replace(/<[^>]+>/g, '').trim()}
                        </p>
                      )}
                    </div>

                    {/* Genres Pills & Airing Info */}
                    <div className="flex items-center justify-between gap-2 mt-2 pt-2 border-t border-white/5">
                      <div className="flex items-center gap-1 overflow-hidden">
                        {item.genres?.slice(0, 2).map((g) => (
                          <span
                            key={g}
                            className="text-[10px] text-slate-400 px-1.5 py-0.5 rounded bg-white/5 shrink-0"
                          >
                            {g}
                          </span>
                        ))}
                      </div>

                      <span className="text-[11px] font-medium text-amber-400/90 group-hover:text-amber-300 flex items-center gap-0.5 shrink-0">
                        Ver detalhes <ChevronRight className="w-3.5 h-3.5" />
                      </span>
                    </div>
                  </div>
                </article>
              );
            })
          )}
        </div>

        {/* Footer info */}
        <div className="p-3 border-t border-white/10 bg-[#11141c] text-center">
          <p className="text-[11px] text-slate-400 flex items-center justify-center gap-1.5">
            <Sparkles className="w-3 h-3 text-amber-400" />
            Sincronizado automaticamente com a temporada atual da AniList
          </p>
        </div>
      </aside>
    </>
  );
};
