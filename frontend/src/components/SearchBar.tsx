import React, { useEffect, useRef, useState } from 'react';
import { Search, Star, X, Command, CheckCircle2 } from 'lucide-react';
import { fetchCatalog } from '../services/api';
import type { AnimePublic } from '../types';
import { handleCoverError } from '../utils/imageFallback';

interface SearchBarProps {
  onQuickRate: (anime: AnimePublic, rating: number) => void;
  onSelectAnime?: (anime: AnimePublic) => void;
  ratedMap?: Record<number, number>;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  onQuickRate,
  onSelectAnime,
  ratedMap,
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<AnimePublic[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Keyboard shortcut Ctrl+K or Cmd+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
        setIsOpen(true);
      } else if (e.key === 'Escape') {
        setIsOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // 300ms Debounced search query
  useEffect(() => {
    const trimmed = query.trim();
    if (!trimmed) {
      setResults([]);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    const timer = setTimeout(async () => {
      try {
        const res = await fetchCatalog({ q: trimmed, limit: 8 });
        setResults(res.items);
      } finally {
        setIsLoading(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  return (
    <div ref={containerRef} className="relative w-full max-w-xl">
      <div className="relative flex items-center">
        <Search className="w-4 h-4 text-slate-400 absolute left-3.5 pointer-events-none" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onFocus={() => setIsOpen(true)}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          placeholder="Buscar animes, gêneros, temas..."
          aria-label="Buscar animes"
          className="w-full bg-[#171b26] text-sm text-slate-100 placeholder-slate-400 pl-10 pr-10 sm:pr-20 py-2.5 rounded-xl border border-white/10 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/40 transition-all"
        />
        {query ? (
          <button
            type="button"
            onClick={() => {
              setQuery('');
              setResults([]);
            }}
            aria-label="Limpar busca"
            className="absolute right-3 sm:right-14 text-slate-400 hover:text-slate-200 p-1"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        ) : null}
        <kbd className="hidden sm:inline-flex items-center gap-0.5 px-2 py-0.5 text-[11px] font-mono text-slate-400 bg-[#11141c] border border-white/10 rounded-md absolute right-3">
          <Command className="w-3 h-3" />K
        </kbd>
      </div>

      {/* Floating Autocomplete Palette */}
      {isOpen && query.trim().length > 0 && (
        <div
          role="listbox"
          className="absolute left-0 right-0 mt-2 bg-[#11141c]/95 backdrop-blur-md border border-[#232936] rounded-xl shadow-2xl overflow-hidden z-50"
        >
          {isLoading ? (
            <div className="p-4 text-xs text-slate-400 flex items-center justify-center gap-2">
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-ping" />
              <span>Consultando catálogo...</span>
            </div>
          ) : results.length === 0 ? (
            <div className="p-4 text-xs text-slate-400 text-center">
              Nenhum anime encontrado para "{query}".
            </div>
          ) : (
            <ul className="divide-y divide-[#232936] max-h-80 overflow-y-auto">
              {results.map((anime) => (
                <li
                  key={anime.id}
                  onClick={() => {
                    if (onSelectAnime) {
                      onSelectAnime(anime);
                      setIsOpen(false);
                    }
                  }}
                  className="p-3 hover:bg-[#1f2433] transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 sm:gap-3 cursor-pointer group"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <img
                      src={anime.cover_image}
                      alt={anime.title_romaji}
                      onError={(e) => handleCoverError(e, anime.id)}
                      className="w-10 h-14 object-cover rounded-md shrink-0 bg-slate-800 group-hover:scale-105 transition-transform"
                    />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold text-white group-hover:text-indigo-400 transition-colors truncate">
                        {anime.title_english || anime.title_romaji}
                      </p>
                      <p className="text-xs text-slate-400 truncate">
                        {anime.season_year ?? 'Anime'} • {anime.genres.slice(0, 3).join(' • ')}
                      </p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {anime.tags.slice(0, 2).map((tag) => (
                          <span
                            key={tag}
                            className="px-1.5 py-0.5 text-[10px] rounded bg-slate-800/80 text-slate-300"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Quick Rating Actions or Locked Evaluated Badge */}
                  <div className="flex items-center justify-end gap-1 shrink-0 pt-1 sm:pt-0 border-t sm:border-t-0 border-white/5" onClick={(e) => e.stopPropagation()}>
                    {ratedMap && ratedMap[anime.id] !== undefined ? (
                      <span className="px-2.5 py-1 rounded-md text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center gap-1 shadow-sm">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                        <span>Avaliado ({ratedMap[anime.id]}/10)</span>
                      </span>
                    ) : (
                      [7, 8, 9, 10].map((score) => (
                        <button
                          key={score}
                          type="button"
                          onClick={() => {
                            onQuickRate(anime, score);
                            setIsOpen(false);
                          }}
                          className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-indigo-500/10 hover:bg-indigo-500/30 text-indigo-300 border border-indigo-500/20 flex items-center gap-0.5 transition-colors"
                          title={`Avaliar com nota ${score}`}
                        >
                          <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                          {score}
                        </button>
                      ))
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
};
