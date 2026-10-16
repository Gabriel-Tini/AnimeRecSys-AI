import React, { useState } from 'react';
import {
  Sliders,
  ShieldCheck,
  Activity,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Ban,
} from 'lucide-react';
import type { UserProfileData } from '../types';

interface ProfileMetricsProps {
  profile: UserProfileData;
  spoilerTagsSet: Set<string>;
}

const OFFICIAL_GENRES_MAP: Record<string, string> = {
  action: 'Ação (Action)',
  adventure: 'Aventura (Adventure)',
  comedy: 'Comédia (Comedy)',
  drama: 'Drama',
  fantasy: 'Fantasia (Fantasy)',
  'sci-fi': 'Ficção Científica (Sci-Fi)',
  romance: 'Romance',
  mystery: 'Mistério (Mystery)',
  psychological: 'Psicológico (Psychological)',
  supernatural: 'Sobrenatural (Supernatural)',
  thriller: 'Suspense / Thriller',
  'slice of life': 'Cotidiano (Slice of Life)',
  sports: 'Esportes (Sports)',
  horror: 'Terror (Horror)',
  mecha: 'Mecha (Robôs Gigantes)',
  music: 'Música (Music)',
  ecchi: 'Ecchi',
  'mahou shoujo': 'Garotas Mágicas (Mahou Shoujo)',
};

export const ProfileMetrics: React.FC<ProfileMetricsProps> = ({
  profile,
  spoilerTagsSet,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const allEntries = Object.entries(profile.feature_weights);
  const l2Norm = Math.sqrt(allEntries.reduce((acc, [, w]) => acc + w * w, 0));

  const publicEntries = allEntries
    .filter(([tag]) => !spoilerTagsSet.has(tag.trim().toLowerCase()))
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  // Separate official genres from technical AniList sub-tags
  const genreEntries = publicEntries.filter(([tag]) =>
    Object.prototype.hasOwnProperty.call(OFFICIAL_GENRES_MAP, tag.trim().toLowerCase())
  );
  const positiveGenres = genreEntries
    .filter(([, w]) => w > 0)
    .sort((a, b) => b[1] - a[1]);
  const negativeGenres = genreEntries
    .filter(([, w]) => w < 0)
    .sort((a, b) => a[1] - b[1]);

  const maxGenreWeight = Math.max(
    0.01,
    ...genreEntries.map(([, w]) => Math.abs(w))
  );

  const blockedGenres = profile.blocked_genres ?? [];

  return (
    <section
      data-testid="profile-metrics"
      className="p-5 rounded-2xl bg-[#11141c] border border-[#232936] space-y-5"
    >
      {/* Simplified Header for Regular Users */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/5 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">
              Seus Gêneros de Preferência
            </h2>
            <p className="text-xs text-slate-400">
              Resumo dos estilos de anime que mais combinam com seu gosto atual
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => setShowAdvanced((prev) => !prev)}
          data-testid="toggle-advanced-profile"
          className="px-3.5 py-2 rounded-xl text-xs font-medium bg-[#171b26] hover:bg-[#1f2433] text-slate-200 border border-white/10 flex items-center gap-1.5 transition-colors self-start sm:self-auto"
        >
          <Sliders className="w-3.5 h-3.5 text-indigo-400" />
          <span>
            {showAdvanced ? 'Ocultar opções avançadas' : 'Mostrar opções avançadas'}
          </span>
          {showAdvanced ? (
            <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          )}
        </button>
      </div>

      {/* Main Simplified Content: Genres Only */}
      {genreEntries.length === 0 ? (
        <p className="text-xs text-slate-400 text-center py-4">
          Você ainda não avaliou animes suficientes. Marque obras como vistas ou avalie seus animes favoritos para descobrir seus gêneros principais!
        </p>
      ) : (
        <div className="space-y-4">
          {positiveGenres.length > 0 && (
            <div className="space-y-2.5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-emerald-400">
                Gêneros que Você Mais Gosta
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {positiveGenres.map(([genreKey, weight]) => {
                  const label =
                    OFFICIAL_GENRES_MAP[genreKey.trim().toLowerCase()] ?? genreKey;
                  const relativePct = Math.max(
                    12,
                    Math.min(100, Math.round((Math.abs(weight) / maxGenreWeight) * 100))
                  );
                  return (
                    <div
                      key={genreKey}
                      className="p-3 rounded-xl bg-[#171b26] border border-white/5 space-y-2"
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-white truncate">
                          {label}
                        </span>
                        <span className="text-[11px] font-medium text-emerald-400">
                          {relativePct}% afinidade
                        </span>
                      </div>
                      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          style={{ width: `${relativePct}%` }}
                          className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {(negativeGenres.length > 0 || blockedGenres.length > 0) && (
            <div className="space-y-2.5 pt-2 border-t border-white/5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-rose-400">
                Gêneros em Baixa ou Bloqueados por Você
              </h3>
              <div className="flex flex-wrap gap-2">
                {blockedGenres.map((bg) => (
                  <span
                    key={`blocked-${bg}`}
                    className="px-2.5 py-1 rounded-lg text-xs font-medium bg-rose-950/60 text-rose-200 border border-rose-500/40 flex items-center gap-1.5"
                  >
                    <Ban className="w-3 h-3 text-rose-400" />
                    <span>
                      {OFFICIAL_GENRES_MAP[bg.trim().toLowerCase()] ?? bg} (Bloqueado)
                    </span>
                  </span>
                ))}
                {negativeGenres.map(([genreKey]) => {
                  const label =
                    OFFICIAL_GENRES_MAP[genreKey.trim().toLowerCase()] ?? genreKey;
                  return (
                    <span
                      key={genreKey}
                      className="px-2.5 py-1 rounded-lg text-xs font-medium bg-[#171b26] text-rose-300 border border-rose-500/20"
                    >
                      {label} (Menor interesse)
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Collapsible Advanced Vector & Hyperparameter Diagnostics */}
      {showAdvanced && (
        <div
          data-testid="advanced-profile-section"
          className="pt-5 border-t border-[#232936] space-y-5 animate-in fade-in duration-150"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-indigo-300">
                Insights Avançados: Estado do Vetor L2, Hiperparâmetros & Tags Técnicas
              </h3>
              <p className="text-xs text-slate-400">
                Pesos normalizados na hiperesfera unitária com decaimento temporal γ ={' '}
                {profile.hyperparameters.gamma}
              </p>
            </div>

            <div className="flex items-center gap-2.5 text-xs font-mono">
              <span className="px-2.5 py-1 rounded-lg bg-[#171b26] border border-white/10 text-emerald-400 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5" />
                ||u||₂ = {l2Norm > 0 ? l2Norm.toFixed(4) : '0.0000'}
              </span>
              <span className="px-2.5 py-1 rounded-lg bg-[#171b26] border border-white/10 text-indigo-300 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5" />
                Spoilers Ocultos
              </span>
            </div>
          </div>

          {/* Hyperparameters Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-xl bg-[#171b26] border border-white/5">
              <span className="text-[11px] text-slate-400 block">Taxa Nota (η)</span>
              <span className="text-sm font-mono font-bold text-white">
                {profile.hyperparameters.eta}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-[#171b26] border border-white/5">
              <span className="text-[11px] text-slate-400 block">Impulso Semântico (α)</span>
              <span className="text-sm font-mono font-bold text-indigo-400">
                {profile.hyperparameters.alpha}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-[#171b26] border border-white/5">
              <span className="text-[11px] text-slate-400 block">Decaimento (γ)</span>
              <span className="text-sm font-mono font-bold text-emerald-400">
                {profile.hyperparameters.gamma}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-[#171b26] border border-white/5">
              <span className="text-[11px] text-slate-400 block">Diversidade MMR (λ)</span>
              <span className="text-sm font-mono font-bold text-purple-400">
                {profile.hyperparameters.lambda_mmr}
              </span>
            </div>
          </div>

          {/* All Raw Feature Weights (Genres + Technical Sub-Tags) */}
          {publicEntries.length > 0 && (
            <div className="space-y-2">
              <span className="text-xs font-mono text-slate-400 block">
                Pesos Brutos do Espaço Vetorial ({publicEntries.length} dimensões públicas ativas):
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {publicEntries.slice(0, 18).map(([tag, weight]) => {
                  const pct = Math.min(100, Math.round(Math.abs(weight) * 100));
                  const isPositive = weight >= 0;
                  return (
                    <div
                      key={tag}
                      className="p-2.5 rounded-lg bg-[#171b26] border border-white/5 space-y-1.5"
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-slate-200 truncate">{tag}</span>
                        <span
                          className={`font-mono font-semibold ${
                            isPositive ? 'text-emerald-400' : 'text-rose-400'
                          }`}
                        >
                          {weight >= 0 ? `+${weight.toFixed(3)}` : weight.toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          style={{ width: `${pct}%` }}
                          className={`h-full rounded-full ${
                            isPositive ? 'bg-emerald-500' : 'bg-rose-500'
                          }`}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
};
