import React, { useEffect, useState } from 'react';
import { Sparkles, X, Send, CheckCircle2, Ban, RotateCcw } from 'lucide-react';
import type { AnimePublic, SemanticTuningResponse } from '../types';

interface QualitativeModalProps {
  anime: AnimePublic | null;
  rating: number;
  mode?: 'like' | 'drop';
  isOpen: boolean;
  isSubmitting: boolean;
  tuningResult: SemanticTuningResponse | null;
  onClose: () => void;
  onSubmitFeedback: (animeId: number, feedback: string) => void;
  onCancelRating?: (animeId: number) => void;
}

const QUICK_SUGGESTION_PILLS = [
  'Plot twist inesperado e narrativa envolvente',
  'Trilha sonora marcante e direção impecável',
  'Desenvolvimento psicológico profundo dos personagens',
  'Animação de combate espetacular e ritmo acelerado',
  'Construção de mundo fascinante e atmosfera única',
];

const QUICK_DROP_PILLS = [
  'Ritmo muito lento e arrastado nos primeiros episódios',
  'Protagonista irritante e excesso de clichês shounen',
  'Narrativa confusa de viagem no tempo sem foco claro',
  'Comédia forçada que quebrou a tensão dramática',
  'Tom excessivamente trágico e pesado para o meu gosto',
];

export const QualitativeModal: React.FC<QualitativeModalProps> = ({
  anime,
  rating,
  mode = 'like',
  isOpen,
  isSubmitting,
  tuningResult,
  onClose,
  onSubmitFeedback,
  onCancelRating,
}) => {
  const [feedback, setFeedback] = useState('');

  useEffect(() => {
    setFeedback('');
  }, [anime?.id, isOpen, mode]);

  if (!isOpen || !anime) return null;

  const displayTitle = anime.title_english || anime.title_romaji;
  const charCount = feedback.length;
  const isDropMode = mode === 'drop';
  const activePills = isDropMode ? QUICK_DROP_PILLS : QUICK_SUGGESTION_PILLS;

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!feedback.trim() || isSubmitting) return;
    onSubmitFeedback(anime.id, feedback.trim().slice(0, 300));
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="qualitative-modal-title"
      className="fixed inset-0 z-[90] flex items-center justify-center bg-black/80 backdrop-blur-md p-3 sm:p-4 animate-in fade-in duration-150"
    >
      <div className="w-full max-w-lg max-h-[90vh] overflow-y-auto bg-[#11141c] border border-[#232936] rounded-2xl shadow-2xl animate-in fade-in zoom-in-95 duration-150">
        {/* Modal Header */}
        <div className="px-4 sm:px-5 py-3.5 sm:py-4 bg-[#171b26] border-b border-[#232936] flex items-center justify-between sticky top-0 z-10">
          <div className="flex items-center gap-2.5 min-w-0">
            <div
              className={`p-2 rounded-lg border shrink-0 ${
                isDropMode
                  ? 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                  : 'bg-indigo-500/15 text-indigo-400 border-indigo-500/30'
              }`}
            >
              {isDropMode ? <Ban className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
            </div>
            <div className="min-w-0">
              <h3 id="qualitative-modal-title" className="text-sm font-bold text-white truncate">
                {isDropMode
                  ? 'Análise de Abandono (Drop) • Ajuste Inteligente'
                  : 'Calibração Semântica do Algoritmo'}
              </h3>
              <p className="text-xs text-slate-400 truncate">
                {isDropMode ? (
                  <>
                    Você decidiu dropar <strong>{displayTitle}</strong>
                  </>
                ) : (
                  <>
                    Você avaliou <strong>{displayTitle}</strong> com nota{' '}
                    <span className="font-mono text-amber-400 font-bold">{rating}/10</span>
                  </>
                )}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              setFeedback('');
              onClose();
            }}
            aria-label="Fechar modal"
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-white/5 shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-4 sm:p-5 space-y-4">
          {tuningResult ? (
            <div className="space-y-4">
              <div
                className={`p-4 rounded-xl border space-y-2 ${
                  isDropMode
                    ? 'bg-rose-950/30 border-rose-500/30'
                    : 'bg-emerald-950/30 border-emerald-500/30'
                }`}
              >
                <div
                  className={`flex items-center gap-2 text-xs font-semibold ${
                    isDropMode ? 'text-rose-300' : 'text-emerald-400'
                  }`}
                >
                  <CheckCircle2 className="w-4 h-4" />
                  <span>
                    {isDropMode
                      ? 'Penalização Semântica Aplicada (Projeção L2)'
                      : 'Pesos Semânticos Calibrados (α = 0.25 + Projeção L2)'}
                  </span>
                </div>
                <p className="text-xs text-slate-200 leading-relaxed">
                  {tuningResult.public_safe_reason}
                </p>

                {tuningResult.identified_tags.length > 0 && (
                  <div className="pt-2 flex flex-wrap gap-1.5">
                    {tuningResult.identified_tags.map((t) => (
                      <span
                        key={t.tag_name}
                        className={`px-2 py-0.5 rounded text-[11px] font-mono border ${
                          t.impact_multiplier < 0
                            ? 'bg-rose-950/60 text-rose-300 border-rose-500/30'
                            : 'bg-emerald-950/60 text-emerald-300 border-emerald-500/30'
                        }`}
                      >
                        {t.tag_name}: {t.impact_multiplier > 0 ? `+${t.impact_multiplier}` : t.impact_multiplier}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => {
                    setFeedback('');
                    onClose();
                  }}
                  className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
                >
                  Concluir e Ver Vitrine
                </button>
              </div>
            </div>
          ) : (
            <form onSubmit={handleFormSubmit} className="space-y-4">
              <div>
                <label
                  htmlFor="gemini-feedback-input"
                  className="block text-sm font-semibold text-white mb-1"
                >
                  {isDropMode
                    ? 'Por que você decidiu dropar este anime?'
                    : 'O que mais você gostou nesse anime?'}
                </label>
                <p className="text-xs text-slate-400 mb-3">
                  {isDropMode
                    ? 'Escreva o que incomodou você na obra. Nosso algoritmo identificará os elementos responsáveis pelo drop e reduzirá o peso deles nas recomendações.'
                    : 'Nosso algoritmo interpreta sua resposta livre e ajusta os pesos do seu perfil sem expor spoilers.'}
                </p>

                {/* Quick Suggestion Pills */}
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {activePills.map((pill) => (
                    <button
                      key={pill}
                      type="button"
                      onClick={() => setFeedback(pill)}
                      className={`px-2.5 py-1 rounded-full text-[11px] bg-[#171b26] text-slate-300 border border-white/10 text-left transition-colors ${
                        isDropMode
                          ? 'hover:bg-rose-600/20 hover:text-rose-300 hover:border-rose-500/30'
                          : 'hover:bg-indigo-600/20 hover:text-indigo-300 hover:border-indigo-500/30'
                      }`}
                    >
                      + {pill}
                    </button>
                  ))}
                </div>

                <textarea
                  id="gemini-feedback-input"
                  rows={3}
                  maxLength={300}
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value.slice(0, 300))}
                  placeholder={
                    isDropMode
                      ? 'Ex: Achei o ritmo muito lento e arrastado, e o protagonista é muito clichê...'
                      : 'Ex: Ele teve um desenvolvimento psicológico incrível e reviravoltas marcantes...'
                  }
                  className="w-full rounded-xl bg-[#0a0c10] border border-white/10 focus:border-indigo-500/60 focus:outline-none p-3 text-sm text-slate-100 placeholder-slate-500 resize-none"
                />
                <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1">
                  <span>Protegido por sanitizador em 4 camadas</span>
                  <span className="font-mono">{charCount}/300</span>
                </div>
              </div>

              {isSubmitting ? (
                <div className="p-3 rounded-xl bg-indigo-950/30 border border-indigo-500/30 flex items-center justify-center gap-2.5 text-xs text-indigo-300">
                  <Sparkles className="w-4 h-4 animate-spin text-indigo-400" />
                  <span>
                    {isDropMode
                      ? 'Analisando o motivo do drop e refinando recomendações...'
                      : 'Calibrando pesos semânticos do perfil...'}
                  </span>
                </div>
              ) : (
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-2 border-t border-white/5">
                  {!isDropMode && onCancelRating ? (
                    <button
                      type="button"
                      onClick={() => {
                        setFeedback('');
                        onCancelRating(anime.id);
                        onClose();
                      }}
                      data-testid="cancel-rating-undo-btn"
                      className="px-3 py-2 rounded-xl text-xs font-semibold text-rose-300 hover:text-white bg-rose-950/40 hover:bg-rose-600/80 border border-rose-500/30 flex items-center justify-center gap-1.5 transition-colors"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>Cancelar Avaliação (Desfazer Nota)</span>
                    </button>
                  ) : (
                    <div />
                  )}

                  <div className="flex items-center justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setFeedback('');
                        onClose();
                      }}
                      className="px-3.5 py-2 rounded-xl text-xs font-medium text-slate-300 hover:text-white bg-[#171b26] hover:bg-white/10 border border-white/10 transition-colors"
                    >
                      {isDropMode ? 'Cancelar' : 'Pular Comentário (Manter Nota)'}
                    </button>
                    <button
                      type="submit"
                      disabled={!feedback.trim()}
                      className={`px-4 py-2 rounded-xl text-xs font-semibold disabled:opacity-40 text-white flex items-center gap-1.5 transition-colors ${
                        isDropMode
                          ? 'bg-rose-600 hover:bg-rose-500'
                          : 'bg-indigo-600 hover:bg-indigo-500'
                      }`}
                    >
                      <Send className="w-3.5 h-3.5" />
                      <span>
                        {isDropMode ? 'Registrar Motivo do Drop' : 'Calibrar Recomendações'}
                      </span>
                    </button>
                  </div>
                </div>
              )}
            </form>
          )}
        </div>
      </div>
    </div>
  );
};
