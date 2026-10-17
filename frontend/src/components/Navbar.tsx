import React from 'react';
import {
  Compass,
  Heart,
  Home,
  Bookmark,
  User,
  Sparkles,
  RotateCcw,
  Cloud,
  LogOut,
} from 'lucide-react';
import type { GoogleUserAccount } from '../types';

interface NavbarProps {
  activeTab: string;
  onSelectTab: (tab: string) => void;
  isOnline?: boolean;
  totalRatings: number;
  onResetProfile: () => void;
  googleAccount?: GoogleUserAccount | null;
  onOpenGoogleModal?: () => void;
  onGoogleLogout?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onSelectTab,
  totalRatings,
  onResetProfile,
  googleAccount,
  onOpenGoogleModal,
  onGoogleLogout,
}) => {
  const navItems = [
    { id: 'home', label: 'Início', icon: Home },
    { id: 'explore', label: 'Explorar', icon: Compass },
    { id: 'list', label: 'Minha Lista', icon: Bookmark },
    { id: 'favorites', label: 'Favoritos', icon: Heart },
    { id: 'profile', label: 'Meu Perfil', icon: User },
  ];

  return (
    <>
      {/* Desktop Sidebar (>= 768px) */}
      <aside className="hidden md:flex md:w-60 shrink-0 bg-[#11141c]/95 border-r border-[#232936] flex-col justify-between h-screen sticky top-0 overflow-y-auto select-none z-30">
        <div className="p-5">
          {/* Brand Header inspired by AnimeNest */}
          <div className="flex items-center gap-2.5 mb-8 px-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shadow-sm">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <span className="font-bold text-base tracking-tight text-white block leading-none">
                AnimeRecSys
              </span>
              <span className="text-[11px] text-slate-400 font-mono">
                Recomendações Inteligentes
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    active
                      ? 'bg-[#1f2433] text-white border border-white/10 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#171b26]'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${active ? 'text-indigo-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Status, Google Sync & Reset Controls */}
        <div className="p-4 border-t border-[#232936] space-y-3">
          {/* Google Account Save Progress Card */}
          {googleAccount ? (
            <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-500/30 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  {googleAccount.picture ? (
                    <img
                      src={googleAccount.picture}
                      alt={googleAccount.name}
                      referrerPolicy="no-referrer"
                      className="w-6 h-6 rounded-full object-cover shrink-0 border border-emerald-400/50"
                    />
                  ) : (
                    <div className="w-6 h-6 rounded-full bg-emerald-600 text-white text-xs font-bold flex items-center justify-center shrink-0">
                      {googleAccount.name.charAt(0).toUpperCase()}
                    </div>
                  )}
                  <div className="min-w-0">
                    <span className="text-xs font-semibold text-white block truncate">
                      {googleAccount.name}
                    </span>
                    <span className="text-[10px] text-emerald-300 block truncate">
                      {googleAccount.email}
                    </span>
                  </div>
                </div>
                {onGoogleLogout && (
                  <button
                    type="button"
                    onClick={onGoogleLogout}
                    title="Sair da conta Google"
                    className="p-1 rounded-lg text-slate-400 hover:text-rose-300 hover:bg-rose-950/40"
                  >
                    <LogOut className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-emerald-300 font-medium">
                <Cloud className="w-3 h-3 shrink-0" />
                <span>Sincronização automática ativa</span>
              </div>
            </div>
          ) : (
            onOpenGoogleModal && (
              <button
                type="button"
                onClick={onOpenGoogleModal}
                data-testid="sidebar-google-login-btn"
                className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl text-xs font-semibold bg-white hover:bg-slate-100 text-slate-900 shadow transition-colors"
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
                <span>Entrar com o Google</span>
              </button>
            )
          )}

          <div className="px-3 py-2.5 rounded-lg bg-[#171b26] border border-white/5 text-xs">
            <div className="flex items-center justify-between text-slate-400">
              <span>Avaliações</span>
              <span className="font-mono font-bold text-slate-200">{totalRatings}</span>
            </div>
          </div>

          <button
            type="button"
            onClick={onResetProfile}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-slate-400 hover:text-rose-300 bg-[#171b26] hover:bg-rose-950/30 border border-white/5 hover:border-rose-500/30 transition-colors"
            title="Resetar histórico e preferências para o estado inicial"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Resetar Preferências</span>
          </button>
        </div>
      </aside>

      {/* Mobile Bottom Navigation Bar (< 768px) */}
      <nav
        aria-label="Navegação Principal Mobile"
        className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-[#0d1017]/95 backdrop-blur-lg border-t border-[#232936] px-2 py-1.5 flex items-center justify-around select-none shadow-2xl"
      >
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectTab(item.id)}
              className={`flex flex-col items-center justify-center flex-1 py-1 px-1 rounded-xl text-[10px] font-medium transition-colors relative ${
                active
                  ? 'text-indigo-400 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className={`p-1 rounded-lg ${active ? 'bg-indigo-600/20 text-indigo-400' : ''}`}>
                <Icon className="w-5 h-5" />
              </div>
              <span className="truncate mt-0.5">{item.label}</span>
              {item.id === 'list' && totalRatings > 0 && (
                <span className="absolute top-1 right-1/4 px-1 rounded-full text-[9px] font-mono bg-indigo-600 text-white font-bold">
                  {totalRatings > 99 ? '99+' : totalRatings}
                </span>
              )}
            </button>
          );
        })}
      </nav>
    </>
  );
};
