import React from 'react';

/**
 * Fallback placeholder image encoded as a data URI SVG.
 * Ensures that if external CDNs return 404 or fail to load,
 * the UI renders a clean dark placeholder rather than a broken image graphic.
 */
export const FALLBACK_ANIME_POSTER =
  'data:image/svg+xml;utf8,' +
  encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" fill="none">
  <rect width="300" height="400" fill="#121620"/>
  <rect x="20" y="20" width="260" height="360" rx="12" fill="#1b202e" stroke="#2d3748" stroke-width="1.5" stroke-dasharray="4 4"/>
  <circle cx="150" cy="170" r="45" fill="#242b3d"/>
  <path d="M135 150 L175 170 L135 190 Z" fill="#6366f1"/>
  <text x="150" y="240" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="13" font-weight="600" text-anchor="middle">
    AnimeRecSys
  </text>
  <text x="150" y="260" fill="#64748b" font-family="system-ui, sans-serif" font-size="11" text-anchor="middle">
    Capa indisponível
  </text>
</svg>
`.trim());

export const FALLBACK_ANIME_BANNER =
  'data:image/svg+xml;utf8,' +
  encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300" viewBox="0 0 800 300" fill="none">
  <rect width="800" height="300" fill="#10131a"/>
  <rect x="20" y="20" width="760" height="260" rx="12" fill="#171b26" stroke="#232936" stroke-width="1.5"/>
  <path d="M385 130 L425 150 L385 170 Z" fill="#6366f1"/>
  <text x="400" y="200" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="14" font-weight="600" text-anchor="middle">
    AnimeRecSys
  </text>
</svg>
`.trim());

/**
 * Safely handles image load errors:
 * 1. If loading from an external CDN URL fails (e.g. AniList rotated CDN hash),
 *    it attempts to load the persistently cached version from our local backend proxy:
 *    `/api/media/cover/${animeId}` or `/api/media/banner/${animeId}`.
 * 2. If the local proxy also fails or no animeId is present, it gracefully displays
 *    the custom SVG fallback poster/banner.
 */
export const handleCoverError = (
  e: React.SyntheticEvent<HTMLImageElement, Event>,
  animeId?: number
) => {
  const img = e.currentTarget;
  const localProxyUrl = animeId ? `/api/media/cover/${animeId}` : null;

  if (localProxyUrl && !img.src.includes(`/api/media/cover/${animeId}`)) {
    img.src = localProxyUrl;
  } else if (img.src !== FALLBACK_ANIME_POSTER) {
    img.src = FALLBACK_ANIME_POSTER;
  }
};

export const handleBannerError = (
  e: React.SyntheticEvent<HTMLImageElement, Event>,
  animeId?: number
) => {
  const img = e.currentTarget;
  const localBannerUrl = animeId ? `/api/media/banner/${animeId}` : null;

  if (localBannerUrl && !img.src.includes(`/api/media/banner/${animeId}`)) {
    img.src = localBannerUrl;
  } else if (img.src !== FALLBACK_ANIME_BANNER) {
    img.src = FALLBACK_ANIME_BANNER;
  }
};
