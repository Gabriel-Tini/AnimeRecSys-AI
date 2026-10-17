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

const resolvedMediaCache = new Map<number, { cover?: string | null; banner?: string | null }>();
const pendingMediaRequests = new Map<number, Promise<{ cover?: string | null; banner?: string | null } | null>>();

function isLocalDevServer(): boolean {
  if (typeof window === 'undefined') return false;
  const h = window.location.hostname;
  return h === 'localhost' || h === '127.0.0.1' || h.startsWith('192.168.') || h.startsWith('10.');
}

async function fetchFreshAniListMedia(
  animeId: number
): Promise<{ cover?: string | null; banner?: string | null } | null> {
  if (resolvedMediaCache.has(animeId)) {
    return resolvedMediaCache.get(animeId) ?? null;
  }
  const existing = pendingMediaRequests.get(animeId);
  if (existing) return existing;

  const promise = (async () => {
    try {
      const res = await fetch('https://graphql.anilist.co', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          query: `query ($id: Int) { Media(id: $id, type: ANIME) { coverImage { extraLarge large } bannerImage } }`,
          variables: { id: animeId },
        }),
      });
      if (!res.ok) return null;
      const json = await res.json();
      const media = json?.data?.Media;
      if (!media) return null;
      const entry = {
        cover: media.coverImage?.extraLarge || media.coverImage?.large || null,
        banner: media.bannerImage || null,
      };
      resolvedMediaCache.set(animeId, entry);
      return entry;
    } catch {
      return null;
    } finally {
      pendingMediaRequests.delete(animeId);
    }
  })();

  pendingMediaRequests.set(animeId, promise);
  return promise;
}

/**
 * Safely handles image load errors:
 * 1. On local dev server, tries `/api/media/cover/${animeId}` first.
 * 2. Immediately applies `FALLBACK_ANIME_POSTER` so broken image icons never appear.
 * 3. Queries AniList GraphQL in the background to self-heal rotated CDN hashes on static hosting.
 */
export const handleCoverError = (
  e: React.SyntheticEvent<HTMLImageElement, Event>,
  animeId?: number
) => {
  const img = e.currentTarget;
  const failedSrc = img.src;
  const localProxyUrl = animeId ? `/api/media/cover/${animeId}` : null;

  if (
    isLocalDevServer() &&
    localProxyUrl &&
    !failedSrc.includes(`/api/media/cover/${animeId}`) &&
    !img.dataset.triedLocalProxy
  ) {
    img.dataset.triedLocalProxy = '1';
    img.src = localProxyUrl;
    return;
  }

  if (img.src !== FALLBACK_ANIME_POSTER) {
    img.src = FALLBACK_ANIME_POSTER;
  }

  if (animeId && !img.dataset.triedAnilistHeal) {
    img.dataset.triedAnilistHeal = '1';
    void fetchFreshAniListMedia(animeId).then((fresh) => {
      if (fresh?.cover && fresh.cover !== failedSrc) {
        img.src = fresh.cover;
      }
    });
  }
};

export const handleBannerError = (
  e: React.SyntheticEvent<HTMLImageElement, Event>,
  animeId?: number
) => {
  const img = e.currentTarget;
  const failedSrc = img.src;
  const localBannerUrl = animeId ? `/api/media/banner/${animeId}` : null;

  if (
    isLocalDevServer() &&
    localBannerUrl &&
    !failedSrc.includes(`/api/media/banner/${animeId}`) &&
    !img.dataset.triedLocalProxy
  ) {
    img.dataset.triedLocalProxy = '1';
    img.src = localBannerUrl;
    return;
  }

  if (img.src !== FALLBACK_ANIME_BANNER) {
    img.src = FALLBACK_ANIME_BANNER;
  }

  if (animeId && !img.dataset.triedAnilistHeal) {
    img.dataset.triedAnilistHeal = '1';
    void fetchFreshAniListMedia(animeId).then((fresh) => {
      if (fresh?.banner && fresh.banner !== failedSrc) {
        img.src = fresh.banner;
      }
    });
  }
};
