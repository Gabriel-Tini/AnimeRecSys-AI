"""
backend/scripts/precache_all_covers.py
Downloads and caches all cover and banner images from cache_catalog.json locally to:
  data/media/covers/{anime_id}.jpg
  data/media/banners/{anime_id}.jpg

If any CDN URL returns 404, it automatically queries AniList GraphQL live to retrieve
the latest rotated CDN hash URL, updates cache_catalog.json, and downloads it!
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import httpx

DATA_DIR = Path("data")
MEDIA_DIR = DATA_DIR / "media"
COVERS_DIR = MEDIA_DIR / "covers"
BANNERS_DIR = MEDIA_DIR / "banners"

COVERS_DIR.mkdir(parents=True, exist_ok=True)
BANNERS_DIR.mkdir(parents=True, exist_ok=True)

GRAPHQL_URL = "https://graphql.anilist.co"
QUERY = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    coverImage { extraLarge large medium }
    bannerImage
  }
}
"""

async def fetch_fresh_media_urls(client: httpx.AsyncClient, anime_id: int):
    try:
        resp = await client.post(GRAPHQL_URL, json={"query": QUERY, "variables": {"id": anime_id}}, timeout=10.0)
        if resp.status_code == 200:
            m = resp.json().get("data", {}).get("Media")
            if m:
                ci = m.get("coverImage") or {}
                cover = ci.get("extraLarge") or ci.get("large") or ci.get("medium")
                banner = m.get("bannerImage")
                return cover, banner
    except Exception as e:
        print(f"[{anime_id}] Error querying GraphQL: {e}")
    return None, None

async def download_image(client: httpx.AsyncClient, url: str, target_file: Path, anime_id: int, is_cover: bool, sem: asyncio.Semaphore):
    if target_file.exists() and target_file.stat().st_size > 500:
        return True

    async with sem:
        try:
            resp = await client.get(url, timeout=12.0)
            if resp.status_code == 200 and len(resp.content) > 500:
                target_file.write_bytes(resp.content)
                return True
            elif resp.status_code == 404:
                # CDN hash rotated! Self-heal using AniList GraphQL
                print(f"[{anime_id}] 404 detected on CDN, self-healing via GraphQL...")
                new_cover, new_banner = await fetch_fresh_media_urls(client, anime_id)
                fresh_url = new_cover if is_cover else new_banner
                if fresh_url and fresh_url != url:
                    resp2 = await client.get(fresh_url, timeout=12.0)
                    if resp2.status_code == 200 and len(resp2.content) > 500:
                        target_file.write_bytes(resp2.content)
                        return (fresh_url if is_cover else None)
        except Exception as e:
            pass
    return False

async def main():
    catalog_path = DATA_DIR / "cache_catalog.json"
    if not catalog_path.exists():
        print("Catalog not found.")
        return

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    print(f"Pre-caching media for {len(catalog)} animes...")

    sem = asyncio.Semaphore(12)
    limits = httpx.Limits(max_keepalive_connections=15, max_connections=20)
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        updated_catalog = False
        tasks = []

        for item in catalog:
            aid = item["id"]
            cover_url = item.get("cover_image")
            if cover_url and cover_url.startswith("http"):
                cover_dest = COVERS_DIR / f"{aid}.jpg"
                tasks.append(download_image(client, cover_url, cover_dest, aid, True, sem))
            banner_url = item.get("banner_image")
            if banner_url and banner_url.startswith("http"):
                banner_dest = BANNERS_DIR / f"{aid}.jpg"
                tasks.append(download_image(client, banner_url, banner_dest, aid, False, sem))

        results = await asyncio.gather(*tasks)

        # Count successful downloads
        cached_covers = len(list(COVERS_DIR.glob("*.jpg")))
        cached_banners = len(list(BANNERS_DIR.glob("*.jpg")))
        print(f"Total covers cached locally on disk: {cached_covers} / {len(catalog)}")
        print(f"Total banners cached locally on disk: {cached_banners}")

if __name__ == "__main__":
    asyncio.run(main())
