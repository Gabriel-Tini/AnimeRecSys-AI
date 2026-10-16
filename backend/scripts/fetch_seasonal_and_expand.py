import json
import urllib.request
import urllib.parse
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CATALOG_PATH = BASE_DIR / "data" / "cache_catalog.json"
SEASONAL_PATH = BASE_DIR / "data" / "seasonal_releases.json"
SYNOPSES_PATH = BASE_DIR / "data" / "synopses_pt.json"

def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    import re
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"</?[biu]>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Source:.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Written by.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Note:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def translate_gtx(text: str) -> str:
    cleaned = clean_html(text)
    if not cleaned:
        return ""
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=pt-BR&dt=t&q=" + urllib.parse.quote(cleaned[:1000].strip())
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = "".join([p[0] for p in data[0] if p and p[0]])
            if translated:
                return translated.strip()
    except Exception as e:
        print(f"Translation error: {e}")
    return cleaned

def fetch_graphql(query: str, variables: dict):
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        "https://graphql.anilist.co",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_current_season():
    now = datetime.now()
    year = now.year
    month = now.month
    if month in (1, 2, 3):
        return "WINTER", year
    elif month in (4, 5, 6):
        return "SPRING", year
    elif month in (7, 8, 9):
        return "SUMMER", year
    else:
        return "FALL", year

def fetch_seasonal_anime():
    season, year = get_current_season()
    print(f"Fetching seasonal anime for {season} {year}...")
    
    query = """
    query GetSeasonal($season: MediaSeason, $seasonYear: Int, $page: Int) {
      Page(page: $page, perPage: 30) {
        pageInfo {
          hasNextPage
        }
        media(type: ANIME, season: $season, seasonYear: $seasonYear, sort: [POPULARITY_DESC]) {
          id
          title {
            romaji
            english
            native
          }
          format
          episodes
          status
          seasonYear
          season
          averageScore
          popularity
          genres
          coverImage {
            extraLarge
            large
          }
          bannerImage
          description(asHtml: false)
          studios(isMain: true) {
            nodes {
              name
            }
          }
          nextAiringEpisode {
            episode
            timeUntilAiring
          }
        }
      }
    }
    """
    
    res = fetch_graphql(query, {"season": season, "seasonYear": year, "page": 1})
    media = res.get("data", {}).get("Page", {}).get("media", [])
    
    # If currently in between seasons or year is ahead, fallback to latest popular airing season (e.g. 2024 / 2025)
    if not media:
        print(f"No media found for {season} {year}, falling back to current active airing season...")
        query_airing = """
        query GetAiring($page: Int) {
          Page(page: $page, perPage: 30) {
            media(type: ANIME, status: RELEASING, sort: [POPULARITY_DESC]) {
              id
              title {
                romaji
                english
                native
              }
              format
              episodes
              status
              seasonYear
              season
              averageScore
              popularity
              genres
              coverImage {
                extraLarge
                large
              }
              bannerImage
              description(asHtml: false)
              studios(isMain: true) {
                nodes {
                  name
                }
              }
              nextAiringEpisode {
                episode
                timeUntilAiring
              }
            }
          }
        }
        """
        res = fetch_graphql(query_airing, {"page": 1})
        media = res.get("data", {}).get("Page", {}).get("media", [])

    print(f"Found {len(media)} seasonal items.")
    return season, year, media

def main():
    season, year, seasonal_items = fetch_seasonal_anime()
    
    # Load synopses cache
    synopses = {}
    if SYNOPSES_PATH.exists():
        try:
            with open(SYNOPSES_PATH, "r", encoding="utf-8") as f:
                synopses = json.load(f)
        except Exception:
            pass

    processed_seasonal = []
    for item in seasonal_items:
        cid = str(item["id"])
        studios = [s["name"] for s in item.get("studios", {}).get("nodes", [])]
        studio_name = studios[0] if studios else None
        
        raw_desc = item.get("description") or ""
        pt_desc = synopses.get(cid)
        if not pt_desc and raw_desc:
            print(f"Translating seasonal synopsis: {item['title']['romaji']}...")
            pt_desc = translate_gtx(raw_desc)
            if pt_desc:
                synopses[cid] = pt_desc
                time.sleep(0.2)

        cov = item.get("coverImage") or {}
        next_ep = item.get("nextAiringEpisode")

        processed_seasonal.append({
            "id": item["id"],
            "title_romaji": item.get("title", {}).get("romaji") or "",
            "title_english": item.get("title", {}).get("english") or item.get("title", {}).get("romaji") or "",
            "title_native": item.get("title", {}).get("native") or "",
            "cover_image": cov.get("extraLarge") or cov.get("large") or "",
            "banner_image": item.get("bannerImage") or "",
            "format": item.get("format") or "TV",
            "episodes": item.get("episodes"),
            "status": item.get("status") or "RELEASING",
            "season": item.get("season") or season,
            "season_year": item.get("seasonYear") or year,
            "average_score": item.get("averageScore"),
            "popularity": item.get("popularity"),
            "genres": item.get("genres") or [],
            "description": pt_desc or raw_desc,
            "studio": studio_name,
            "next_episode": next_ep.get("episode") if next_ep else None,
            "time_until_airing": next_ep.get("timeUntilAiring") if next_ep else None,
        })

    # Save seasonal releases file
    seasonal_payload = {
        "season": season,
        "year": year,
        "updated_at": datetime.now().isoformat(),
        "items": processed_seasonal,
    }
    with open(SEASONAL_PATH, "w", encoding="utf-8") as f:
        json.dump(seasonal_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(processed_seasonal)} seasonal releases to {SEASONAL_PATH}.")

    # Save updated synopses
    with open(SYNOPSES_PATH, "w", encoding="utf-8") as f:
        json.dump(synopses, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
