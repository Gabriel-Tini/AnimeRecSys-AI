import json
import urllib.request
import urllib.parse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from app.services.anilist_service import AniListService
from app.services.translation_service import get_translation_service
from app.core.storage_manager import StorageManager
from app.core.config import get_settings

SEARCH_QUERY = """
query ($search: String) {
  Page(page: 1, perPage: 15) {
    media(search: $search, type: ANIME, sort: [POPULARITY_DESC]) {
      id
      title {
        romaji
        english
        native
      }
      format
      episodes
      duration
      status
      seasonYear
      averageScore
      meanScore
      popularity
      genres
      tags {
        id
        name
        category
        rank
        isMediaSpoiler
      }
      coverImage {
        extraLarge
        large
      }
      bannerImage
      description(asHtml: false)
    }
  }
}
"""

FRANCHISES = [
  "Dorohedoro",
  "JoJo",
  "Witch Hat Atelier",
  "Vinland Saga",
  "Chainsaw Man",
  "Jujutsu Kaisen",
  "Bleach",
  "Naruto",
  "One Piece",
  "Frieren",
  "Solo Leveling",
  "Hunter x Hunter",
  "Fullmetal Alchemist",
  "Mob Psycho",
  "Demon Slayer",
  "Spy x Family",
  "Blue Lock",
  "Kaiju No. 8",
  "Dandadan",
  "Hell's Paradise",
]

def main():
    settings = get_settings()
    storage = StorageManager()
    svc = AniListService(settings=settings, storage_manager=storage)
    trans = get_translation_service()

    catalog = storage.load_catalog()
    catalog_by_id = {a.id: a for a in catalog}
    print(f"Current catalog count: {len(catalog_by_id)}")

    new_added = 0

    for term in FRANCHISES:
        print(f"Searching AniList for franchise: {term}...")
        payload = json.dumps({"query": SEARCH_QUERY, "variables": {"search": term}}).encode("utf-8")
        req = urllib.request.Request(
            "https://graphql.anilist.co",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data", {}).get("Page", {}).get("media", [])
                for raw in items:
                    aid = raw.get("id")
                    if not aid:
                        continue

                    # If already in catalog, continue
                    if aid in catalog_by_id:
                        continue

                    try:
                        entity = svc.parse_media_item(raw)
                    except Exception as parse_err:
                        continue

                    # Translate description
                    if entity.description:
                        pt_desc = trans.get_synopsis_pt(entity.id, entity.description, allow_network=True)
                        if pt_desc:
                            entity.description = pt_desc

                    catalog_by_id[aid] = entity
                    new_added += 1
                    print(f"  + Added [{aid}] {entity.title_romaji} ({entity.format})")

        except Exception as e:
            print(f"Error fetching franchise {term}: {e}")
        time.sleep(0.4)

    print(f"Total new franchise titles added: {new_added}. Catalog now: {len(catalog_by_id)}")
    storage.save_catalog(list(catalog_by_id.values()))
    print("Catalog saved successfully.")

if __name__ == "__main__":
    main()
