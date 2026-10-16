import json
import urllib.request
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CATALOG_PATH = BASE_DIR / "data" / "cache_catalog.json"

def fetch_anilist_media_batch(ids):
    query = """
    query ($ids: [Int]) {
      Page(page: 1, perPage: 50) {
        media(id_in: $ids, type: ANIME) {
          id
          coverImage {
            extraLarge
            large
          }
          bannerImage
        }
      }
    }
    """
    payload = json.dumps({"query": query, "variables": {"ids": ids}}).encode("utf-8")
    req = urllib.request.Request(
        "https://graphql.anilist.co",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", {}).get("Page", {}).get("media", [])

def main():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    all_ids = [item["id"] for item in catalog]
    print(f"Total anime to update: {len(all_ids)}")

    media_map = {}
    for i in range(0, len(all_ids), 40):
        batch = all_ids[i:i+40]
        print(f"Fetching batch {i} to {i+len(batch)}...")
        try:
            results = fetch_anilist_media_batch(batch)
            for m in results:
                media_map[m["id"]] = m
        except Exception as e:
            print(f"Error fetching batch: {e}")
        time.sleep(0.5)

    updated_count = 0
    for item in catalog:
        cid = item["id"]
        if cid in media_map:
            m = media_map[cid]
            cov = m.get("coverImage", {})
            new_cover = cov.get("extraLarge") or cov.get("large")
            new_banner = m.get("bannerImage")
            if new_cover and new_cover != item.get("cover_image"):
                print(f"Updated cover for {cid}: {item.get('cover_image')} -> {new_cover}")
                item["cover_image"] = new_cover
                updated_count += 1
            if new_banner and new_banner != item.get("banner_image"):
                item["banner_image"] = new_banner

    print(f"Updated {updated_count} covers.")
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print("Catalog updated successfully.")

if __name__ == "__main__":
    main()
