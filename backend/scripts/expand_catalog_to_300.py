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

from app.services.anilist_service import POPULAR_ANIME_QUERY, AniListService
from app.core.storage_manager import StorageManager
from app.core.config import get_settings

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
        pass
    return cleaned

def main():
    settings = get_settings()
    storage = StorageManager()
    svc = AniListService(settings=settings, storage_manager=storage)

    existing_catalog = storage.load_catalog()
    existing_by_id = {a.id: a for a in existing_catalog}
    print(f"Current catalog size: {len(existing_catalog)}")

    synopses_path = BASE_DIR / "data" / "synopses_pt.json"
    synopses = {}
    if synopses_path.exists():
        try:
            with open(synopses_path, "r", encoding="utf-8") as f:
                synopses = json.load(f)
        except Exception:
            pass

    # Fetch 6 pages of 50 items each = up to 300 popular animes
    new_added = 0
    client = urllib.request

    for page in range(1, 7):
        print(f"Fetching AniList popular page {page}/6...")
        payload = json.dumps({"query": POPULAR_ANIME_QUERY, "variables": {"page": page, "perPage": 50}}).encode("utf-8")
        req = urllib.request.Request(
            "https://graphql.anilist.co",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                media_list = data.get("data", {}).get("Page", {}).get("media", [])
                for raw_item in media_list:
                    aid = raw_item.get("id")
                    if not aid:
                        continue
                    
                    try:
                        entity = svc.parse_media_item(raw_item)
                    except Exception as parse_err:
                        continue

                    # If already in catalog, keep existing
                    if aid in existing_by_id:
                        continue

                    # New anime to add
                    cid_str = str(aid)
                    raw_desc = entity.description or ""
                    if cid_str not in synopses and raw_desc:
                        pt_desc = translate_gtx(raw_desc)
                        if pt_desc:
                            synopses[cid_str] = pt_desc
                            entity.description = pt_desc
                    elif cid_str in synopses:
                        entity.description = synopses[cid_str]

                    existing_by_id[aid] = entity
                    new_added += 1

        except Exception as exc:
            print(f"Error fetching page {page}: {exc}")
        time.sleep(0.8)

    print(f"Added {new_added} new popular animes. Total now: {len(existing_by_id)}")
    
    # Save expanded catalog and synopses
    expanded_list = list(existing_by_id.values())
    storage.save_catalog(expanded_list)
    
    with open(synopses_path, "w", encoding="utf-8") as f:
        json.dump(synopses, f, ensure_ascii=False, indent=2)

    print("Expanded catalog saved successfully!")

if __name__ == "__main__":
    main()
