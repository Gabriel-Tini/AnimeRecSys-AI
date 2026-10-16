"""
Audits every catalog entry against AniList by ID.
If the stored title does not match the real AniList title for that ID, the entry is
corrupted (wrong ID in seed data). Fix strategy:
  1. Rebuild the entry at that ID from AniList (the ID is the source of truth).
  2. Drop its cached PT synopsis (it described the wrong anime) and re-translate.
  3. Search AniList by the *old title* and ingest the correct anime under its real ID.
  4. Remap user lists / watch history from the wrong ID to the correct ID.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"c:\Dev\Ent ML")
sys.path.insert(0, str(BASE / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.core.storage_manager import StorageManager  # noqa: E402
from app.services.anilist_service import AniListService  # noqa: E402
from app.services.translation_service import get_translation_service  # noqa: E402

MEDIA_FIELDS = """
  id
  title { romaji english native }
  format episodes duration status seasonYear averageScore meanScore popularity genres
  tags { id name category rank isMediaSpoiler }
  coverImage { extraLarge large }
  bannerImage
  description(asHtml: false)
"""
BY_IDS = "query ($ids: [Int]) { Page(page: 1, perPage: 50) { media(id_in: $ids, type: ANIME) {" + MEDIA_FIELDS + "} } }"
BY_SEARCH = "query ($s: String) { Page(page: 1, perPage: 1) { media(search: $s, type: ANIME, sort: [SEARCH_MATCH]) {" + MEDIA_FIELDS + "} } }"


def gql(query, variables):
    req = urllib.request.Request(
        "https://graphql.anilist.co",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())["data"]["Page"]["media"]


def norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def main():
    storage = StorageManager()
    svc = AniListService(settings=get_settings(), storage_manager=storage)
    trans = get_translation_service()

    catalog = storage.load_catalog()
    by_id = {a.id: a for a in catalog}
    ids = list(by_id)
    live = {}
    for i in range(0, len(ids), 50):
        for m in gql(BY_IDS, {"ids": ids[i:i + 50]}):
            live[m["id"]] = m
        time.sleep(0.7)

    mismatches = []
    for aid, ent in by_id.items():
        m = live.get(aid)
        if not m:
            print(f"[MISSING ON ANILIST] {aid} {ent.title_romaji}")
            continue
        if norm(m["title"]["romaji"]) != norm(ent.title_romaji):
            mismatches.append((aid, ent.title_romaji, ent.title_english, m))

    print(f"Checked {len(by_id)} entries. Mismatches: {len(mismatches)}")
    id_remap = {}
    for aid, old_romaji, old_english, m in mismatches:
        print(f"  ! ID {aid}: catalog said '{old_romaji}' but AniList says '{m['title']['romaji']}'")
        # 1+2: rebuild entry at this ID from truth
        trans._cache.pop(str(aid), None)
        fixed = svc.parse_media_item(m)
        if fixed.description:
            fixed.description = trans.get_synopsis_pt(fixed.id, fixed.description, allow_network=True) or fixed.description
        by_id[aid] = fixed
        # 3: ingest the anime the old title actually referred to
        found = gql(BY_SEARCH, {"s": old_english or old_romaji})
        time.sleep(0.7)
        if found:
            correct = svc.parse_media_item(found[0])
            if correct.description:
                correct.description = trans.get_synopsis_pt(correct.id, correct.description, allow_network=True) or correct.description
            by_id[correct.id] = correct
            id_remap[aid] = correct.id
            print(f"    -> correct anime '{correct.title_romaji}' is ID {correct.id}")

    trans._save_cache()
    storage.save_catalog(list(by_id.values()))

    # 4: remap user data that referenced the wrong ID (user meant the old title)
    if id_remap:
        def remap_list(lst):
            return [id_remap.get(x, x) for x in lst]

        def modifier(prof):
            return prof.model_copy(update={
                "watchlist_anime_ids": remap_list(prof.watchlist_anime_ids),
                "favorite_anime_ids": remap_list(prof.favorite_anime_ids),
                "skipped_anime_ids": remap_list(prof.skipped_anime_ids),
                "dropped_anime_ids": remap_list(prof.dropped_anime_ids),
            })
        storage.update_user_profile(modifier)

        hist_path = storage.data_dir / "watch_history.json" if hasattr(storage, "data_dir") else BASE / "data" / "watch_history.json"
        if hist_path.exists():
            raw = json.loads(hist_path.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("items", [])
            for h in items:
                if h.get("anime_id") in id_remap:
                    h["anime_id"] = id_remap[h["anime_id"]]
            hist_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Remapped user data:", id_remap)

    print(f"Done. Catalog size: {len(by_id)}")


if __name__ == "__main__":
    main()
