import json
import urllib.request
import urllib.parse
import re
import time
import sys
from pathlib import Path

# Ensure stdout handles Unicode in Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CATALOG_PATH = BASE_DIR / "data" / "cache_catalog.json"
SYNOPSES_PATH = BASE_DIR / "data" / "synopses_pt.json"

EN_MARKERS = [" the ", " and ", " of ", " is ", " with ", " to ", " was ", " in ", " for ", " that "]

def is_english(text: str) -> bool:
    if not text:
        return False
    t = f" {text.lower()} "
    matches = sum(1 for m in EN_MARKERS if m in t)
    return matches >= 2

def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"</?[biu]>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Source:.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Written by.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Note:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def translate_to_portuguese(text: str) -> str:
    cleaned = clean_html(text)
    if not cleaned:
        return ""
    
    # Take up to 1200 characters for good detailed synopsis
    snippet = cleaned[:1200].strip()
    
    # Try Google translate GTX endpoint
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=pt-BR&dt=t&q=" + urllib.parse.quote(snippet)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = "".join([part[0] for part in data[0] if part and part[0]])
            if translated and not is_english(translated):
                return translated.strip()
    except Exception as e:
        print(f"Error with Google GTX: {e}")
    
    # Fallback to MyMemory
    try:
        url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode({
            "q": snippet[:350],
            "langpair": "en|pt-BR"
        })
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            res_text = data.get("responseData", {}).get("translatedText", "")
            if res_text and not res_text.startswith("MYMEMORY WARNING") and not is_english(res_text):
                return res_text.strip()
    except Exception as e:
        print(f"Error with MyMemory: {e}")
    
    return cleaned

def main():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    synopses = {}
    if SYNOPSES_PATH.exists():
        try:
            with open(SYNOPSES_PATH, "r", encoding="utf-8") as f:
                synopses = json.load(f)
        except Exception:
            synopses = {}

    print(f"Loaded {len(catalog)} catalog items. Existing synopses: {len(synopses)}")

    translated_count = 0
    skipped_count = 0

    for item in catalog:
        cid = str(item["id"])
        title = item.get("title", {}).get("romaji", "")
        existing = synopses.get(cid, "")
        
        # If existing is already good Portuguese, keep it
        if existing and not is_english(existing):
            skipped_count += 1
            continue

        raw_desc = item.get("description") or ""
        if not raw_desc:
            print(f"[{cid}] {title}: No description available.")
            continue

        print(f"[{cid}] Translating {title}...")
        pt_desc = translate_to_portuguese(raw_desc)
        if pt_desc and not is_english(pt_desc):
            synopses[cid] = pt_desc
            translated_count += 1
            print(f"  -> OK ({len(pt_desc)} chars)")
        else:
            print(f"  -> Translation failed or returned English.")
        
        time.sleep(0.2)

    with open(SYNOPSES_PATH, "w", encoding="utf-8") as f:
        json.dump(synopses, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Translated: {translated_count}, Already PT: {skipped_count}, Total cached: {len(synopses)}")

if __name__ == "__main__":
    main()
