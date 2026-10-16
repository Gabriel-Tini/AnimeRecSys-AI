import urllib.request
import json
import urllib.parse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_query(q):
    url = "http://127.0.0.1:8000/api/catalog?q=" + urllib.parse.quote(q) + "&limit=5"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"=== Query: {q} ({data.get('total')} results) ===")
        for it in data.get("items", []):
            print(" -", it.get("id"), it.get("title_romaji"), "|", it.get("title_english"), f"({it.get('format')})")

with urllib.request.urlopen("http://127.0.0.1:8000/api/health") as resp:
    print("HEALTH:", resp.read().decode("utf-8"))

test_query("Dorohedoro")
test_query("JoJo")
test_query("Witch Hat")
