import urllib.request
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

query = """
query ($search: String) {
  Page(page: 1, perPage: 10) {
    media(search: $search, type: ANIME, sort: [POPULARITY_DESC]) {
      id
      title {
        romaji
        english
      }
      format
      seasonYear
      popularity
    }
  }
}
"""

for term in ['Dorohedoro', 'JoJo', 'Witch Hat Atelier']:
    payload = json.dumps({'query': query, 'variables': {'search': term}}).encode('utf-8')
    req = urllib.request.Request('https://graphql.anilist.co', data=payload, headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        items = data.get('data', {}).get('Page', {}).get('media', [])
        print(f'=== Results for {term} ===')
        for it in items:
            print(it['id'], it['title']['romaji'], '|', it['title']['english'], it['format'], it['seasonYear'])
