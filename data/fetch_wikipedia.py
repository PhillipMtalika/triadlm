"""Fetch free, license-clean corpus text from Wikipedia (CC BY-SA).

EN + Chichewa (ny) Malawi topics, one API request per language. Writes
data/raw/wiki_en.txt and data/raw/wiki_ny.txt, then re-run preprocess with
--license "CC-BY-SA (Wikipedia) + project-internal (seed)".

Usage: python -m data.fetch_wikipedia [--out-dir data/raw]
$0 budget: public API, no key, 2 requests total.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOPICS_EN: list[str] = [
    "Malawi", "Lilongwe", "Blantyre", "Lake Malawi", "Chichewa language",
    "Culture of Malawi", "Economy of Malawi", "History of Malawi",
    "Geography of Malawi", "Mzuzu", "Zomba, Malawi", "Malawian cuisine",
]

TOPICS_NY: list[str] = [
    "Malawi", "Lilongwe", "Blantyre", "Nyanja", "Chichewa",
    "Lake Malawi", "Mzuzu", "Zomba",
]

UA = {"User-Agent": "TriadLM-research-corpus/0.1 (zero-budget research project)"}


def fetch(lang: str, titles: list[str]) -> dict[str, str]:
    """Return {title: plaintext extract} for existing pages.

    One request per title: the API caps whole-article extracts at 1 per call.
    Still cheap (~20 small requests, $0, no key).
    """
    import time
    import requests
    url = f"https://{lang}.wikipedia.org/w/api.php"
    out: dict[str, str] = {}
    for t in titles:
        for attempt in range(4):
            r = requests.get(url, params={"action": "query", "prop": "extracts",
                                          "explaintext": True,
                                          "titles": t, "format": "json"},
                             headers=UA, timeout=30)
            if r.status_code == 429 and attempt < 3:
                time.sleep(2 ** attempt * 5)
                continue
            r.raise_for_status()
            break
        for p in r.json().get("query", {}).get("pages", {}).values():
            if "missing" not in p and p.get("extract", "").strip():
                out[p["title"]] = p["extract"].strip()
        time.sleep(2.0)
    return out


def write_corpus(lang: str, articles: dict[str, str], out_dir: str) -> str:
    """Write one raw corpus file; returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"wiki_{lang}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for title, text in sorted(articles.items()):
            f.write(f"# {title}\n\n{text}\n\n")
    return path


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw")
    args = ap.parse_args()
    en = fetch("en", TOPICS_EN)
    print(f"en: {len(en)}/{len(TOPICS_EN)} pages")
    ny = fetch("ny", TOPICS_NY)
    print(f"ny: {len(ny)}/{len(TOPICS_NY)} pages")
    print(write_corpus("en", en, args.out_dir))
    print(write_corpus("ny", ny, args.out_dir))
    print("license: CC BY-SA (Wikipedia text) — record in manifest via "
          "preprocess --license")


if __name__ == "__main__":
    main()
