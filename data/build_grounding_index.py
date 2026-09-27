"""Build grounding passages for the local index from Malawi wiki articles.

Appends paragraph passages (id geo-N) to data/documents.jsonl. Rerunnable
(dedups by id). Run: python -m data.build_grounding_index
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = "data/raw/wiki_en.txt"
OUT = "data/documents.jsonl"
MIN_CHARS = 80
PER_ARTICLE = 8


def articles(text: str) -> list[tuple[str, list[str]]]:
    """Split raw file into (title, paragraphs); first block is untitled."""
    out: list[tuple[str, list[str]]] = []
    title, cur = "intro", []
    for p in [x.strip() for x in text.split("\n\n")]:
        if not p:
            continue
        if p.startswith("# "):
            if cur:
                out.append((title, cur))
            title, cur = p[2:].strip(), []
        else:
            cur.append(p)
    if cur:
        out.append((title, cur))
    return out


def build() -> int:
    """Append missing geo-N passages (quota per article); returns added."""
    have: set[str] = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                have.add(json.loads(line).get("id", ""))
    with open(SRC, encoding="utf-8") as f:
        arts = articles(f.read())
    texts: list[str] = []
    for _, paras in arts:
        got = 0
        for p in paras:
            if got >= PER_ARTICLE or len(p) < MIN_CHARS:
                continue
            texts.append(p)
            got += 1
    added = 0
    n = sum(1 for h in have if h.startswith("geo-")) + 1
    with open(OUT, "a", encoding="utf-8") as f:
        for p in texts:
            pid = f"geo-{n}"
            n += 1
            if pid in have:
                continue
            f.write(json.dumps({"id": pid, "text": p[:600],
                                "url": "local://wikipedia-malawi"}) + "\n")
            added += 1
    return added


def main() -> None:
    print(f"added {build()} passages to {OUT}")


if __name__ == "__main__":
    main()
