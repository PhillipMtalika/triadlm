"""Quality report: dedup rate, language histogram, length histogram, top dups.

Usage: python -m data.quality_report --manifest data/manifests/toy.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quality_report(corpus_glob: str, out_path: str) -> dict:
    """Scan raw corpus and write the quality JSON sidecar."""
    from data.preprocess import guess_lang
    files = sorted(glob.glob(corpus_glob, recursive=True))
    hashes: Counter[str] = Counter()
    lang_hist: Counter[str] = Counter()
    lengths: list[int] = []
    n_paras = 0
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for para in f.read().split("\n\n"):
                text = para.strip()
                if not text:
                    continue
                n_paras += 1
                hashes[hashlib.sha256(text.encode()).hexdigest()[:16]] += 1
                lang_hist[guess_lang(text)] += 1
                lengths.append(len(text.split()))
    total = max(1, n_paras)
    dup_docs = sum(c - 1 for c in hashes.values() if c > 1)
    report = {
        "dedup_rate": dup_docs / total,
        "language_histogram": dict(lang_hist),
        "doc_length_histogram": {
            "min": min(lengths) if lengths else 0,
            "p50": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0,
        },
        "top_duplicates": hashes.most_common(5),
        "documents": n_paras,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--corpus-glob", default="data/raw/*.txt")
    args = ap.parse_args()
    out = args.manifest.replace(".json", ".quality.json")
    print(json.dumps(quality_report(args.corpus_glob, out), indent=2))


if __name__ == "__main__":
    main()
