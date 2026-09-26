"""Build corpus_v2 from free sources: local raw + Wikipedia EN/NY streams +
JW300 en-ny (all $0, license-clean). Runs on Kaggle (free bandwidth/disk);
upload shards + manifest to the Hub after.

Usage: python -m data.build_large_corpus --tokenizer-dir ... --shard-dir ...
Lazily imports `datasets` (pip install datasets) so repo tests stay light.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LICENSE_V2 = ("CC-BY-SA (Wikipedia) + OPUS CCAligned en-ny "
              "(source licenses vary, see opus.nlpl.eu/CCAligned-v1.php) "
              "+ project-internal (seed)")

OPUS_URLS = [
    "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/en-ny.txt.zip",
    "https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1/moses/en-ny.txt.zip",
]


def iter_local(corpus_glob: str):  # generator of paragraphs
    """Yield paragraphs from local raw files."""
    from data.preprocess import iter_file_paragraphs
    for fp in sorted(glob.glob(corpus_glob, recursive=True)):
        yield from iter_file_paragraphs(fp)


def iter_wikipedia(lang: str, max_articles: int | None):
    """Yield article paragraphs from the streamed HF wikipedia dump."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                      split="train", streaming=True)
    n = 0
    for row in ds:
        if max_articles is not None and n >= max_articles:
            break
        text = (row.get("text") or "").strip()
        if len(text) < 500:
            continue
        n += 1
        yield f"# {row.get('title', '')}\n\n{text}"


def iter_opus_en_ny(max_pairs: int | None, cache_dir: str = "data/tmp"):
    """Yield en + ny lines from OPUS CCAligned moses files (free, cached).

    Each line is its own document. Downloads once (KBs), reuses cache.
    """
    import glob as _g
    import urllib.request
    import zipfile
    os.makedirs(cache_dir, exist_ok=True)
    hits = [p for p in _g.glob(os.path.join(cache_dir, "*.ny"))
            if os.path.getsize(p) > 0]
    if not hits:
        for url in OPUS_URLS:
            try:
                zpath = os.path.join(cache_dir, "en-ny.txt.zip")
                print(f"downloading {url} ...")
                urllib.request.urlretrieve(url, zpath)
                with zipfile.ZipFile(zpath) as z:
                    names = [n for n in z.namelist() if n.endswith(".ny")]
                    if not names:
                        raise ValueError(f"no .ny file in {z.namelist()[:5]}")
                    z.extractall(cache_dir)
                break
            except Exception as e:
                print(f"mirror failed ({e}), trying next")
        hits = [p for p in _g.glob(os.path.join(cache_dir, "*.ny"))
                if os.path.getsize(p) > 0]
        if not hits:
            print("OPUS en-ny unavailable, skipping")
            return
        print("downloaded + extracted")
    txt = sorted(hits)[0]
    en_path = txt[:-3] + ".en"
    n = 0
    with open(en_path, encoding="utf-8", errors="ignore") as fe, \
            open(txt, encoding="utf-8", errors="ignore") as fn:
        for en, ny in zip(fe, fn):
            if max_pairs is not None and n >= max_pairs:
                break
            en, ny = en.strip(), ny.strip()
            if en and ny:
                n += 1
                yield en
                yield ny
    print(f"opus en-ny pairs: {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--shard-dir", default="data/shards/corpus_v2")
    ap.add_argument("--manifest", default="data/manifests/corpus_v2.json")
    ap.add_argument("--corpus-glob", default="data/raw/*.txt")
    ap.add_argument("--max-en", type=int, default=20000)
    ap.add_argument("--ny-full", action="store_true",
                    help="stream all of ny.wikipedia (small); else 2000 articles")
    ap.add_argument("--max-jw", type=int, default=50000)
    ap.add_argument("--max-shard-tokens", type=int, default=2_000_000)
    args = ap.parse_args()
    from data.preprocess import process_stream
    from triadlm.data import build_manifest
    from triadlm.tokenizer import TriadTokenizer
    import json

    tok = TriadTokenizer.load(args.tokenizer_dir)

    def gen():
        yield from iter_local(args.corpus_glob)
        yield from iter_wikipedia("en", args.max_en)
        try:
            yield from iter_wikipedia("ny", None if args.ny_full else 2000)
        except Exception as e:
            print(f"ny wikipedia stream skipped ({e})")
        yield from iter_opus_en_ny(args.max_jw)

    stats = process_stream(gen(), tok, 0.98, args.shard_dir,
                           args.max_shard_tokens)
    manifest = build_manifest([{
        "corpus": "corpus_v2",
        "source": "wikimedia/wikipedia en+ny (stream) + opus CCAligned en-ny + " + args.corpus_glob,
        "license": LICENSE_V2, "language": "mixed",
        "documents": stats["documents"], "tokens": stats["tokens"],
        "filtering_rules": ["dedup:sha256-exact", "min_length:1_doc",
                            "lang_id_threshold:heuristic"],
        "train_split": 0.98, "val_split": 0.02,
        "shard_dir": args.shard_dir}], args.manifest)
    with open(args.manifest.replace(".json", ".quality.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
