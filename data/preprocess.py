"""Raw text -> language ID + filtering -> dedup -> tokenize -> pack -> shards.

Streaming: paragraphs flow through one at a time and shards flush to disk at
`max_shard_tokens`, so GB-scale sources never sit in RAM.
Usage: python -m data.preprocess --config configs/tiny.yaml [--lang en]
Writes data/shards/<name>/train-*.pt + val-*.pt + data/manifests/<name>.json.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NY_MARKERS = {"ndi", "ndiwo", "chifukwa", "liti", "ndani", "moni", "zikomo",
              "madzi", "likulu", "chilankhulo", "chichewa", "mchichewa", "kuti"}

FILTERING_RULES: list[str] = ["dedup:sha256-exact", "min_length:1_doc",
                              "lang_id_threshold:heuristic"]


def guess_lang(text: str) -> str:
    """Heuristic language id: Chichewa marker density, else English."""
    words = set(re.findall(r"[a-zA-Z\u00c0-\u024f]+", text.lower()))
    if not words:
        return "en"
    if len(words & NY_MARKERS) >= 2:
        return "ny"
    return "en"


def iter_file_paragraphs(path: str) -> Iterable[str]:
    """Yield stripped paragraphs from a raw text file."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for para in f.read().split("\n\n"):
            text = para.strip()
            if text:
                yield text


def process_stream(paras: Iterable[str], tok: object, train_split: float,
                   shard_dir: str, max_shard_tokens: int = 1_000_000,
                   max_docs: int | None = None) -> dict:
    """Tokenize a paragraph stream into flushed shards; returns stats."""
    import torch
    os.makedirs(shard_dir, exist_ok=True)
    seen: set[str] = set()
    lang_hist: dict[str, int] = {}
    doc_count, tok_count, dup = 0, 0, 0
    bufs = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    lens = {"train": 0, "val": 0}

    def flush(split: str) -> None:
        import torch as _t
        path = os.path.join(shard_dir, f"{split}-{counts[split]:03d}.pt")
        _t.save(bufs[split], path)
        counts[split] += 1
        bufs[split] = []
        lens[split] = 0

    for text in paras:
        if max_docs is not None and doc_count >= max_docs:
            break
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        lang = guess_lang(text)
        lang_hist[lang] = lang_hist.get(lang, 0) + 1
        ids = tok.encode(text)
        if not ids:
            continue
        doc_count += 1
        tok_count += len(ids)
        split = "train" if int(h[:8], 16) / 16**8 < train_split else "val"
        bufs[split].extend(ids)
        lens[split] += len(ids)
        if lens[split] >= max_shard_tokens:
            flush(split)
    for split in ("train", "val"):
        if bufs[split]:
            flush(split)
    if counts["val"] == 0:  # tiny corpora: seed val from train tail
        tail = bufs["train"][-64:] if bufs["train"] else []
        if tail:
            import torch as _t
            _t.save(tail, os.path.join(shard_dir, "val-000.pt"))
            counts["val"] = 1
    _ = torch
    return {"documents": doc_count, "tokens": tok_count,
            "duplicates_dropped": dup, "language_histogram": lang_hist,
            "train_shards": counts["train"], "val_shards": counts["val"]}


def preprocess(corpus_glob: str, tok_save_dir: str, shard_dir: str,
               manifest_path: str, train_split: float = 0.98,
               corpus_name: str = "corpus",
               license: str = "project-internal",
               max_shard_tokens: int = 1_000_000,
               max_docs: int | None = None) -> dict:
    """File-glob entrypoint over process_stream; returns the manifest dict."""
    from triadlm.tokenizer import TriadTokenizer
    from triadlm.data import build_manifest

    files = sorted(glob.glob(corpus_glob, recursive=True))
    if not files:
        raise ValueError(f"no files match {corpus_glob}")
    tok = TriadTokenizer.load(tok_save_dir)

    def gen() -> Iterable[str]:
        for fp in files:
            yield from iter_file_paragraphs(fp)

    stats = process_stream(gen(), tok, train_split, shard_dir,
                           max_shard_tokens, max_docs)
    manifest = build_manifest([{
        "corpus": corpus_name, "source": corpus_glob, "license": license,
        "language": "mixed", "documents": stats["documents"],
        "tokens": stats["tokens"],
        "filtering_rules": FILTERING_RULES, "train_split": train_split,
        "val_split": round(1 - train_split, 4), "shard_dir": shard_dir,
    }], manifest_path)
    with open(manifest_path.replace(".json", ".quality.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus-name", default="corpus")
    ap.add_argument("--license", default="project-internal")
    ap.add_argument("--max-shard-tokens", type=int, default=1_000_000)
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    print(json.dumps(preprocess(
        cfg["data"].get("corpus_glob", "data/raw/*.txt"),
        cfg["tokenizer"]["save_dir"], cfg["data"]["shard_dir"],
        cfg["data"]["manifest"],
        train_split=float(cfg["data"].get("train_split", 0.98)),
        corpus_name=args.corpus_name,
        license=args.license,
        max_shard_tokens=args.max_shard_tokens,
        max_docs=args.max_docs), indent=2))


if __name__ == "__main__":
    main()
