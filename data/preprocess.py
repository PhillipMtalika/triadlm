"""Raw text -> language ID + filtering -> dedup -> tokenize -> pack -> shards.

Usage: python -m data.preprocess --config configs/tiny.yaml [--lang en]
Writes data/shards/<name>/*.pt + data/manifests/<name>.json sidecar.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

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


def preprocess(corpus_glob: str, tok_save_dir: str, shard_dir: str,
               manifest_path: str, train_split: float = 0.98,
               corpus_name: str = "corpus",
               license: str = "project-internal") -> dict:
    """Run the full pipeline; returns the manifest dict."""
    from triadlm.tokenizer import TriadTokenizer
    from triadlm.data import build_manifest

    files = sorted(glob.glob(corpus_glob, recursive=True))
    if not files:
        raise ValueError(f"no files match {corpus_glob}")
    tok = TriadTokenizer.load(tok_save_dir)
    os.makedirs(shard_dir, exist_ok=True)

    seen: set[str] = set()
    lang_hist: dict[str, int] = {}
    doc_count, tok_count, dup = 0, 0, 0
    train_ids: list[int] = []
    val_ids: list[int] = []
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for para in f.read().split("\n\n"):
                text = para.strip()
                if not text:
                    continue
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
                # deterministic split by hash
                if int(h[:8], 16) / 16**8 < train_split:
                    train_ids.extend(ids)
                else:
                    val_ids.extend(ids)
    import torch
    torch.save(train_ids, os.path.join(shard_dir, "train.pt"))
    torch.save(val_ids or train_ids[-64:], os.path.join(shard_dir, "val.pt"))
    manifest = build_manifest([{
        "corpus": corpus_name, "source": corpus_glob, "license": license,
        "language": "mixed", "documents": doc_count, "tokens": tok_count,
        "filtering_rules": FILTERING_RULES, "train_split": train_split,
        "val_split": round(1 - train_split, 4), "shard_dir": shard_dir,
    }], manifest_path)
    quality = {"documents": doc_count, "tokens": tok_count, "duplicates_dropped": dup,
               "language_histogram": lang_hist}
    with open(manifest_path.replace(".json", ".quality.json"), "w") as f:
        json.dump(quality, f, indent=2)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus-name", default="corpus")
    ap.add_argument("--license", default="project-internal")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    print(json.dumps(preprocess(
        cfg["data"].get("corpus_glob", "data/raw/*.txt"),
        cfg["tokenizer"]["save_dir"], cfg["data"]["shard_dir"],
        cfg["data"]["manifest"],
        train_split=float(cfg["data"].get("train_split", 0.98)),
        corpus_name=args.corpus_name,
        license=args.license), indent=2))


if __name__ == "__main__":
    main()
