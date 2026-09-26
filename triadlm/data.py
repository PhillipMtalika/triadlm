"""Dataset manifest loader + packed token shards (spec §4)."""
import hashlib
import json
import os
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(sources: list[dict], out_path: str) -> dict:
    """Write one manifest JSON per corpus (schema §4) + return it.

    Each source: {name, source, license, language, documents, tokens,
                  filtering_rules, train_split, val_split}.
    """
    docs = sum(s.get("documents", 0) for s in sources)
    toks = sum(s.get("tokens", 0) for s in sources)
    shard_dir = sources[0].get("shard_dir", "") if sources else ""
    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt")) if shard_dir and os.path.isdir(shard_dir) else []
    import datetime
    h = hashlib.sha256()
    for s in shards:
        h.update(_sha256_file(os.path.join(shard_dir, s)).encode())
    manifest: dict = {
        "name": sources[0].get("corpus", "corpus") if sources else "corpus",
        "source": "; ".join(s.get("source", "") for s in sources),
        "license": "; ".join(sorted({s.get("license", "unknown") for s in sources})),
        "language": sources[0].get("language", "mixed") if len(sources) == 1 else "mixed",
        "document_count": docs,
        "token_count": toks,
        "filtering_rules": sources[0].get("filtering_rules", []) if sources else [],
        "train_split": sources[0].get("train_split", 0.98) if sources else 0.98,
        "val_split": sources[0].get("val_split", 0.02) if sources else 0.02,
        "sha256_of_shards": h.hexdigest() if shards else "",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_manifest(path: str) -> dict:
    """Load a manifest JSON."""
    with open(path) as f:
        return json.load(f)


def shard_paths(shard_dir: str, split: str) -> list[str]:
    """Sorted shard paths for a split (multi-shard names, legacy fallback)."""
    import glob as _g
    paths = sorted(_g.glob(os.path.join(shard_dir, f"{split}-*.pt")))
    if not paths:
        legacy = os.path.join(shard_dir, f"{split}.pt")
        paths = [legacy] if os.path.exists(legacy) else []
    return paths


class PackedDataset(Dataset):
    """Packed token blocks from `.pt` shards; returns (x, y), y = x shifted by 1."""

    def __init__(self, shard_paths: list[str], block_size: int) -> None:
        ids: list[int] = []
        for p in sorted(shard_paths):
            ids.extend(torch.load(p, map_location="cpu", weights_only=True))
        self.block_size = block_size
        n = (len(ids) - 1) // (block_size + 1)
        self.chunks: list[list[int]] = [
            ids[i * (block_size + 1):(i + 1) * (block_size + 1)] for i in range(n)]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        c = torch.tensor(self.chunks[idx], dtype=torch.long)
        return c[:-1].contiguous(), c[1:].contiguous()
