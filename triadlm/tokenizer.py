"""Byte-level BPE tokenizer trained on the project's own corpus.

An off-the-shelf English tokenizer is deliberately NOT used: one goal is
measuring the English/Chichewa gap, and a pretrained English tokenizer would
bias that result before training starts.
"""
import hashlib
import json
import os
from tokenizers import ByteLevelBPETokenizer, Tokenizer

SPECIAL_TOKENS: dict[str, int] = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3}
_SPECIALS_ORDERED: list[str] = ["<bos>", "<eos>", "<pad>", "<unk>"]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TriadTokenizer:
    """Spec §2 interface. Special-token ids are fixed (see SPECIAL_TOKENS)."""

    vocab_size: int
    special_tokens: dict[str, int]

    def __init__(self, tok: Tokenizer, vocab_size: int) -> None:
        self._tok = tok
        self.vocab_size = vocab_size
        self.special_tokens = dict(SPECIAL_TOKENS)

    @classmethod
    def train(cls, corpus_paths: list[str], vocab_size: int, save_dir: str) -> "TriadTokenizer":
        """Train byte-level BPE on project corpus + write JSON metadata sidecar."""
        if not corpus_paths:
            raise ValueError("no corpus files")
        os.makedirs(save_dir, exist_ok=True)
        bpe = ByteLevelBPETokenizer()
        bpe.train(files=sorted(corpus_paths), vocab_size=vocab_size,
                  min_frequency=1, special_tokens=_SPECIALS_ORDERED)
        tok_path = os.path.join(save_dir, "tokenizer.json")
        bpe.save(tok_path)
        meta = {
            "vocab_size": vocab_size,
            "corpus_paths": sorted(corpus_paths),
            "corpus_sha256": [_sha256_file(p) for p in sorted(corpus_paths)],
            "special_tokens": dict(SPECIAL_TOKENS),
        }
        with open(os.path.join(save_dir, "tokenizer_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return cls.load(save_dir)

    @classmethod
    def load(cls, save_dir: str) -> "TriadTokenizer":
        """Load a tokenizer + its metadata sidecar."""
        with open(os.path.join(save_dir, "tokenizer_meta.json")) as f:
            meta = json.load(f)
        tok = Tokenizer.from_file(os.path.join(save_dir, "tokenizer.json"))
        return cls(tok, int(meta["vocab_size"]))

    def _special_ids(self) -> set[int]:
        return {self._tok.token_to_id(t) for t in _SPECIALS_ORDERED
                if self._tok.token_to_id(t) is not None}

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Encode text; bos/eos wrap with the fixed special ids."""
        ids: list[int] = self._tok.encode(text).ids
        if add_bos:
            ids = [SPECIAL_TOKENS["<bos>"]] + ids
        if add_eos:
            ids = ids + [SPECIAL_TOKENS["<eos>"]]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode, stripping wrapper special tokens so round-trip holds."""
        drop = self._special_ids()
        return self._tok.decode([i for i in ids if i not in drop])

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of texts."""
        return [self.encode(t) for t in texts]
