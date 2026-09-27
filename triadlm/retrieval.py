"""Local document index + retriever (spec §8.2). No external deps.

Citation format: `[doc_id:start-end]` inline plus a trailing sources list.
"""
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable
from torch import Tensor


@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    score: float
    offset: tuple[int, int]


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z\u00c0-\u024f0-9]+", s.lower())


class DocumentIndex:
    """TF-IDF bag-of-words index over added documents."""

    def __init__(self, embedding_fn: Callable[[str], Tensor] | None = None) -> None:
        self.docs: dict[str, tuple[str, dict]] = {}
        self._idf: Counter[str] = Counter()
        self._doc_tf: dict[str, Counter[str]] = {}
        self._n = 0
        _ = embedding_fn  # reserved: plug in a neural embedder later

    def add(self, doc_id: str, text: str, metadata: dict) -> None:
        """Add one document (id must be unique)."""
        tf = Counter(_tokens(text))
        if doc_id not in self.docs:
            self._n += 1
            for t in set(tf):
                self._idf[t] += 1
        self.docs[doc_id] = (text, metadata)
        self._doc_tf[doc_id] = tf

    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]:
        """Return top-k passages with char offsets into the source doc."""
        qt = Counter(_tokens(query))
        scored: list[tuple[float, str]] = []
        for doc_id, (text, _) in self.docs.items():
            tf = self._doc_tf[doc_id]
            s = sum(qt[t] * tf[t] * math.log(1 + self._n / (1 + self._idf[t]))
                    for t in qt if t in tf)
            if s > 0:
                scored.append((s, doc_id))
        scored.sort(reverse=True)
        out: list[RetrievedPassage] = []
        for s, doc_id in scored[:k]:
            text, _ = self.docs[doc_id]
            m = re.search(re.escape(query.split()[0]) if query.split() else "",
                          text, re.IGNORECASE)
            start = m.start() if m else 0
            out.append(RetrievedPassage(doc_id, text[:500], round(s, 3),
                                        (start, start + min(500, len(text)))))
        return out

    @classmethod
    def from_jsonl(cls, path: str) -> "DocumentIndex":
        """Build an index from {"id","text",...} JSONL."""
        idx = cls()
        for line in open(path):
            if line.strip():
                d = json.loads(line)
                idx.add(d.get("id", str(len(idx.docs))), d.get("text", ""), d)
        return idx


def _content_tokens(s: str) -> set[str]:
    toks = set(re.findall(r"[a-zA-Z\u00c0-\u024f]{3,}", s.lower()))
    return toks - {"the", "and", "with", "from", "that", "this", "ndi", "ndiwo"}


def select_evidence(passages: list[RetrievedPassage], query: str,
                    max_sents: int = 3) -> list[tuple[str, RetrievedPassage]]:
    """Pick the sentences most overlapping the query (extractive QA).

    Returns (sentence, passage) pairs; the caller quotes them with the
    passage's [doc_id:start-end] marker. Deterministic, no model needed —
    this is what makes grounded answers useful even with a weak generator.
    """
    q = _content_tokens(query)
    scored: list[tuple[int, str, RetrievedPassage]] = []
    for p in passages:
        for sent in re.split(r"(?<=[.!?])\s+", p.text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            scored.append((len(q & _content_tokens(sent)), sent, p))
    scored.sort(key=lambda t: -t[0])
    return [(s, p) for o, s, p in scored[:max_sents] if o > 0]
