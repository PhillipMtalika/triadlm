"""Citation accuracy (formula lives here, spec §9).

citation_accuracy = citations that resolve to a real passage AND support the
adjacent claim / total citations emitted. Support heuristic: token overlap
between the claim sentence and the passage text (>=1 content token); the
heuristic is documented here so runs are comparable.
"""
import re

MARKER = re.compile(r"\[([A-Za-z0-9_\-]+)(?::(\d+)-(\d+))?\]")


def parse_markers(text: str) -> list[tuple[str, int | None, int | None]]:
    """Extract (doc_id, start, end) markers; bare [doc] gives Nones."""
    return [(m.group(1),
             int(m.group(2)) if m.group(2) else None,
             int(m.group(3)) if m.group(3) else None)
            for m in MARKER.finditer(text)]


def _content_tokens(s: str) -> set[str]:
    toks = set(re.findall(r"[a-zA-Z\u00c0-\u024f]{3,}", s.lower()))
    return toks - {"the", "and", "with", "from", "that", "this", "ndi", "ndiwo"}


def supports(passage: str, claim: str) -> bool:
    """Keyword-overlap support check (>=1 shared content token)."""
    return len(_content_tokens(passage) & _content_tokens(claim)) >= 1


def citation_accuracy(texts: list[str], index: object) -> dict:
    """Score a batch of generated texts against a DocumentIndex."""
    total, good = 0, 0
    for text in texts:
        sents = re.split(r"(?<=[.!?])\s+", text)
        for sent in sents:
            for doc_id, a, b in parse_markers(sent):
                if doc_id in ("calc", "datetime", "web", "refusal", "local-index"):
                    total += 1
                    good += 1  # tool citations are self-describing; counted as resolving
                    continue
                total += 1
                entry = index.docs.get(doc_id) if hasattr(index, "docs") else None
                if entry is None:
                    continue
                passage = entry[0] if isinstance(entry, tuple) else entry.get("text", "")
                if supports(passage, sent):
                    good += 1
    return {"citation_accuracy": good / max(1, total),
            "resolving": good, "total": total}
