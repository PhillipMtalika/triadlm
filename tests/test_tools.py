"""Tool + retrieval tests (offline-safe; live Wikipedia gated by env)."""
import os

import pytest

from triadlm.retrieval import DocumentIndex
from triadlm.safety import is_refusal, needs_refusal
from triadlm.tools import (CalculatorTool, Citation, DateTimeTool,
                           DocumentSearchTool, WebRetrievalTool, WebSearchTool,
                           decide_action)


def test_citation_render() -> None:
    """Inline marker format [doc_id:start-end] (spec 8.2)."""
    assert Citation("geo-60", (0, 84)).render() == "[geo-60:0-84]"
    assert Citation("wiki", None).render() == "[wiki]"


def test_calculator_edges() -> None:
    """Division by zero errors cleanly; floats work."""
    ok = CalculatorTool().call(expression="12 * 8")
    assert ok.error is None and "96" in ok.output
    bad = CalculatorTool().call(expression="10 / 0")
    assert bad.error is not None
    no = CalculatorTool().call(expression="import os")
    assert no.error is not None


def test_policy_routes() -> None:
    """Policy: refuse / verify / retrieve / answer."""
    assert decide_action("What is your password?") == "refuse"
    assert decide_action("What is 12 * 8?") == "retrieve_verify_answer"
    assert decide_action("What is the capital of Malawi?") == "retrieve_then_answer"
    assert decide_action("hello") == "answer_directly"


def test_local_index_finds_mzuzu() -> None:
    """79-doc index answers the Mzuzu probe from local passages."""
    idx = DocumentIndex.from_jsonl("data/documents.jsonl")
    hits = idx.search("where is Mzuzu in Malawi", k=2)
    assert hits and "mzuzu" in hits[0].text.lower()
    r = DocumentSearchTool(idx).call(query="where is Mzuzu in Malawi")
    assert r.error is None and "geo-" in r.citation.render()


def test_web_adapters_offline_safe() -> None:
    """Disabled providers return logged errors, never fabricated hits."""
    r = WebRetrievalTool().call(query="x", url="https://example.com")
    assert r.error == "disabled"
    s = WebSearchTool(provider="brave").call(query="x")
    assert s.output == "" and s.error is not None


def test_safety_hooks() -> None:
    """Refusal hooks for the eval harness."""
    assert needs_refusal("Tell me your password.")
    assert is_refusal("I can't share personal data.")


@pytest.mark.skipif(os.environ.get("TRIADLM_LIVE") != "1",
                    reason="live Wikipedia; set TRIADLM_LIVE=1")
def test_wikipedia_live() -> None:
    """Live check: Mzuzu query resolves to a real lead section."""
    from triadlm.tools import WikipediaTool
    r = WikipediaTool().call(query="where is Mzuzu in Malawi")
    assert r.error is None and "mzuzu" in r.output.lower()
