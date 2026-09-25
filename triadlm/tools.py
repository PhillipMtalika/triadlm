"""Tool interface + shipped tools + tool-use policy (spec §8.1, §8.3)."""
import ast
import operator as op
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol

from .retrieval import DocumentIndex

Action = Literal["answer_directly", "retrieve_then_answer",
                 "retrieve_verify_answer", "refuse"]


@dataclass
class Citation:
    doc_id: str
    span: tuple[int, int] | None = None
    url: str | None = None
    timestamp: str | None = None

    def render(self) -> str:
        """Inline marker: [doc_id:start-end] (spec §8.2)."""
        if self.span is not None:
            return f"[{self.doc_id}:{self.span[0]}-{self.span[1]}]"
        return f"[{self.doc_id}]"


@dataclass
class ToolResult:
    output: str
    citation: Citation | None
    error: str | None = None


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict

    def call(self, **kwargs: object) -> ToolResult: ...


class CalculatorTool:
    """Typed numeric input; unit-tested edge cases (div-by-zero, floats)."""
    name = "calculator"
    description = "Evaluate an arithmetic expression."
    input_schema = {"type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]}

    _OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
            ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
            ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
    _ALLOWED = {ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                ast.USub, ast.UAdd, ast.FloorDiv, ast.Load}

    def call(self, **kwargs: object) -> ToolResult:
        expr = str(kwargs.get("expression", "")).replace("^", "**")
        try:
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if type(node) not in self._ALLOWED:
                    raise ValueError(f"disallowed: {type(node).__name__}")
                if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                    raise ValueError("only numbers allowed")

            def _ev(n: ast.AST) -> float:
                if isinstance(n, ast.Expression):
                    return _ev(n.body)
                if isinstance(n, ast.Constant):
                    return n.value
                if isinstance(n, ast.BinOp):
                    return self._OPS[type(n.op)](_ev(n.left), _ev(n.right))
                if isinstance(n, ast.UnaryOp):
                    return self._OPS[type(n.op)](_ev(n.operand))
                raise ValueError("bad node")

            val = _ev(tree)
            return ToolResult(f"{expr} = {val}",
                              Citation("calc", None), None)
        except Exception as e:
            return ToolResult("", Citation("calc", None), f"calc-error: {e}")


class DateTimeTool:
    """Current date/time in a fixed format, no network."""
    name = "datetime"
    description = "Return the current UTC date/time."
    input_schema: dict = {"type": "object", "properties": {}}

    def call(self, **kwargs: object) -> ToolResult:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return ToolResult(now, Citation("datetime", None), None)


class DocumentSearchTool:
    """Wrap retrieval.py's local index; returns passages with id + offset."""
    name = "doc_search"
    description = "Search the local document index."
    input_schema = {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index

    def call(self, **kwargs: object) -> ToolResult:
        hits = self.index.search(str(kwargs.get("query", "")), k=3)
        if not hits:
            return ToolResult("no local results", Citation("local-index", None), None)
        lines = [f"{h.doc_id}:{h.offset[0]}-{h.offset[1]}: {h.text[:200]}" for h in hits]
        first = hits[0]
        return ToolResult("\n".join(lines),
                          Citation(first.doc_id, first.offset), None)


class WebRetrievalTool:
    """Optional web adapter. Records {url, timestamp, query, passage} on EVERY
    call — including failures. Offline stub unless allow_network=True."""
    name = "web_retrieval"
    description = "Fetch a URL from the allowlist."
    input_schema = {"type": "object",
                    "properties": {"url": {"type": "string"},
                                   "query": {"type": "string"}},
                    "required": ["url"]}

    def __init__(self, log_path: str = "experiments/runs/web_log.jsonl",
                 allow_network: bool = False,
                 allowlist: list[str] | None = None) -> None:
        self.log_path = log_path
        self.allow_network = allow_network
        self.allowlist = allowlist or []

    def _log(self, query: str, url: str, status: str, passage: str = "") -> None:
        import json
        import os
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"url": url,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "query": query, "passage": passage[:500],
                                "status": status}) + "\n")

    def call(self, **kwargs: object) -> ToolResult:
        import json  # noqa: F401  (kept explicit: logging schema is JSON)
        url, query = str(kwargs.get("url", "")), str(kwargs.get("query", ""))
        if not self.allow_network or not any(url.startswith(a) for a in self.allowlist):
            self._log(query, url or "stub", "stubbed-offline")
            return ToolResult("web disabled (offline stub).",
                              Citation("web", None, url or "stub",
                                       datetime.now(timezone.utc).isoformat()),
                              "disabled")
        try:
            import requests
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            text = r.text[:2000]
            self._log(query, url, "ok", text)
            return ToolResult(text, Citation("web", None, url,
                                             datetime.now(timezone.utc).isoformat()), None)
        except Exception as e:
            self._log(query, url, f"error:{e}")
            return ToolResult("", Citation("web", None, url,
                                           datetime.now(timezone.utc).isoformat()), str(e))


_REFUSE = re.compile(r"social security|password|home address|phone number|dox|credit card", re.I)
_ARITH = re.compile(r"\d\s*[-+*/^%]\s*\d|calculate|plus|minus|times|divided", re.I)
_FACT = re.compile(r"\b(when|who|where|source|cite|liti|chifukwa|ndani)\b", re.I)
_INJECT = re.compile(r"ignore (previous|above)|system\s*:|jailbreak", re.I)


def decide_action(query: str, model: object = None, tok: object = None) -> Action:
    """Rule-based policy first (spec §8.3); swap for a trained classifier only
    if this is a measured bottleneck."""
    _ = (model, tok)
    q = query.strip()
    if _REFUSE.search(q):
        return "refuse"
    if _INJECT.search(q):
        return "retrieve_verify_answer"
    if _ARITH.search(q):
        return "retrieve_verify_answer"
    if _FACT.search(q) or "?" in q:
        return "retrieve_then_answer"
    return "answer_directly"
