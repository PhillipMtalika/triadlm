# 008 — extractive grounded answers (tiny pilot)

Problem: the grounded branch let a weak generator rewrite evidence into mush
(demo: Mzuzu query cited chichewa-1; loops; template echo).

Change: evidence-first composition — `select_evidence()` ranks sentences by
query overlap (deterministic, model-free); outputs lead with quoted sentences
+ `[doc:start-end]` markers, model text demoted to a "note". Live Wikipedia
lead via `WikipediaTool`/`WebSearchTool` (free, logged, 1s pacing + retry,
arithmetic excluded) appended as `[wiki]`; eval harness keeps web OFF by
default (`--web` opts in) for speed/determinism.

Runs (tiny sft base, offline, 79-doc index):
- grounded-extractive: run_id `20260927180256-b1ea` — factuality 0.10,
  citation_accuracy 0.85, refusal_precision 1.0.
- Local check: "where is Mzuzu" -> Mzuzu passage top hit with correct quotes.

Decision: ship extractive ordering in the demo (sampling 0.7/top-40/rep-1.15
stays for the note). Full M1 re-eval of grounded after the SFT scale-up.
Caveat: per-probe live web is too slow/flaky for CI while Wikipedia throttles
burst traffic — demo-only until pacing proves out.
