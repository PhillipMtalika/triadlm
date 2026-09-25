# Threat model — TriadLM

1. **Prompt injection via docs/tools** — tool and document content is untrusted data,
   never instructions. Mitigations: `retrieve_verify_answer` path, safety hooks
   (`triadlm/safety.py`), adversarial `prompt_injection` probes, web adapter logs
   every call (url/timestamp/query/passage) even on failure.
2. **Privacy leakage** — model must refuse personal-data requests. Mitigations:
   constitutional privacy principle, refusal classifier, `privacy` probes +
   refusal_precision metric, DPO negatives.
3. **Fabricated citations** — markers that resolve nowhere. Mitigations: enforced
   `[doc_id:start-end]` format, `score_citations.py` resolution+support checks.
4. **Unsafe assistance** — dangerous how-tos. Mitigations: refuse + safe alternative
   (principle 4), `safety` probes, over-refusal tracked separately so helpfulness
   regressions are visible.
5. **Metric gaming** — hardcoded/fabricated numbers. Forbidden by guardrail: every
   number in `experiments/` traces to a `run_id`; registry is append-only.
