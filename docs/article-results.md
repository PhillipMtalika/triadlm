# Same 30M Brain, Four Behaviors: What Alignment and Tools Actually Change

*TriadLM results post — every number traces to a run_id in `experiments/`.*

## Thesis

A chatbot's personality is not the model. We trained one 29.6M decoder-only
model on 30M EN/Chichewa tokens ($0: Kaggle T4 + Colab-tier free services),
froze it, then built three descendants without touching the backbone: SFT,
constitutional alignment (critique-and-revision + DPO), and grounded tool use.
Same 60 probes for all four. This is what each layer buys — and costs.

## The table (no invented numbers)

| variant | factuality | instruction | citation acc | over-refusal | ny gap | run_id |
|---|---|---|---|---|---|---|
| base | 0.033 | 0.0 | 0.0 | 0.0 | 0.077 | 20260926131203 |
| constitutional | 0.20 | 0.167 | — | 0.146 | 0.077 | 20260926140726 |
| dpo | 0.0 | 0.0 | — | 0.0 | 0.0 | 20260926141400 |
| grounded | 0.033 | — | 1.0 | 0.0 | 0.077 | 20260926141623 |

Base perplexity 38.5 on held-out text: the model learned the language; it had
never been taught to answer. Constitutional CSFT gave 6x factuality and real
instruction-following — plus an alignment tax (over-refusal 0.146). DPO on 15
pairs regressed to zero: preference learning starved of data is worse than
none. Tools don't make prose smarter; they make guarantees: citations 1.0,
refusal precision 1.0, and (now) evidence quoted before the model speaks.

## What failed (the interesting part)

- Base loops and SFT parrots template markers — 37 demos teach format, not knowledge.
- Constitutional mode-collapses onto single sentences with only 20 triples.
- Grounded cited the wrong doc when the index lacked the passage; fixed with a
  79-passage Malawi index + extractive quoting, not a bigger model.
- Wikipedia throttles burst traffic: live grounding is demo-only, evals stay local.

## Limitations (read before citing)

30M params, 30M tokens, synthetic preferences, Chichewa still a minority of
the corpus, refusal/citation metrics are rule-based. Diagnostic, not a ranking
of anything commercial. Nothing here reproduces any proprietary system.

## Next

2k-pair SFT, DPO retry with 10x pairs + beta sweep, tokenizer/mixture
ablations — all on the same $0 rails. Code, manifests, probes, and logs are
public; rerun anything.
