# 002 — constitution ablation (sft vs constitutional, toy scale)

Falsifiable question: does one CSFT pass over 4 synthetic
(initial, critique, revised) triples change over-refusal or factuality vs SFT?

Runs:
- sft: run_id `20260925221237-6c6a`
- constitutional: run_id `20260925221415-6c5f`
  (data/constitutional_toy.jsonl, principles from docs/constitution.md)

Result: factuality 0.067 -> 0.033; over_refusal 0.0 -> 0.0; chichewa_gap -0.08 -> 0.0.
No signal at toy scale — the triples inherit the base model's gibberish, so CSFT
has nothing to distill. Decision: constitutional data generation requires a
competent generator (post-M1 base_50m); keep the pipeline (proven working) but do
not judge the method by toy numbers. Next: DPO run once M1 generator exists.
