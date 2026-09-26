# 006 — alignment comparison on the M1 base (Kaggle free T4)

Falsifiable question: does each alignment stage (SFT → constitutional/CSFT →
DPO → grounded tools) move the 60-probe metrics vs the raw base?

Runs (checkpoints all in `phillipmtalika/triadlm-m1-50m`):
- base: `20260926131203-12b5` — factuality 0.033, instruction 0.0
- constitutional: `20260926140726-7ba9` — factuality 0.20, instruction 0.167,
  over_refusal 0.146, gap 0.077
- dpo: `20260926141400-4517` — factuality 0.0, instruction 0.0,
  over_refusal 0.0, gap 0.0
- grounded (off dpo): `20260926141623-c680` — factuality 0.033,
  citation_accuracy 1.0, refusal_precision 1.0, gap 0.077
- sft-only eval: not separately recorded (rerun cheap: m1align/sft.pt)

Result: constitutional CSFT is the clear winner (6x factuality, instruction
off zero) at the cost of over-refusal 0.146 (alignment tax). DPO on 15 pairs
REGRESSED to 0.0 — likely too few pairs + untuned beta on a weak generator,
not a verdict on DPO itself. Grounded tools guarantee citations (1.0) and
perfect refusal precision regardless of generator quality.

Decision: ship constitutional + grounded as the demo branches; revisit DPO
with 10x pairs + beta sweep. Next: Spaces demo (notebook #5 = deploy, not
training).
