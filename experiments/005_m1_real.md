# 005 — M1 real 50M training on corpus_v2 (Kaggle free T4)

Falsifiable question: does the 8L/512 architecture learn (not memorize) on
30M mixed EN/Chichewa tokens with the spec pipeline?

Run: train run_id `20260926125035-0a09` (configs/m3_50m.yaml, 4000 steps,
1h38m, fp16, batch 16 x accum 2 x 512 ctx, 65.5M training tokens, 29.6M params).

Result: train 9.12 -> 3.63, val 5.62 -> 3.66, ppl 275 -> 38.7. Train and val
track together (gap 0.03) — genuine learning, no overfit. First model in this
project with a real held-out curve.

Decision: freeze this checkpoint as the M1 base. Next: eval all variants
(SFT/constitutional/grounded need building on THIS base, notebook #4), then
Spaces demo. Longer training (more epochs / bigger corpus) only if eval shows
underfitting.
