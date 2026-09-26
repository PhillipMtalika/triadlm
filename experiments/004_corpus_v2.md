# 004 — corpus_v2 build (Kaggle free CPU)

Falsifiable question: can we build a license-clean 10M+ token EN/Chichewa
corpus on $0 infra?

Result: yes — 189,421 docs, 30,117,392 tokens (manifest `corpus_v1.json`
naming aside, file is `corpus_v2.json`):
- wikimedia/wikipedia EN stream (5000 articles) + full ny.wikipedia stream
- OPUS CCAligned en-ny parallel (~100k pairs, 13MB Chichewa)
- local seed (Wikipedia Malawi topics + demos)

License: CC-BY-SA (Wikipedia) + OPUS CCAligned (source licenses vary,
see opus.nlpl.eu/CCAligned-v1.php) + project-internal (seed).

Artifacts (free Hub dataset): `phillipmtalika/triadlm-corpus-v2`
(16 files, 83.7MB: multi-shard train/val + tokenizer + manifest).

Decision: 30M tokens ≈ 1 token/param for the 50M net — enough for a genuine
M1 run (m3_train_50m, 4000 steps ≈ 2.2 epochs). Chichewa still minority;
next corpus iteration adds JW300-style sources if an en-ny pair appears.
