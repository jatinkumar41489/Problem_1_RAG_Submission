# Evaluation summary

Generated 2026-10-18T09:00:40+00:00 | dataset: 29 questions (23 answerable) | embedding: **sentence-transformers / sentence-transformers/all-MiniLM-L6-v2** (384-d) | chunks: 180 words / 30 overlap | k = 5

## Stage status

| Stage | Status |
|---|---|
| Retrieval metrics | completed |
| Abstention calibration | ok |
| Answer quality (faithfulness / relevance / EM / F1) | completed |
| Latency benchmark | completed |
| Chunking experiment | completed |
| Cost comparison | see cost_comparison.md |

## Retrieval (k = 5, 23 eligible queries)

| Metric | Value |
|---|---|
| Hit Rate@5 | 1.000 |
| Recall@5 (chunk-level) | 1.000 |
| Evidence recall@5 | 1.000 |
| MRR@5 | 0.935 |
| nDCG@5 | 0.947 |
| Context precision@5 | 0.220 |

All k:

| k | Hit | Recall | EvRecall | MRR | nDCG | CtxPrec |
|---|---|---|---|---|---|---|
| 1 | 0.870 | 0.826 | 0.848 | 0.870 | 0.870 | 0.870 |
| 3 | 1.000 | 0.978 | 0.978 | 0.935 | 0.935 | 0.348 |
| 5 | 1.000 | 1.000 | 1.000 | 0.935 | 0.947 | 0.220 |
| 10 | 1.000 | 1.000 | 1.000 | 0.935 | 0.947 | 0.120 |

## Abstention

Configured threshold 0.3: answerable passed 23, answerable wrongly blocked 0, unanswerable blocked 2, unanswerable passed to the LLM 4 (the LLM prompt is the second line of defence).

Calibration suggests threshold **0.4633** (in-sample balanced accuracy 0.9783, leave-one-out accuracy 0.931). Threshold is specific to the embedding model; scores of different models are not comparable. Calibrated on the same small set it is evaluated on - use the LOO figure as the honest estimate.

## Answer quality

Generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b`.

| Metric | Value |
|---|---|
| Faithfulness (judge, 0-1) | 1.000 (n=11) |
| Answer relevance (judge, 0-1) | 1.000 |
| Exact match | 0.000 |
| Token F1 | 0.178 |
| Contains-gold (lenient) | 0.435 |
| Citation precision | 0.955 |

## Retrieval latency

p50 **36.316 ms**, p95 **61.063 ms** over 145 samples (k=5); cold start 51.54 ms. Embedding p50 28.082 ms, vector search p50 8.069 ms. 

## Chunking experiment

| size/overlap | chunks | Hit | MRR | nDCG | CtxPrec | p50 ms |
|---|---|---|---|---|---|---|
| 80/15 | 53 | 1.000 | 0.928 | 0.927 | 0.235 | 26.681 |
| 180/30 | 25 | 1.000 | 0.935 | 0.947 | 0.220 | 23.374 |
| 400/60 | 16 | 0.957 | 0.957 | 0.953 | 0.219 | 22.604 |

- Only 25-ish chunks exist at the default size, so differences between configs rest on a handful of queries.
- Word-based chunk sizes; the 400-word config would exceed MiniLM's 256 word-piece input limit (silent truncation) with the real model, an effect a hashed bag-of-words encoder cannot show.
- Not evidence that one configuration is universally best.
