# Evaluation methodology

## Dataset (`data/evaluation/`)
29 fixed questions (`Q01`-`Q29`): 23 answerable, 6 not answerable. Types: factual, pdf_page (page-level evidence), multi_document, metadata_filter, synthesis, unanswerable (in-domain distractors), unanswerable_out_of_domain, filter_excludes_evidence (answer exists but the metadata filter excludes it).
Files: `questions.json`, `gold_answers.json` (short gold answers), `relevance_labels.json`.

### How relevance labels were made - and their limits
Labels are **evidence phrases** (source file, optional page, <=10-word verbatim phrase, grade 2=primary / 1=supporting). A chunk is relevant if it comes from that source/page and contains the phrase (whitespace/case-normalised). This keeps labels valid under any chunking. Any phrase of <= `overlap` words lies wholly inside at least one chunk, so no question loses its evidence at the tested overlaps (15/30/60); a unit test enforces phrase presence and length.
**The corpus, questions, gold answers and labels were all written by the same author (the AI assistant that built this project, reviewed by nobody else). They are a consistent yardstick, not independently validated ground truth**, and the corpus is tiny (12 files, 25 chunks at default settings), which makes retrieval easier than in real deployments.

## Retrieval metrics (`app/evaluation/retrieval_metrics.py`, k stated in every output)
Aggregation: macro-average over *eligible* queries = answerable with >=1 relevant chunk among filter-eligible chunks (23 of 29; excluded ids are listed).
- **Hit Rate@k** = share of queries with >=1 relevant chunk in top-k.
- **Recall@k (chunk-level)** = relevant retrieved in top-k / all relevant chunks in the (filter-eligible) corpus. Overlapping chunks can duplicate an evidence phrase, which caps chunk recall, so we also report **evidence recall@k** = labelled evidence items covered by top-k / items.
- **MRR@k** = mean of 1/rank of first relevant chunk in top-k (0 if none).
- **nDCG@k** = DCG/IDCG with linear gain = grade, discount 1/log2(rank+1); IDCG from the corpus's relevant grades. Cross-checked against `sklearn.metrics.ndcg_score` in tests. Binary-vs-graded: grades 2/1 only exist for synthesis question Q23; elsewhere effectively binary.
- **Context precision@k** = relevant chunks in top-k / chunks retrieved (empty retrieval -> 0). With 1-2 relevant chunks per question, precision@5 is bounded near 0.2-0.4 by construction.
All metric functions have hand-calculated unit tests.

## Abstention
`MIN_RELEVANCE_SCORE` gates on top-1 **cosine similarity** (not a probability). `python -m app.cli calibrate` picks the balanced-accuracy-maximising threshold and reports leave-one-out accuracy as the honest estimate. The threshold is model-specific; **the MiniLM threshold has not been calibrated here** (model unavailable in the build sandbox).

## Answer quality (needs a Groq key)
- **Faithfulness / answer relevance**: LLM judge (`JUDGE_MODEL`, separate from `GROQ_MODEL`), 1-5 integer rubric (anchors in `app/evaluation/judge.py`), normalised to 0-1, with supported/unsupported claim lists. Malformed/out-of-range judge output is recorded as unavailable. Judge scores are an automated proxy with their own bias/noise, not human truth; the report warns if judge == generator model.
- **EM / token-F1** vs short gold answers (SQuAD-style normalisation). They measure string overlap, not correctness: a correct full sentence scores low EM/F1, so a lenient `contains_gold` is reported separately and labelled as such.
- **Citation precision** = share of cited chunks that are labelled relevant. **Abstention behaviour** = answered/abstained counts split by answerable/unanswerable.
Without a key the stage is written as `not_run` with the reason; no values are invented.

## Latency (`app/evaluation/latency.py`)
Monotonic `time.perf_counter`; single thread; cold start (first query) reported separately; warm-up passes untimed; every sample saved to `latency_samples.csv`; percentiles by linear interpolation (verified equal to numpy). *Retrieval latency* = query embedding + filtered ANN search (+ result assembly); *end-to-end* additionally includes prompt building and the Groq call. p95 flagged unreliable below 100 samples.

## Chunking experiment
80/15 vs 180/30 vs 400/60 words, same questions/embedder/k, fresh store per config (`results/chunking_experiment.json`). Small corpus => differences rest on a few queries.
