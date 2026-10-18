# Mapping to `submission_template.docx` - Problem 1

> **Read first.** Numbers below come from the **build sandbox run with the offline `hash` lexical embedder** (no HuggingFace/Groq access there). The template says "fill in real numbers": re-run on your machine with `EMBEDDING_BACKEND=sentence-transformers` and a Groq key (`scripts/run_all.ps1`) and paste **those** numbers from `results/evaluation_summary.md`. Do not submit sandbox numbers as MiniLM results. Fields marked **YOU** must be written by you.

## Candidate details (YOU)
Name, email, date, repo link, time spent. *Problem 1 vector store chosen:* **Chroma** (ChromaDB persistent).

## 1.1 Architecture diagram
Use the Mermaid diagram in `docs/ARCHITECTURE.md` (render at mermaid.live or in GitHub, then paste as an image). It shows PDF/HTML/MD -> chunking -> embeddings -> vector store + metadata -> filtered top-k -> grounded answer with citations, plus the no-context branch.

## 1.2 Setup & run
- **Prerequisites:** Python 3.11/3.12, Windows 10/11 (or Linux/macOS), ~2 GB free disk (torch + model), internet on first run (model download), 8 GB RAM recommended. Built/tested on Linux, Python 3.12.3.
- **Environment variables (names only):** `GROQ_API_KEY, GROQ_MODEL, JUDGE_MODEL, GROQ_TIMEOUT, GROQ_MAX_RETRIES, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, EMBEDDING_BACKEND, EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MAX_TOP_K, MIN_RELEVANCE_SCORE, LOG_LEVEL, LOG_QUERY_TEXT, ALLOWED_INGEST_ROOTS, EVALUATION_DATASET_PATH, EVALUATION_OUTPUT_DIR, SAMPLE_DOCUMENTS_DIR, BENCHMARK_WARMUP, BENCHMARK_REPEATS`.
- **Install / ingest / query commands:** README sections 7-10. Defaults: **chunk size 180 words, overlap 30 words**.
- **Vector store + why:** Chroma - embedded, no server to run, built-in metadata filtering and id-based upsert/delete make idempotent ingestion simple; weaker than pgvector/Qdrant for multi-writer or >~10M vectors.

## 1.3.1 Retrieval metrics - k = 5, 23 eligible queries (sandbox, hash embedder - REPLACE with your MiniLM run)
| Metric | Value (sandbox) | How computed |
|---|---|---|
| Recall@5 / Hit Rate@5 | 0.978 / 1.000 | chunk-level recall; hit = any relevant chunk in top 5 |
| MRR@5 | 0.859 | mean 1/rank of first relevant chunk |
| nDCG@5 | 0.878 | linear gain, log2 discount, ideal from corpus relevant chunks |
| Context precision@5 | 0.211 | relevant/retrieved; "relevant" = chunk from labelled source/page containing the labelled evidence phrase |

## 1.3.2 Answer quality
Run live (needs Groq). Values come from `results/answer_quality.json`. Currently **`not_run`** (no key in sandbox). If, on your machine, a metric truly cannot be computed, write `N/A` + the one-line reason. Explain in Notes that EM/F1 compare against *short* gold answers so verbose correct answers score low (see `contains_gold`).

## 1.3.3 Cost comparison (`results/cost_comparison.md`)
| Vectors | Chroma self-hosted ($/mo, infra) | Managed - Pinecone serverless ($/mo) | Notes |
|---|---|---|---|
| 100K | 70.14 (m6i.large) | 50.00 (plan minimum) | managed cheaper; break-even ~14M queries/mo |
| 1M | 70.70 (m6i.large) | 50.00 | managed cheaper at 300K q/mo; break-even ~1.4M q/mo |
| 10M | 190.13 (r6i.xlarge, ~30 GiB RAM) | 174.31 | managed slightly cheaper at 300K q/mo; self-hosted wins above ~0.33M q/mo |

Say honestly that self-hosting is **not** cheaper at this query volume against serverless pricing. Assumptions box: copy the "Key assumptions" list from `results/cost_comparison.md` (384-d float32, 1.1 KB text + 0.45 KB metadata per vector, 300K queries/mo, 10% monthly rewrites, AWS us-east-1 on-demand prices from third-party aggregators, Pinecone Standard prices verified 2026-10-18, ops cost excluded from the headline numbers; unverified items are flagged in `docs/COST_ASSUMPTIONS.md`).

## 1.3.4 Latency (sandbox: 1 vCPU, 3.9 GB RAM, 25 vectors, hash embedder, k=5, 145 samples) - REPLACE with your run
| Metric | Value (ms) | Notes |
|---|---|---|
| Retrieval p50 | 3.05 | embed 0.39 + search 2.68 (medians); not representative of MiniLM (embedding is much slower) |
| Retrieval p95 | 4.76 | |
| End-to-end p95 | N/A here | needs Groq; `evaluate` records generation latency per query in `answer_quality_per_query.csv` |

## 1.4 Design decisions - evidence-backed notes (turn into your own short paragraphs)
Source material: `docs/DESIGN_DECISIONS.md` (store choice; 180/30 vs 800/120 and the 256-word-piece truncation argument; chunking experiment in `results/chunking_experiment.json`; MiniLM 384-d trade-off; two-layer no-context handling and its measured gate behaviour; idempotency design with test evidence; accepted trade-offs and switch-back conditions). Be accurate about what you tried: the build tried 80/180/400-word chunking, but only with the offline embedder.

## 1.5 Subtask evidence & reflection
- **Screenshots:** capture yourself - `screenshots/README.md`, `scripts/capture_evidence.ps1`.
- **Walk through one query (YOU, no AI):** trace `POST /query` through `api/routes.py` -> `AnswerService.answer` -> `Retriever.retrieve` -> `ChromaStore.query` -> relevance gate -> prompt -> `GroqClient.complete` -> JSON parse -> citation validation -> response. Write it in your own words.
- **Something that broke first time (YOU):** write a genuine incident you actually hit while running this on your machine. Do not copy one from here. (For reference only, the build hit real issues - e.g. my first cost model contradicted the pre-written narrative - but those are not yours.)
- **AI usage disclosure (edit to be true for you):** "The implementation, tests, sample corpus, evaluation questions, cost model and documentation were generated by Claude (Anthropic) from the assignment text; I [ran / modified / reviewed ...]. The reflection sections are my own." Keep only what is true.
