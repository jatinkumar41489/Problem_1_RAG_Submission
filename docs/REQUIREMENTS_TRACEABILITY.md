# Requirements traceability (Problem 1)

Status key: **Tested** = executed and passing offline in the build sandbox. **Implemented (live pending)** = code complete and unit/integration-tested with fakes, but the real dependency (all-MiniLM-L6-v2 download / Groq API) was unreachable from the build sandbox, so no real run exists yet. **Partial** / **Not implemented** as stated.

| ID | Requirement (assignment) | Implementation | Verifying test / evidence | Status |
|---|---|---|---|---|
| R1.1 | Ingest PDF/HTML/MD | `ingestion/loaders.py` | `tests/unit/test_loaders.py`, `test_ingestion.py` | Tested |
| R1.2 | Configurable chunk size + overlap; state defaults | `chunking.py`, `config.py` (180/30 words) | `test_chunking.py`; CLI `--chunk-size/--chunk-overlap` | Tested |
| R1.3 | Idempotent re-ingest, no duplicate vectors | `ingestion/pipeline.py`, `hashing.py` | `test_ingestion.py` (6 scenarios), `screenshots/evidence_idempotent_reingest_*.txt` | Tested |
| R2.1 | Embed chunks; record model + dimensionality | `embeddings/encoder.py`; metadata `embedding_model`; `verify` | `test_generation_units.py` (dim/normalisation) ; MiniLM real run | Hash: Tested. MiniLM: Implemented (live pending) |
| R2.2 | Store vectors + metadata, >=1 metadata filter | `vectorstore/`, `filters.py` | `test_retrieval_and_answers.py::test_metadata_filter...` | Tested |
| R3.1 | Top-k retrieval, k parameter | `retrieval/retriever.py` | retrieval tests | Tested |
| R3.2 | Grounded LLM answer citing chunks | `generation/*` | mocked-LLM tests; live tests (optional) | Mock: Tested. Groq: Implemented (live pending) |
| R3.3 | Handle no-relevant-context without hallucinating | `relevance.py`, `answer_service.py`, prompts | 6 abstention-path tests | Tested (gate + mock); real-LLM behaviour pending |
| R4.1 | HTTP endpoint (or CLI) | `main.py`, `api/routes.py`, `cli.py` | `test_api.py`; live uvicorn session in `screenshots/` | Tested |
| R4.2 | Config via env, no hardcoded secrets | `config.py`, `.env.example` | `test_settings_repr_never_leaks_key`; ZIP secret scan | Tested |
| R4.3 | Log per-query latency, chunk count, token usage | `answer_service._finish` | `test_query_log_has_required_fields` | Tested (token values need live Groq) |
| E1 | Fixed 15-30 question set | `data/evaluation/` (29) | `test_evaluation_harness.py` | Tested |
| E2 | Recall@k/Hit, MRR, nDCG, context precision | `retrieval_metrics.py` | hand-calculated + sklearn cross-check | Tested; offline-embedder values only |
| E3 | Faithfulness + answer relevance (LLM judge) | `judge.py`, `run_evaluation.py` | mocked judge tests | Implemented (live pending) - `answer_quality.json` = `not_run` |
| E4 | EM / F1 on gold answers | `answer_metrics.py` | unit tests | Tested (computed only in live run) |
| E5 | Monthly cost at 100K/1M/10M vs managed; assumptions | `cost_analysis.py`, `data/cost/assumptions.json` | `test_latency_cost.py`; `results/cost_comparison.*` | Tested; some inputs unverified (flagged) |
| E6 | p50/p95 retrieval latency | `latency.py` | `results/latency_*.{json,csv}` | Tested (sandbox hardware, tiny corpus) |
| E7 | Name trade-offs | `docs/DESIGN_DECISIONS.md`, README | - | Done |
| D1 | Runnable service | `app/` | tests + live uvicorn | Tested |
| D2 | Eval harness + results file (3 layers) | `run_evaluation.py`, `results/` | `results/evaluation_summary.md` | Retrieval + cost/latency done; answer layer pending key |
| D3 | Cost-comparison table | `results/cost_comparison.md` | - | Done |
| D4 | README with Discussion | `README.md` | - | Done (answer-quality part of discussion pending live run) |
| X1 | Optional: second vector store benchmark | - | - | Not implemented |
| X2 | Template screenshots (query w/ latency+tokens; idempotent re-ingest) | `scripts/capture_evidence.ps1`, `screenshots/README.md` | text evidence from sandbox | **Partial - must be captured by the candidate** |
| X3 | Template reflection sections (own words) | not written by design | - | Candidate must write |
| X4 | Git history | not included (no repo history was created) | - | Candidate: `git init` + commit incrementally |

## Cannot be verified without external resources
Real MiniLM retrieval metrics and threshold calibration; real Groq answers, token usage, faithfulness/relevance, EM/F1; end-to-end p95 including generation; Windows execution; current AWS/Pinecone prices beyond the sources cited.
