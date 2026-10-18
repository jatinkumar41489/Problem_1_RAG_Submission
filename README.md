# Cost-Efficient RAG on ChromaDB (Problem 1)

A grounded, cited question-answering service over PDF/HTML/Markdown, backed by a low-cost embedded vector store (ChromaDB), with an evaluation harness for retrieval quality, answer quality, latency and cost.

> **Honest status.** The build environment could reach neither huggingface.co nor the Groq API. Everything that does not need them is implemented, executed and tested (134 tests passing, 2 optional live tests skipped). The real `all-MiniLM-L6-v2` encoder and Groq generation/judging are **implemented but have not been run**. All results shipped in `results/` were produced with an **offline lexical `hash` embedder** and are labelled as such; the answer-quality stage is `not_run`. Run `scripts\run_all.ps1` once on your machine to produce the real numbers (last section).

## 1-3. Overview, problem, objectives
Managed vector DBs bill by stored vectors; this project builds a QA service on a low-cost store and measures whether it is a credible alternative. It ingests documents idempotently, retrieves top-k chunks with metadata filters, answers only from retrieved evidence with citations, abstains when evidence is missing, exposes an HTTP API + CLI, and evaluates retrieval / answers / latency / cost. See `docs/ARCHITECTURE.md` for the diagram (Mermaid) and module map.

```
PDF/HTML/MD -> loaders -> chunker (180w/30w) -> embedder (MiniLM 384-d) -> ChromaDB (vectors + text + metadata)
query -> embed -> filtered top-k -> relevance gate --no--> "no_relevant_context" (LLM not called)
                                          |yes
                                          v
                          Groq LLM (JSON: answerable/answer/citations) -> validate citations -> response
```

## 4-5. Stack
Python 3.11/3.12 (built on 3.12.3) - ChromaDB (persistent, cosine) - sentence-transformers all-MiniLM-L6-v2 (384-d) - Groq (LLM + judge, separate model ids) - FastAPI/Uvicorn - pypdf, BeautifulSoup4, Markdown - pytest. Pinned in `requirements.txt` (all except `sentence-transformers` were tested).

## 6-9. Prerequisites, install (Windows PowerShell), configuration
```powershell
# Python 3.11 or 3.12 installed. From the project folder:
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
# optional, smaller download than the default CUDA build of torch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
Copy-Item .env.example .env
notepad .env                          # set GROQ_API_KEY (never commit .env)
python -m app.cli verify --list-models   # checks config, 384-d embeddings, and that GROQ_MODEL / JUDGE_MODEL exist for your key
```
Alternative to a `.env` key for the current session only:
```powershell
$sec = Read-Host "Groq API key" -AsSecureString
$env:GROQ_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
```
Groq model ids change; if `verify` reports a missing model, set `GROQ_MODEL` / `JUDGE_MODEL` (use different model families for generator and judge). The app never silently switches models. All settings are documented in `.env.example`. The PowerShell commands here were not executed on Windows by the author of this build (CLI/server logic was run on Linux).

## 10-14. Formats, chunking, embeddings, persistence
Formats: `.pdf` (per page, page numbers kept; scanned PDFs are reported, **OCR not implemented**), `.html/.htm` (scripts/styles/nav removed), `.md/.markdown` (front matter dropped, headings kept). Chunking: **180 words, 30 overlap** (word = whitespace token; chosen because MiniLM truncates at 256 word-pieces). Embedding: `sentence-transformers/all-MiniLM-L6-v2`, 384-d, L2-normalised, verified by `verify`. Persistence: `CHROMA_PERSIST_DIRECTORY` (default `chroma_data/`), cosine space, one server process per directory.

## 15-16. Ingestion
```powershell
python -m app.cli ingest --path data/sample_documents          # add --prune to delete chunks of files removed from disk
python scripts/demo_idempotency.py                              # shows vector count before/after repeated ingest
```
Actual output (sandbox, hash backend):
```
[start]            vectors = 0
[ingest run #1]    vectors = 25 | new=12 updated=0 unchanged=0 upserted=25
[ingest run #2]    vectors = 25 | new=0 updated=0 unchanged=12 upserted=0
idempotent: YES - count unchanged after re-ingest
```
Identity: `chunk_id = sha256(source_id + content)[:12] + "-" + index`; unchanged files are skipped without embedding; modified files are upserted then their stale chunks deleted; same basename in different folders never collides; identical content under two paths is stored twice with a warning. Add your own corpus: put files in a folder under `data/` (or extend `ALLOWED_INGEST_ROOTS`) and ingest it.

## 17-21. API, citations, no-context behaviour, filters
```powershell
uvicorn app.main:app --port 8000          # docs: http://127.0.0.1:8000/docs
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ingest -ContentType "application/json" -Body '{"path":"data/sample_documents"}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/query -ContentType "application/json" `
  -Body '{"query":"What is the maximum hotel cost per night in tier-1 cities?","k":5}' | ConvertTo-Json -Depth 6
# metadata filter (fields: category, document_type, source_id, source_name):
Invoke-RestMethod -Method Post http://127.0.0.1:8000/query -ContentType "application/json" `
  -Body '{"query":"How quickly must a SEV1 page be acknowledged?","filters":{"category":"engineering"}}' | ConvertTo-Json -Depth 6
```
Full field reference: `docs/API.md`. A real captured session (offline backend) is in `screenshots/evidence_api_session_offline_hash_backend.txt`; it includes the 403 path-traversal block and the 503 `configuration` response when no Groq key is set.
**Citations**: each cited chunk is returned with `chunk_id`, source, page (PDF) / section (HTML, MD) and an excerpt copied from the chunk; refs the model invents are listed in `invalid_citations` and never shown as valid.
**No-context behaviour**: (1) empty store / filter excludes everything / top cosine < `MIN_RELEVANCE_SCORE` -> `no_relevant_context`, LLM not called; (2) otherwise the model must return `answerable` + valid citations, else `insufficient_evidence`; (3) an LLM outage is `generation_error` with the retrieved chunks still returned.

## 22-24. CLI, tests, evaluation commands
```powershell
python -m app.cli query "How many days of paid annual leave do employees get?" -k 5
python -m app.cli query "..." --category hr --retrieve-only        # no LLM needed
python -m app.cli stats
python -m app.cli calibrate            # recommended MIN_RELEVANCE_SCORE for YOUR embedder
python -m app.cli evaluate             # retrieval + answers + latency + chunking experiment + cost -> results/
python -m app.cli evaluate --skip-answers --skip-chunking
python -m app.cli benchmark            # latency only
python -m app.cli cost                 # cost tables only
python -m pytest tests                 # offline; mocks Groq. Live tests: $env:RUN_LIVE_GROQ="1"; python -m pytest tests/integration/test_live_groq.py
```
Saved test run: `results/test_results.txt` (134 passed, 2 skipped, Linux, offline).

## 25. Metric definitions
See `docs/EVALUATION_METHODOLOGY.md` (Recall@k chunk- and evidence-level, Hit Rate, MRR, nDCG with linear gain, context precision, EM/F1, faithfulness/relevance judge rubric, citation precision).

## 26. Actual evaluation results (SANDBOX: hash lexical embedder, NOT MiniLM)
Dataset: 29 questions (23 answerable), 12 files, 25 chunks (180/30), k=5, `MIN_RELEVANCE_SCORE=0.15`.

| Metric @5 | Value |
|---|---|
| Hit Rate | 1.000 |
| Recall (chunk) | 0.978 |
| MRR | 0.859 |
| nDCG | 0.878 |
| Context precision | 0.211 |

At k=1: hit 0.739 / MRR 0.739 / nDCG 0.739. Abstention gate at 0.15: 20/23 answerable passed, 3 wrongly blocked; 5/6 unanswerable blocked, 1 (Q24, pets) reached the LLM. Threshold calibration leave-one-out accuracy 0.79. Answer quality (faithfulness, relevance, EM, F1): **not run** (no Groq access). Full tables: `results/evaluation_summary.md`.

## 27-29. Cost comparison, assumptions, latency
Estimated monthly cost (model, not bills; details/formulas in `docs/COST_ASSUMPTIONS.md` and `results/cost_comparison.md`):

| Vectors | Chroma on one AWS node (infra) | Pinecone serverless Standard | Comment |
|---|---|---|---|
| 100K | $70.14 | $50.00 (minimum) | managed cheaper |
| 1M | $70.70 | $50.00 | managed cheaper at 300K queries/mo (break-even ~1.4M q/mo) |
| 10M | $190.13 | $174.31 | roughly equal; self-hosted wins above ~0.33M q/mo |

**This does not show self-hosting is cheaper at low query volume** against serverless pricing; it quantifies the break-even. Always-on pod pricing (the assignment's framing) was not modelled because current prices could not be verified. Ops time (assumed $75/h) makes managed cheaper at the baseline.
Latency (sandbox, 1 vCPU, 25 vectors, hash embedder, 145 samples): retrieval **p50 3.05 ms, p95 4.76 ms** (embed 0.39 + search 2.68 ms median). This is a toy corpus and a trivial encoder; real MiniLM query embedding on CPU is typically several to tens of ms, so re-measure. End-to-end p95 requires Groq.

## 30. Known limitations
Real MiniLM/Groq results absent (see top); tiny author-written corpus and labels (not independent ground truth); chunking experiment only with the lexical embedder; one stored copy per source path; no OCR; single-writer Chroma; the relevance threshold must be re-calibrated per embedding model (0.30 default is uncalibrated); LLM-judge scores are a biased proxy; no second vector store benchmarked; Windows not tested; no authentication on the HTTP API (do not expose it publicly).

## 31. Retrieval vs generation: which is the weak link?
**Evidence so far covers retrieval only.** With the lexical embedder: the relevant chunk is in the top 5 for every answerable question (hit 1.00) but at rank 1 for only 74%, and the abstention gate misclassifies about 1 in 5 questions - so even a good generator is capped by ranking and gating quality; the misses cluster on short, numeric PDF questions (Q08-Q10) where lexical overlap is weak. Whether the real MiniLM retrieval is better, and whether generation (faithfulness, citation precision, over/under-abstention) is the weaker link, **cannot be concluded until the answer-quality stage runs.** How to decide once it has: if hit@k is high but faithfulness/citation precision are low -> generation/prompt; if hit@k is low -> chunking/embedding/threshold; if many answerable questions are `insufficient_evidence` despite a relevant chunk in context -> prompt too conservative; compare `answer_quality_per_query.csv` against `retrieval_per_query.csv` question by question.

## 32-33. Trade-offs; when to switch back to managed
Trade-offs accepted: you own backups/patching/availability; single node; no auto-scaling; local-disk durability. Switch to a managed vector DB (or a server-based pgvector/Qdrant) when nobody can own ops, you need HA or multiple writers, the index outgrows one node's RAM (~10M x 384-d here), or traffic is below the computed break-even so the managed bill is lower.

## 34-36. Troubleshooting, security, reports
`docs/TROUBLESHOOTING.md`. Security: secrets only via env/.env (git-ignored), never logged or returned (`Settings.__repr__` masks the key; `/health` returns only a boolean); `/ingest` is confined to allowed roots (traversal tested); retrieved text is treated as untrusted in prompts; raw queries are not logged unless `LOG_QUERY_TEXT=true`; API has no auth. Reports: `results/` (`evaluation_summary.md`, `retrieval_metrics.json`, `retrieval_per_query.csv`, `answer_quality*.{json,csv}`, `latency_*`, `chunking_experiment.json`, `cost_comparison.{md,csv,json}`, `test_results.txt`); docs in `docs/`; submission mapping in `docs/SUBMISSION_TEMPLATE_GUIDE.md`.

## Producing the real results on your machine
```powershell
python -m app.cli verify --list-models      # fix GROQ_MODEL / JUDGE_MODEL if needed
python -m app.cli calibrate                 # set MIN_RELEVANCE_SCORE in .env to the recommended value
.\scripts\run_all.ps1                       # overwrites results/ with real MiniLM + Groq numbers
.\scripts\capture_evidence.ps1              # then screenshot the two terminal outputs
```
