# Screenshots required by the submission template (section 1.5)

**No screenshots are included, on purpose:** the build sandbox had no access to the real embedding model or the Groq API, so any "end-to-end query with token usage" image would have been faked. Capture them yourself on your machine:

1. Activate the venv, put your key in `.env`, then run `.\scripts\capture_evidence.ps1`.
2. **Screenshot 1 - end-to-end query** must visibly show: the question, the grounded answer, the citations, `latency: retrieval=... | generation=... | end-to-end=...`, `retrieved_chunks=N`, and `tokens: prompt=... completion=... total=...`. (The JSON log line printed to the terminal contains the same fields.) If you used the HTTP API instead, screenshot the `Invoke-RestMethod` response showing `latency`, `retrieved_chunk_count`, `token_usage`.
3. **Screenshot 2 - idempotent re-ingest** must show `vectors = N` before and the same `N` after the second ingest (output of `python scripts/demo_idempotency.py`).
4. Paste both into the dashed boxes of the template.

`evidence_*_offline_hash_backend.txt` in this folder are **real captured text outputs from the build sandbox** (offline lexical embedder, no Groq key). They show the pipeline behaviour (idempotency, API, filters, path-traversal block, missing-key handling) but are not a substitute for your own screenshots, and the query one deliberately shows `token_usage.available=false`.
