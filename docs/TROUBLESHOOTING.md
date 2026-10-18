# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not load embedding model ...` | First run downloads the model from huggingface.co; check internet/proxy or pre-download. Offline smoke test only: `$env:EMBEDDING_BACKEND="hash"` (not for reporting). |
| `Collection ... was created with embedding model X` | Collection bound to another model/dim. Use a new `CHROMA_COLLECTION_NAME` or `CHROMA_PERSIST_DIRECTORY`; nothing was modified. |
| `/query` -> 503 `configuration` | `GROQ_API_KEY` missing. Set it in `.env` or `$env:GROQ_API_KEY`. Retrieval still works (`/retrieve`). |
| `model_not_found` | Model id not available to your key. `python -m app.cli verify --list-models`, then set `GROQ_MODEL`/`JUDGE_MODEL`. |
| `rate_limit` (429) | Bounded retries with backoff already applied; slow down or reduce eval size (`--skip-answers`). |
| Everything abstains | `MIN_RELEVANCE_SCORE` too high for the model; run `python -m app.cli calibrate`. |
| `no extractable text` for a PDF | Scanned PDF; OCR is not implemented. |
| `path_not_allowed` | `/ingest` only reads inside `ALLOWED_INGEST_ROOTS` (default `data/`). |
| Windows: PermissionError deleting `chroma_data` | Stop the server first (SQLite handles are open); tests call `store.close()`. Windows was not tested by the author of this build. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
