# HTTP API  (`uvicorn app.main:app --port 8000`, interactive docs at `/docs`)

All errors: `{"error": {"code": "...", "message": "..."}}` - no stack traces, no secrets.

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | ingest a folder/file under an allowed root (default `data/`); body `{"path": "data/sample_documents", "prune": false, "source_prefix": ""}` |
| POST | `/query` | grounded answer with citations |
| POST | `/retrieve` | retrieval only (no LLM) - for debugging/benchmarks |
| GET | `/health` | store + embedding + whether a Groq key is configured (never the key) |
| GET | `/stats` | vector count, per-source/category/type counts, last ingest summary |

## POST /query
Request: `{"query": "...", "k": 5, "filters": {"category": "hr"}, "request_id": "optional-id"}`
(`filters` fields: `category`, `document_type`, `source_id`, `source_name`; a value or a list of values.)

Response fields: `request_id`, `status` (`answered | no_relevant_context | insufficient_evidence | generation_error`), `abstained`, `reason`, `answer`, `citations[]` (`ref, chunk_id, source_id, source_name, page, section, excerpt, score`), `invalid_citations[]`, `retrieved_chunks[]`, `retrieved_chunk_count`, `context_chunks_used`, `context_truncated`, `top_score`, `relevance_threshold`, `latency{retrieval_ms, generation_ms, end_to_end_ms}`, `token_usage{available, prompt_tokens, completion_tokens, total_tokens}`, `models{generator, embedding}`, `error`.

HTTP codes: 200 answered/abstained; 422 validation/invalid filter; 403 path not allowed; 404 ingest path missing; 429 Groq rate limit; 503 Groq not configured / auth etc. (generation unavailable); 504 Groq timeout; 502 other upstream failure. For generation failures the body still contains the retrieved chunks.

## Citation format
`citations[i].ref` is the short id the model saw (`C1`); `chunk_id` is the real stored id (`<doc_id>-<index>`); `page` is set for PDFs, `section` for HTML/Markdown; `excerpt` is the first ~220 characters of the retrieved chunk (taken from the chunk, never from model output).

## Token usage
Taken from the Groq response `usage`. If no LLM call was made (abstention) or the provider returned none, `available=false` and values are `null` - never estimated.
