# Design decisions and trade-offs

**Vector store: ChromaDB (persistent).** Embedded, zero-ops, metadata filtering built in, simple upsert/delete by id/where - fits an idempotent-ingest requirement with little code. Rejected for this build: FAISS (no metadata filtering/persistence layer of its own), sqlite-vec (fine, but fewer ready filtering/ANN conveniences), LanceDB/Qdrant-embedded (equally valid; not benchmarked), pgvector (needs a Postgres service; the better choice if you already run Postgres or exceed ~1-5M vectors). **A second store was not benchmarked** (optional bonus not done).

**Chunking: 180 words / 30 overlap, not 800/120.** all-MiniLM-L6-v2 truncates input at 256 word-pieces (~190 words); an 800-token chunk would be silently cut and its tail never searchable. Unit is a whitespace word (documented approximation). Chunks never cross PDF pages (exact page citations) and carry the heading active at their start. Larger chunks -> fewer vectors/cost and more context per hit but lower precision and more prompt tokens; smaller chunks -> sharper hits but split facts and more vectors. The measured 80/180/400 comparison (offline lexical embedder) shows no decisive winner on 23 queries; **the effect of truncation with the real model was not measured.**

**Embeddings: all-MiniLM-L6-v2, 384-d.** Local (no per-token API cost), 1.5 KB/vector float32 (vs 3-6 KB for 768-1536-d models), CPU-friendly; quality below larger models. Dimension is verified programmatically (`python -m app.cli verify`). The `hash` backend is an offline lexical fallback only, never a quality claim.

**No-relevant-context.** Two layers: (1) retrieval gate - empty store, filter excludes everything, or top cosine below `MIN_RELEVANCE_SCORE` -> LLM not called; (2) prompt rule + required JSON `answerable` flag + citation validation -> `insufficient_evidence`. Similarity is not proof of relevance, so layer 2 exists. Offline calibration (lexical embedder): leave-one-out accuracy 0.79 - the gate alone is imperfect (one unanswerable question reached the LLM; three answerable ones were blocked).

**Idempotent ingestion.** Deterministic ids (`sha256(source_id+content)` -> chunk ids), per-chunk text hashes, unchanged-file short-circuit, upsert-then-delete-stale scoped to one source. Same content at two paths is stored twice on purpose (per-source citations/filters) with a warning.

**Prompt-injection posture.** Chunks are delimited, declared untrusted, tag-neutralised; the model can only cite refs that exist; invalid refs are rejected. This reduces, not eliminates, injection risk.

**Accepted trade-offs / when to go back to managed.** See `docs/COST_ASSUMPTIONS.md`. Switch to managed (or pgvector/Qdrant server) when: no one can own ops/backups, you need HA/multi-writer, the index exceeds one node's RAM (~10M x 384-d), or measured traffic is below the break-even where serverless is cheaper.
