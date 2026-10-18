# Architecture

## Flow (matches template section 1.1)

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion (idempotent)"]
        A[PDF / HTML / MD files] --> B[Loaders: pypdf, BeautifulSoup, Markdown<br/>segments = PDF pages / doc body + heading offsets]
        B --> C[Chunker: word windows<br/>size=180, overlap=30 words]
        C --> D[Stable IDs: doc_id = sha256 source+content<br/>chunk_id = doc_id-index]
        D --> E{Unchanged?<br/>same ids + text hashes}
        E -- yes --> Z[skip: no embedding, no write]
        E -- no --> F[Embedder: all-MiniLM-L6-v2, 384-d, L2-normalised]
        F --> G[(ChromaDB persistent, cosine<br/>vector + text + metadata)]
        G --> H[delete stale chunk ids of that source only]
    end
    subgraph Query["Query path"]
        Q[POST /query: question, k, metadata filter] --> R[Embed query - same model]
        R --> S[Filtered top-k search in ChromaDB]
        G -.-> S
        S --> T{Gate: store empty / filter excludes all /<br/>top score below MIN_RELEVANCE_SCORE?}
        T -- yes --> N1[NO-CONTEXT BRANCH: status=no_relevant_context<br/>LLM is NOT called]
        T -- no --> U[Trim to MAX_CONTEXT_TOKENS, build prompt<br/>chunks = untrusted data, refs C1..Cn]
        U --> V[Groq LLM: JSON answerable/answer/citations]
        V --> W{answerable AND >=1 valid citation?}
        W -- no --> N2[status=insufficient_evidence]
        W -- yes --> X[Resolve refs to retrieved chunk ids<br/>invalid refs reported, not shown as valid]
        V -. API failure .-> N3[status=generation_error<br/>retrieved chunks still returned]
        X --> Y[Response: answer + citations + latency + chunk count + token usage]
    end
```

## Modules

| Package | Responsibility |
|---|---|
| `app/config.py` | env-based settings, validation, secret-safe repr |
| `app/ingestion/` | `loaders` (PDF/HTML/MD), `chunking`, `hashing` (stable ids), `pipeline` (idempotent upsert/prune) |
| `app/embeddings/` | `Encoder` ABC; MiniLM backend; offline `hash` fallback |
| `app/vectorstore/` | `ChromaStore` wrapper, `filters` (validated metadata filters) |
| `app/retrieval/` | `Retriever` (timed, LLM-free), `relevance` (abstention gate) |
| `app/generation/` | `prompts`, `groq_client` (bounded retries, categorised errors), `citations`, `answer_service` (statuses) |
| `app/api/`, `app/main.py` | FastAPI routes, error handlers, app factory |
| `app/evaluation/` | dataset, metrics, judge, calibration, latency, cost model, harness |

## Statuses (never conflated)
`answered` - `no_relevant_context` (retrieval gate, LLM not called) - `insufficient_evidence` (LLM called, evidence lacking or no valid citation) - `generation_error` (retrieval ok, LLM failed).

## Persistence and concurrency
ChromaDB `PersistentClient` (SQLite + HNSW files) in `CHROMA_PERSIST_DIRECTORY`. Run **one** server process per directory; multiple writers are not supported. Back up by stopping the service and copying the directory. The collection records the embedding model/dimension and the app refuses to open it with a different model (it never silently resets).
