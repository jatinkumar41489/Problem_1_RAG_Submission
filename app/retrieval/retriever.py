"""Top-k semantic retrieval with metadata filters. Independent of any LLM."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.embeddings.encoder import Encoder
from app.exceptions import RetrievalError
from app.retrieval.relevance import RelevanceDecision, assess_relevance
from app.vectorstore.chroma_store import ChromaStore
from app.vectorstore.filters import validate_filters


@dataclass
class RetrievedChunk:
    rank: int
    chunk_id: str
    text: str
    score: float
    distance: float
    source_id: str
    source_name: str
    title: str
    document_type: str
    category: str
    page: int | None
    section: str

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in ("rank", "chunk_id", "score", "distance", "source_id",
                                           "source_name", "title", "document_type", "category",
                                           "page", "section")}
        d["score"], d["distance"] = round(self.score, 4), round(self.distance, 4)
        if include_text:
            d["text"] = self.text
        return d


@dataclass
class RetrievalResult:
    query: str
    k: int
    filters: dict[str, list[str]]
    chunks: list[RetrievedChunk]
    decision: RelevanceDecision
    embed_ms: float
    search_ms: float
    total_ms: float
    store_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Retriever:
    def __init__(self, settings: Settings, encoder: Encoder, store: ChromaStore) -> None:
        self.settings, self.encoder, self.store = settings, encoder, store

    def retrieve(self, query: str, k: int | None = None, filters: dict | None = None) -> RetrievalResult:
        k = self.settings.top_k if k is None else k
        if not isinstance(k, int) or isinstance(k, bool) or not (1 <= k <= self.settings.max_top_k):
            raise RetrievalError(f"k must be an integer in [1, {self.settings.max_top_k}], got {k!r}")
        if not query or not query.strip():
            raise RetrievalError("query must be a non-empty string")
        norm_filters = validate_filters(filters)  # fail fast on bad filters
        t0 = time.perf_counter()
        qvec = self.encoder.encode_one(query.strip())
        t1 = time.perf_counter()
        count = self.store.count()
        hits = self.store.query(qvec, k, norm_filters or None) if count else []
        t2 = time.perf_counter()
        chunks = []
        for r, h in enumerate(hits, start=1):
            m = h.metadata
            page = m.get("page", -1)
            chunks.append(RetrievedChunk(
                rank=r, chunk_id=h.chunk_id, text=h.text, score=h.score, distance=h.distance,
                source_id=m.get("source_id", ""), source_name=m.get("source_name", ""),
                title=m.get("title", ""), document_type=m.get("document_type", ""),
                category=m.get("category", ""), page=None if page in (-1, None) else int(page),
                section=m.get("section", "")))
        decision = assess_relevance([c.score for c in chunks], self.settings.min_relevance_score,
                                    store_empty=(count == 0))
        return RetrievalResult(query=query, k=k, filters=norm_filters, chunks=chunks, decision=decision,
                               embed_ms=(t1 - t0) * 1000, search_ms=(t2 - t1) * 1000,
                               total_ms=(t2 - t0) * 1000, store_count=count)
