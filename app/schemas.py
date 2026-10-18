"""Pydantic request/response models for the HTTP API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    path: str = Field(..., description="Folder or file to ingest, relative to the project root. "
                                       "Must be inside an allowed ingest root (default: data/).")
    prune: bool = Field(False, description="Delete stored chunks of sources under this prefix that no longer exist on disk.")
    source_prefix: str = Field("", max_length=64, description="Optional prefix added to every source_id.")


class FileResultModel(BaseModel):
    source_id: str
    status: str
    chunks: int = 0
    chunks_deleted: int = 0
    error: Optional[str] = None
    warning: Optional[str] = None


class IngestResponse(BaseModel):
    root: str
    files_discovered: int
    files_processed: int
    files_new: int
    files_updated: int
    files_unchanged: int
    files_failed: int
    files_skipped: int
    chunks_upserted: int
    chunks_deleted: int
    sources_pruned: list[str]
    collection_count_before: int
    collection_count_after: int
    duration_ms: float
    embedding_model: str
    chunking: dict[str, Any]
    results: list[FileResultModel]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = Field(None, ge=1, le=50)
    filters: Optional[dict[str, Any]] = Field(None, description="e.g. {\"category\": \"hr\"} or {\"category\": [\"hr\",\"finance\"]}")
    request_id: Optional[str] = Field(None, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class RetrievedChunkModel(BaseModel):
    rank: int
    chunk_id: str
    score: float
    distance: float
    source_id: str
    source_name: str
    title: str
    document_type: str
    category: str
    page: Optional[int]
    section: str
    text: Optional[str] = None


class CitationModel(BaseModel):
    ref: str
    chunk_id: str
    source_id: str
    source_name: str
    page: Optional[int]
    section: str
    excerpt: str
    score: float


class TokenUsage(BaseModel):
    available: bool
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class Latency(BaseModel):
    retrieval_ms: float
    generation_ms: Optional[float]
    end_to_end_ms: float


class QueryResponse(BaseModel):
    request_id: str
    status: str = Field(description="answered | no_relevant_context | insufficient_evidence | generation_error")
    abstained: bool
    reason: Optional[str]
    answer: str
    citations: list[CitationModel]
    invalid_citations: list[str]
    retrieved_chunks: list[RetrievedChunkModel]
    retrieved_chunk_count: int
    context_chunks_used: int
    context_truncated: bool
    top_score: Optional[float]
    relevance_threshold: float
    latency: Latency
    token_usage: TokenUsage
    models: dict[str, Any]
    error: Optional[dict[str, str]] = None


class RetrieveResponse(BaseModel):
    request_id: str
    retrieved_chunks: list[RetrievedChunkModel]
    retrieved_chunk_count: int
    relevant: bool
    reason: str
    top_score: Optional[float]
    latency: dict[str, float]
