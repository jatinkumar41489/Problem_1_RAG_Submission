"""HTTP routes."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import BASE_DIR, resolve_path
from app.exceptions import PathNotAllowedError, SourceNotFoundError
from app.generation.answer_service import STATUS_ERROR, AnswerResult
from app.schemas import (IngestRequest, IngestResponse, QueryRequest, QueryResponse, RetrieveResponse)
from app.services import Services

router = APIRouter()


def svc(request: Request) -> Services:
    return request.app.state.services


def _safe_path(raw: str, services: Services) -> Path:
    """Resolve ``raw`` against the project root and require it to live inside an allowed ingest root.
    ``resolve()`` collapses ``..`` and follows symlinks, so traversal cannot escape the roots."""
    p = resolve_path(raw)
    for root in services.settings.ingest_roots:
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    allowed = [str(r.relative_to(BASE_DIR)) if r.is_relative_to(BASE_DIR) else r.name
               for r in services.settings.ingest_roots]
    raise PathNotAllowedError(f"Path is outside the allowed ingest roots {allowed}.")


def _chunk_models(result, include_text: bool = True):
    return [c.to_dict(include_text) for c in result.chunks]


def _to_query_response(r: AnswerResult) -> dict:
    ret = r.retrieval
    err = {"category": r.error_category, "message": r.error_message} if r.status == STATUS_ERROR else None
    return QueryResponse(
        request_id=r.request_id, status=r.status, abstained=r.status in ("no_relevant_context", "insufficient_evidence"),
        reason=r.reason, answer=r.answer, citations=[c.to_dict() for c in r.citations],
        invalid_citations=r.invalid_citations, retrieved_chunks=_chunk_models(ret),
        retrieved_chunk_count=len(ret.chunks), context_chunks_used=r.context_chunks_used,
        context_truncated=r.context_truncated, top_score=None if ret.decision.top_score is None else round(ret.decision.top_score, 4),
        relevance_threshold=ret.decision.threshold,
        latency={"retrieval_ms": round(r.retrieval_ms, 2),
                 "generation_ms": None if r.generation_ms is None else round(r.generation_ms, 2),
                 "end_to_end_ms": round(r.total_ms, 2)},
        token_usage=r.token_usage, models={"generator": r.model, "embedding": svc_embedding_name(ret)},
        error=err).model_dump()


def svc_embedding_name(ret) -> str:
    return ret.extra.get("embedding_model", "")


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
def ingest(body: IngestRequest, request: Request):
    s = svc(request)
    path = _safe_path(body.path, s)
    if not path.exists():
        raise SourceNotFoundError(f"Path does not exist: {body.path}")
    rep = s.pipeline.ingest_path(path, prune=body.prune, source_prefix=body.source_prefix)
    d = rep.to_dict()
    d["collection_count_before"], d["collection_count_after"] = d.pop("count_before"), d.pop("count_after")
    d["root"] = str(Path(d["root"]).relative_to(BASE_DIR)) if Path(d["root"]).is_relative_to(BASE_DIR) else Path(d["root"]).name
    s.last_ingest = {k: v for k, v in d.items() if k != "results"}
    return d


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(body: QueryRequest, request: Request):
    s = svc(request)
    result = s.answers.answer(body.query, k=body.k, filters=body.filters, request_id=body.request_id)
    result.retrieval.extra["embedding_model"] = s.encoder.name
    payload = _to_query_response(result)
    code = 200
    if result.status == STATUS_ERROR:
        code = {"rate_limit": 429, "timeout": 504, "configuration": 503}.get(result.error_category or "", 502)
    return JSONResponse(status_code=code, content=payload)


@router.post("/retrieve", response_model=RetrieveResponse, tags=["query"])
def retrieve(body: QueryRequest, request: Request):
    """Retrieval only (no LLM call) - useful for debugging and for measuring retrieval in isolation."""
    s = svc(request)
    r = s.retriever.retrieve(body.query, k=body.k, filters=body.filters)
    return RetrieveResponse(
        request_id=body.request_id or uuid.uuid4().hex[:12], retrieved_chunks=_chunk_models(r),
        retrieved_chunk_count=len(r.chunks), relevant=r.decision.relevant, reason=r.decision.reason,
        top_score=None if r.decision.top_score is None else round(r.decision.top_score, 4),
        latency={"embed_ms": round(r.embed_ms, 2), "search_ms": round(r.search_ms, 2), "retrieval_ms": round(r.total_ms, 2)})


@router.get("/health", tags=["ops"])
def health(request: Request):
    s = svc(request)
    body = {"status": "ok", "embedding": {"backend": s.encoder.backend, "model": s.encoder.name,
                                          "dim": s.encoder.dim, "loaded": s.encoder.loaded},
            "generation": {"provider": "groq", "model": s.settings.groq_model,
                           "api_key_configured": bool(s.settings.groq_api_key)}}
    try:
        body["vector_store"] = s.store.health()
    except Exception:  # never leak internals
        body["status"] = "degraded"
        body["vector_store"] = {"ok": False}
        return JSONResponse(status_code=503, content=body)
    return body


@router.get("/stats", tags=["ops"])
def stats(request: Request):
    s = svc(request)
    sources = s.store.source_summary()
    by_cat: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for src in sources:
        by_cat[src["category"]] = by_cat.get(src["category"], 0) + src["chunks"]
        by_type[src["document_type"]] = by_type.get(src["document_type"], 0) + src["chunks"]
    return {"collection": s.store.collection_name, "vector_count": s.store.count(), "source_count": len(sources),
            "chunks_by_category": by_cat, "chunks_by_document_type": by_type,
            "chunking_defaults": s.pipeline.chunker.config, "embedding_model": s.encoder.name,
            "embedding_dim": s.encoder.dim, "last_ingest": s.last_ingest, "sources": sources}
