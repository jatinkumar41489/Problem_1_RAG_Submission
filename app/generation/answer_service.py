"""Retrieval -> grounded generation orchestration with explicit statuses.

Statuses (kept distinct on purpose):
  answered                 model answered with >=1 valid citation
  no_relevant_context      retrieval gate abstained (empty store / filter excludes all / low relevance);
                           the LLM is NOT called
  insufficient_evidence    LLM was called and said the context cannot answer (or its answer had no
                           valid citation and was withheld)
  generation_error         retrieval succeeded but the LLM call failed / returned unusable output
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.exceptions import GenerationError
from app.generation.citations import Citation, extract_refs, resolve_citations
from app.generation.groq_client import GroqClient, LLMResponse
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.logging_config import get_logger, log_event
from app.retrieval.retriever import RetrievalResult, RetrievedChunk, Retriever

log = get_logger("query")

STATUS_ANSWERED = "answered"
STATUS_NO_CONTEXT = "no_relevant_context"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_ERROR = "generation_error"

NO_CONTEXT_MESSAGE = "I could not find relevant information in the indexed documents to answer this question."
INSUFFICIENT_MESSAGE = "The retrieved documents do not contain enough information to answer this question."
TOKENS_PER_WORD = 1.35  # explicit estimate used only for context budgeting, never reported as usage


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD) + 1


@dataclass
class AnswerResult:
    request_id: str
    query: str
    status: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    context_chunks_used: int = 0
    context_truncated: bool = False
    reason: str | None = None
    retrieval_ms: float = 0.0
    generation_ms: float | None = None
    total_ms: float = 0.0
    llm: LLMResponse | None = None
    error_category: str | None = None
    error_message: str | None = None
    model: str | None = None

    @property
    def token_usage(self) -> dict[str, Any]:
        if self.llm is None or self.llm.total_tokens is None and self.llm.prompt_tokens is None:
            return {"available": False, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        return {"available": True, "prompt_tokens": self.llm.prompt_tokens,
                "completion_tokens": self.llm.completion_tokens, "total_tokens": self.llm.total_tokens}


def parse_model_json(text: str) -> dict | None:
    """Robustly parse the model's JSON verdict; tolerate code fences and leading/trailing prose."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    for candidate in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t and "}" in t else ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class AnswerService:
    def __init__(self, settings: Settings, retriever: Retriever, llm: GroqClient) -> None:
        self.settings, self.retriever, self.llm = settings, retriever, llm

    # ------------------------------------------------------------------ context budgeting
    def _build_context(self, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], bool]:
        budget = self.settings.max_context_tokens
        used, kept = 0, []
        for ch in chunks:  # chunks are already ranked best-first; drop the tail that doesn't fit
            cost = estimate_tokens(ch.text) + 20
            if used + cost > budget and kept:
                return kept, True
            kept.append(ch)
            used += cost
        return kept, False

    @staticmethod
    def _header(ch: RetrievedChunk) -> str:
        h = ch.source_name
        if ch.page:
            h += f", page {ch.page}"
        if ch.section:
            h += f", section: {ch.section}"
        return h

    # ------------------------------------------------------------------ main
    def answer(self, query: str, k: int | None = None, filters: dict | None = None,
               request_id: str | None = None) -> AnswerResult:
        rid = request_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        ret = self.retriever.retrieve(query, k=k, filters=filters)
        res = AnswerResult(request_id=rid, query=query, status=STATUS_NO_CONTEXT, answer=NO_CONTEXT_MESSAGE,
                           retrieval=ret, retrieval_ms=ret.total_ms, model=self.settings.groq_model)

        if not ret.decision.relevant:
            res.reason = ret.decision.reason
            return self._finish(res, t0)

        used, truncated = self._build_context(ret.chunks)
        res.context_chunks_used, res.context_truncated = len(used), truncated
        ref_map = {f"C{i}": ch for i, ch in enumerate(used, start=1)}
        prompt = build_user_prompt(query, [(r, self._header(ch), ch.text) for r, ch in ref_map.items()])
        t_gen = time.perf_counter()
        try:
            llm = self.llm.complete([{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": prompt}])
        except GenerationError as exc:
            res.generation_ms = (time.perf_counter() - t_gen) * 1000
            res.status, res.answer = STATUS_ERROR, ""
            res.error_category, res.error_message, res.reason = exc.category, exc.message, "llm_call_failed"
            return self._finish(res, t0)
        res.generation_ms, res.llm, res.model = llm.latency_ms, llm, llm.model

        parsed = parse_model_json(llm.text)
        if parsed is None or "answerable" not in parsed:
            res.status, res.answer, res.reason = STATUS_ERROR, "", "malformed_model_output"
            res.error_category, res.error_message = "malformed_response", "Model did not return the required JSON."
            return self._finish(res, t0)

        answer_text = str(parsed.get("answer", "")).strip()
        refs = extract_refs(answer_text, parsed.get("citations") if isinstance(parsed.get("citations"), list) else [])
        valid, invalid = resolve_citations(refs, ref_map)
        res.invalid_citations = invalid
        if not bool(parsed.get("answerable")):
            res.status, res.answer, res.reason = STATUS_INSUFFICIENT, answer_text or INSUFFICIENT_MESSAGE, "model_reported_insufficient"
        elif not valid:
            res.status, res.answer, res.reason = STATUS_INSUFFICIENT, INSUFFICIENT_MESSAGE, "no_valid_citations"
        else:
            res.status, res.answer, res.citations = STATUS_ANSWERED, answer_text, valid
        return self._finish(res, t0)

    def _finish(self, res: AnswerResult, t0: float) -> AnswerResult:
        res.total_ms = (time.perf_counter() - t0) * 1000
        usage = res.token_usage
        fields: dict[str, Any] = dict(
            request_id=res.request_id, status=res.status, reason=res.reason,
            retrieval_ms=round(res.retrieval_ms, 2),
            generation_ms=None if res.generation_ms is None else round(res.generation_ms, 2),
            total_ms=round(res.total_ms, 2),
            retrieved_chunks=len(res.retrieval.chunks) if res.retrieval else 0,
            context_chunks_used=res.context_chunks_used, citations=len(res.citations),
            invalid_citations=len(res.invalid_citations), model=res.model,
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"], token_usage_available=usage["available"],
            error_category=res.error_category, success=res.status != STATUS_ERROR,
            query_sha=hashlib.sha256(res.query.encode()).hexdigest()[:10], query_chars=len(res.query))
        if self.settings.log_query_text:
            fields["query"] = res.query[:200]
        log_event(log, "query", **fields)
        return res
