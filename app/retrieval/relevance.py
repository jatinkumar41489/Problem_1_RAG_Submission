"""Abstention (retrieval-side relevance gate).

Score semantics: ``score`` is cosine similarity (higher = closer), NOT a calibrated probability.
A high similarity does not prove a chunk answers the question, and the threshold is specific to the
embedding model: re-calibrate with ``scripts/calibrate_threshold.py`` after changing the model.
The gate is only the first line of defence; the LLM prompt performs a second, evidence-based check.
"""
from __future__ import annotations

from dataclasses import dataclass

REASON_OK = "ok"
REASON_EMPTY_STORE = "empty_store"
REASON_NO_MATCH = "no_matching_chunks"   # filter excluded everything
REASON_LOW = "low_relevance"             # best score below threshold


@dataclass(frozen=True)
class RelevanceDecision:
    relevant: bool
    reason: str
    top_score: float | None
    threshold: float


def assess_relevance(scores: list[float], min_score: float, store_empty: bool = False) -> RelevanceDecision:
    if store_empty:
        return RelevanceDecision(False, REASON_EMPTY_STORE, None, min_score)
    if not scores:
        return RelevanceDecision(False, REASON_NO_MATCH, None, min_score)
    top = max(scores)
    if top < min_score:
        return RelevanceDecision(False, REASON_LOW, top, min_score)
    return RelevanceDecision(True, REASON_OK, top, min_score)
