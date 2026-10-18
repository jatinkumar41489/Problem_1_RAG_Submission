"""Retrieval metrics (pure functions; hand-verified in tests/unit/test_retrieval_metrics.py).

Inputs are per-query ``ranked_grades``: the relevance grade (0 = irrelevant, >=1 relevant) of each
retrieved chunk in rank order (rank 1 first). Definitions, with k = cut-off:

* Recall@k (chunk-level)  = #relevant chunks in top-k / #relevant chunks in the whole corpus.
                            None when the query has no relevant chunk in the corpus (query excluded).
* Evidence recall@k       = #labelled evidence items covered by >=1 top-k chunk / #evidence items.
                            Not penalised when overlap duplicates an evidence phrase into 2 chunks.
* Hit rate@k              = 1 if any relevant chunk is in top-k else 0 (mean over queries = Hit Rate).
* Reciprocal rank         = 1 / rank of first relevant chunk within top-k, else 0 (mean = MRR@k).
* nDCG@k                  = DCG@k / IDCG@k, linear gain (gain = grade), discount 1/log2(rank+1);
                            IDCG uses the corpus's relevant grades sorted descending, cut at k.
* Context precision@k     = #relevant chunks among the retrieved top-k / #retrieved chunks.
                            Empty retrieval -> 0.0 (nothing useful was retrieved).
"""
from __future__ import annotations

import math
from typing import Sequence


def _topk(ranked_grades: Sequence[int], k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(ranked_grades)[:k]


def recall_at_k(ranked_grades: Sequence[int], total_relevant: int, k: int) -> float | None:
    if total_relevant <= 0:
        return None
    return sum(1 for g in _topk(ranked_grades, k) if g >= 1) / total_relevant


def evidence_recall(found_flags: Sequence[bool]) -> float | None:
    return None if not found_flags else sum(found_flags) / len(found_flags)


def hit_at_k(ranked_grades: Sequence[int], k: int) -> float:
    return 1.0 if any(g >= 1 for g in _topk(ranked_grades, k)) else 0.0


def reciprocal_rank(ranked_grades: Sequence[int], k: int) -> float:
    for i, g in enumerate(_topk(ranked_grades, k), start=1):
        if g >= 1:
            return 1.0 / i
    return 0.0


def dcg(grades: Sequence[int]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg_at_k(ranked_grades: Sequence[int], corpus_relevant_grades: Sequence[int], k: int) -> float | None:
    ideal = sorted((g for g in corpus_relevant_grades if g >= 1), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return None
    return dcg(_topk(ranked_grades, k)) / idcg


def context_precision_at_k(ranked_grades: Sequence[int], k: int) -> float:
    top = _topk(ranked_grades, k)
    if not top:
        return 0.0
    return sum(1 for g in top if g >= 1) / len(top)


def mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
