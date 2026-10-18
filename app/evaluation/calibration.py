"""Calibrate the abstention threshold on the labelled question set.

Positive class = answerable question. Prediction = "answerable" iff top-1 cosine similarity >= t.
We choose t maximising balanced accuracy (ties -> the midpoint of the tied range) and report a
leave-one-out estimate, because choosing t and scoring it on the same 29 questions is optimistic.
"""
from __future__ import annotations

import numpy as np


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    uniq = np.unique(scores)
    cands = np.concatenate(([uniq[0] - 1e-6], (uniq[:-1] + uniq[1:]) / 2, [uniq[-1] + 1e-6]))
    best, ties = -1.0, []
    for t in cands:
        pred = scores >= t
        tpr = (pred & labels).sum() / max(labels.sum(), 1)
        tnr = (~pred & ~labels).sum() / max((~labels).sum(), 1)
        ba = (tpr + tnr) / 2
        if ba > best + 1e-12:
            best, ties = ba, [t]
        elif abs(ba - best) <= 1e-12:
            ties.append(t)
    return float((min(ties) + max(ties)) / 2), float(best)


def calibrate(scores: list[float], answerable: list[bool]) -> dict:
    s, y = np.asarray(scores, dtype=float), np.asarray(answerable, dtype=bool)
    if y.all() or (~y).all():
        return {"status": "not_calibrated", "reason": "need both answerable and unanswerable questions"}
    t, ba = _best_threshold(s, y)
    pred = s >= t
    loo = []
    for i in range(len(s)):
        m = np.ones(len(s), dtype=bool)
        m[i] = False
        ti, _ = _best_threshold(s[m], y[m])
        loo.append((s[i] >= ti) == y[i])
    return {"status": "ok", "recommended_min_relevance_score": round(t, 4), "balanced_accuracy_in_sample": round(ba, 4),
            "loo_accuracy": round(float(np.mean(loo)), 4), "n": int(len(s)),
            "answerable": {"n": int(y.sum()), "min": round(float(s[y].min()), 4), "median": round(float(np.median(s[y])), 4)},
            "unanswerable": {"n": int((~y).sum()), "max": round(float(s[~y].max()), 4), "median": round(float(np.median(s[~y])), 4)},
            "false_abstain": int((~pred & y).sum()), "false_answer": int((pred & ~y).sum()),
            "caveat": "Threshold is specific to the embedding model; scores of different models are not comparable. "
                      "Calibrated on the same small set it is evaluated on - use the LOO figure as the honest estimate."}
