"""Exact Match / token-F1 (SQuAD-style normalisation) + a lenient containment check.

Normalisation: lower-case, remove punctuation EXCEPT characters that carry meaning in numbers
($ % . ,) inside numerals, drop English articles, collapse whitespace.
EM and F1 measure *string overlap with a short gold answer*; they are NOT a measure of semantic
correctness. A verbose, correct sentence scores low EM/F1 against a terse gold answer, which is why
``contains_gold`` is reported alongside as a lenient, clearly-labelled extra.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\[c\d+\]", " ", s)                       # strip our own citation markers
    s = re.sub(r"(?<!\d)[,.](?!\d)", " ", s)              # punctuation, but keep 2,000 and 0.60
    s = re.sub(r"[^\w\s$%.,]", " ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, golds: Sequence[str]) -> float:
    """Best F1 over all acceptable gold answers."""
    return max((_f1(pred, g) for g in golds), default=0.0)


def contains_gold(pred: str, golds: Sequence[str]) -> float:
    p = f" {normalize_answer(pred)} "
    return float(any(f" {normalize_answer(g)} " in p for g in golds))
