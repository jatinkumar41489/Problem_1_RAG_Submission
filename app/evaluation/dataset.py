"""Evaluation dataset loading + chunk-level relevance grading.

Relevance labels are *evidence phrases* (source, optional page, short verbatim phrase, grade), not
chunk ids, so the same labels stay valid under any chunk size/overlap. A chunk is relevant when it
comes from the labelled source (and page, if given) and its whitespace-normalised, lower-cased text
contains the phrase. Grades: 2 = primary answer evidence, 1 = supporting evidence; 0 = irrelevant.

Limitation: labels were written by the project author together with the corpus. They are a
consistent measuring stick, NOT independently validated ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.exceptions import ConfigError
from app.ingestion.hashing import normalize_text


@dataclass
class Question:
    id: str
    question: str
    type: str
    filters: dict | None
    answerable: bool
    gold_answers: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)


def load_dataset(directory: Path) -> list[Question]:
    try:
        qs = json.loads((directory / "questions.json").read_text(encoding="utf-8"))
        gold = json.loads((directory / "gold_answers.json").read_text(encoding="utf-8"))
        rel = json.loads((directory / "relevance_labels.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read evaluation dataset in {directory}: {exc}") from exc
    out = []
    for q in qs:
        out.append(Question(id=q["id"], question=q["question"], type=q["type"], filters=q.get("filters"),
                            answerable=bool(q["answerable"]),
                            gold_answers=gold.get(q["id"], {}).get("answers", []),
                            evidence=rel.get(q["id"], [])))
    if not (15 <= len(out) <= 30):
        raise ConfigError(f"Evaluation set must contain 15-30 questions, found {len(out)}")
    return out


def _matches(chunk: dict[str, Any], ev: dict) -> bool:
    md = chunk["metadata"]
    if md.get("source_id") != ev["source_id"]:
        return False
    if ev.get("page") is not None and md.get("page") != ev["page"]:
        return False
    return normalize_text(ev["evidence"]).lower() in normalize_text(chunk["text"]).lower()


def chunk_grade(chunk: dict[str, Any], evidence: list[dict]) -> int:
    """Max grade over all evidence items matched by this chunk (0 if none)."""
    return max((int(ev.get("grade", 1)) for ev in evidence if _matches(chunk, ev)), default=0)


def evidence_found(chunks: list[dict[str, Any]], evidence: list[dict]) -> list[bool]:
    """For each evidence item: does any of the given chunks contain it?"""
    return [any(_matches(c, ev) for c in chunks) for ev in evidence]
