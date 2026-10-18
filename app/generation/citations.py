"""Citation parsing and validation.

The model only ever sees short refs (C1..Cn). We map them back to real chunk ids and REJECT any ref
that was not in the retrieved context, reporting it transparently instead of presenting it as valid.
Excerpts come from the retrieved chunk text, never from model output, so quotes cannot be invented.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.retriever import RetrievedChunk

_REF = re.compile(r"\[?\bC(\d{1,3})\b\]?")
EXCERPT_CHARS = 220


@dataclass
class Citation:
    ref: str
    chunk_id: str
    source_id: str
    source_name: str
    page: int | None
    section: str
    excerpt: str
    score: float

    def to_dict(self) -> dict:
        return {"ref": self.ref, "chunk_id": self.chunk_id, "source_id": self.source_id,
                "source_name": self.source_name, "page": self.page, "section": self.section,
                "excerpt": self.excerpt, "score": round(self.score, 4)}


def extract_refs(answer_text: str, declared: list | None = None) -> list[str]:
    """Refs used in the answer text plus any declared in the JSON ``citations`` list (ordered, unique)."""
    found: list[str] = []
    for m in re.finditer(r"\[\s*C(\d{1,3})\s*\]", answer_text or ""):
        found.append(f"C{m.group(1)}")
    for d in declared or []:
        m = _REF.fullmatch(str(d).strip())
        if m:
            found.append(f"C{m.group(1)}")
        elif str(d).strip():
            found.append(str(d).strip())  # keep unparseable ref so it is reported as invalid
    seen, out = set(), []
    for r in found:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def resolve_citations(refs: list[str], ref_map: dict[str, RetrievedChunk]) -> tuple[list[Citation], list[str]]:
    valid, invalid = [], []
    for r in refs:
        ch = ref_map.get(r)
        if ch is None:
            invalid.append(r)
            continue
        ex = " ".join(ch.text.split())
        valid.append(Citation(ref=r, chunk_id=ch.chunk_id, source_id=ch.source_id, source_name=ch.source_name,
                              page=ch.page, section=ch.section,
                              excerpt=ex[:EXCERPT_CHARS] + ("…" if len(ex) > EXCERPT_CHARS else ""),
                              score=ch.score))
    return valid, invalid
