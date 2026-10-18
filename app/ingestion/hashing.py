"""Deterministic hashing used for idempotent ingestion."""
from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse all whitespace so cosmetic edits (line endings, indentation) don't change hashes."""
    return _WS.sub(" ", text).strip()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def content_hash(segments: list[tuple[int | None, str]]) -> str:
    """Hash of normalized extracted text, including page structure.

    Hashing *extracted text* (not raw bytes) means a PDF re-saved with a new timestamp but identical
    text is correctly seen as unchanged.
    """
    h = hashlib.sha256()
    for page, text in segments:
        h.update(f"[page={page}]".encode())
        h.update(normalize_text(text).encode("utf-8"))
    return h.hexdigest()


def document_id(source_id: str, c_hash: str) -> str:
    """Stable doc id derived from source identity AND content (12 hex chars)."""
    return sha256_hex(f"{source_id}\n{c_hash}")[:12]


def chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}-{index:04d}"


def text_hash(text: str) -> str:
    return sha256_hex(normalize_text(text))[:16]
