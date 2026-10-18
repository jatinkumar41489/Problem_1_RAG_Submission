"""Token-window chunking.

"Token" here means a whitespace-delimited word (regex ``\\S+``). This is a deliberate, documented
approximation: it is deterministic and tokenizer-free. For all-MiniLM-L6-v2 (max 256 word-pieces)
one word is roughly 1.3 word-pieces, which is why the default chunk size is 180 words, not 800.

Chunks never cross a segment boundary (a PDF page), so page numbers in citations are exact.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from app.exceptions import ConfigError
from app.ingestion.loaders import LoadedDocument

_TOKEN = re.compile(r"\S+")
CHUNKING_METHOD = "whitespace-word-window"


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    page: int | None
    section: str
    token_count: int


def validate_chunk_params(size: int, overlap: int) -> None:
    if size <= 0:
        raise ConfigError("chunk size must be positive")
    if overlap < 0:
        raise ConfigError("chunk overlap must be non-negative")
    if overlap >= size:
        raise ConfigError("chunk overlap must be smaller than chunk size")


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        validate_chunk_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def config(self) -> dict:
        return {"method": CHUNKING_METHOD, "unit": "whitespace word",
                "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}

    def chunk_text(self, text: str, headings: list[tuple[int, str]] | None = None,
                   page: int | None = None, start_index: int = 0) -> list[Chunk]:
        """Slide a window of ``chunk_size`` words with stride ``size - overlap``.

        The loop ends as soon as a window reaches the last word, so the final window always
        contains at least one new word (no fragment that is fully contained in its predecessor)
        and the loop always terminates because the stride is >= 1.
        """
        toks = list(_TOKEN.finditer(text))
        n = len(toks)
        if n == 0:
            return []
        offsets = [o for o, _ in (headings or [])]
        labels = [h for _, h in (headings or [])]
        stride = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        i = 0
        idx = start_index
        while i < n:
            j = min(i + self.chunk_size, n)
            start_char = toks[i].start()
            body = text[start_char:toks[j - 1].end()].strip()
            if body:
                pos = bisect.bisect_right(offsets, start_char) - 1
                section = labels[pos] if pos >= 0 else ""
                chunks.append(Chunk(index=idx, text=body, page=page, section=section, token_count=j - i))
                idx += 1
            if j >= n:
                break
            i += stride
        return chunks

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        out: list[Chunk] = []
        for seg in doc.segments:
            out.extend(self.chunk_text(seg.text, seg.headings, seg.page, start_index=len(out)))
        return out
