"""Idempotent ingestion pipeline.

Identity model
--------------
* ``source_id``   = POSIX path of the file relative to the ingest root (optionally prefixed). Two files
                    with the same basename in different directories therefore never collide.
* ``content_hash``= sha256 of normalised extracted text (+ page structure).
* ``doc_id``      = sha256(source_id + content_hash)[:12]  -> changes when content changes.
* ``chunk_id``    = ``<doc_id>-<index:04d>``               -> deterministic; same input => same ids.

Re-ingest algorithm per file
----------------------------
1. load + chunk (cheap, no embeddings); compute the expected chunk ids and per-chunk text hashes.
2. fetch the chunk ids currently stored for ``source_id``.
3. if stored ids == expected ids and all text hashes match  -> UNCHANGED (no embedding, no write).
4. else embed, ``upsert`` the new chunks, THEN delete stale ids of that source (upsert-before-delete
   so a crash never leaves a source with zero chunks). Unrelated sources are never touched.
5. optional ``prune``: delete chunks of sources under the same prefix that no longer exist on disk.

The same content under two different source paths is stored twice on purpose (citations/filters are
per-source); a ``duplicate_content_of`` warning is reported so the caller can de-duplicate on disk.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.embeddings.encoder import Encoder
from app.exceptions import DocumentLoadError, IngestionError, UnsupportedFileType
from app.ingestion import hashing
from app.ingestion.chunking import CHUNKING_METHOD, Chunker
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from app.logging_config import get_logger, log_event
from app.vectorstore.chroma_store import ChromaStore

log = get_logger("ingestion")


@dataclass
class FileResult:
    source_id: str
    status: str  # new | updated | unchanged | failed | skipped
    chunks: int = 0
    chunks_deleted: int = 0
    error: str | None = None
    warning: str | None = None


@dataclass
class IngestReport:
    root: str
    files_discovered: int = 0
    files_processed: int = 0       # new + updated
    files_new: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_failed: int = 0
    files_skipped: int = 0         # unsupported extension
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    sources_pruned: list[str] = field(default_factory=list)
    count_before: int = 0
    count_after: int = 0
    duration_ms: float = 0.0
    embedding_model: str = ""
    chunking: dict = field(default_factory=dict)
    results: list[FileResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IngestionPipeline:
    def __init__(self, settings: Settings, encoder: Encoder, store: ChromaStore) -> None:
        self.settings, self.encoder, self.store = settings, encoder, store
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap)

    # ------------------------------------------------------------------ discovery
    @staticmethod
    def discover(root: Path) -> tuple[list[Path], list[Path]]:
        """Return (supported_files, skipped_files) under root, sorted, symlinks not followed."""
        supported, skipped = [], []
        if root.is_file():
            files = [root]
        else:
            files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
        for p in files:
            if any(part.startswith(".") for part in p.relative_to(root if root.is_dir() else root.parent).parts):
                continue  # hidden files/dirs
            (supported if p.suffix.lower() in SUPPORTED_EXTENSIONS else skipped).append(p)
        return supported, skipped

    # ------------------------------------------------------------------ main entry
    def ingest_path(self, path: Path, prune: bool = False, source_prefix: str = "") -> IngestReport:
        t0 = time.perf_counter()
        path = Path(path).resolve()
        if not path.exists():
            raise IngestionError(f"Path does not exist: {path}")
        root = path if path.is_dir() else path.parent
        report = IngestReport(root=str(path), count_before=self.store.count(),
                              embedding_model=self.encoder.name,
                              chunking=self.chunker.config)
        files, skipped = self.discover(path)
        report.files_discovered = len(files)
        report.files_skipped = len(skipped)
        for p in skipped:
            report.results.append(FileResult(self._source_id(p, root, source_prefix), "skipped",
                                             error=f"unsupported extension {p.suffix!r}"))
        seen: set[str] = set()
        for p in files:
            sid = self._source_id(p, root, source_prefix)
            seen.add(sid)
            report.results.append(self._ingest_file(p, sid, report))

        if prune:
            for existing in self.store.source_summary():
                sid = existing["source_id"]
                if sid not in seen and (not source_prefix or sid.startswith(source_prefix)):
                    n = self.store.delete_source(sid)
                    report.sources_pruned.append(sid)
                    report.chunks_deleted += n
        report.count_after = self.store.count()
        report.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_event(log, "ingest_done", files=report.files_discovered, new=report.files_new,
                  updated=report.files_updated, unchanged=report.files_unchanged,
                  failed=report.files_failed, count_before=report.count_before,
                  count_after=report.count_after, duration_ms=report.duration_ms)
        return report

    @staticmethod
    def _source_id(p: Path, root: Path, prefix: str) -> str:
        return f"{prefix}{p.relative_to(root).as_posix()}"

    # ------------------------------------------------------------------ per file
    def _ingest_file(self, p: Path, sid: str, report: IngestReport) -> FileResult:
        try:
            doc = load_document(p, sid)
            chunks = self.chunker.chunk_document(doc)
            if not chunks:
                raise DocumentLoadError(f"{p.name}: produced no chunks")
        except (DocumentLoadError, UnsupportedFileType) as exc:
            report.files_failed += 1
            log_event(log, "ingest_file_failed", level=30, source_id=sid, error=exc.message)
            return FileResult(sid, "failed", error=exc.message)

        doc_id = hashing.document_id(sid, doc.content_hash)
        ids = [hashing.chunk_id(doc_id, c.index) for c in chunks]
        t_hashes = [hashing.text_hash(c.text) for c in chunks]
        existing = self.store.get_source_chunks(sid)

        unchanged = (set(existing) == set(ids) and
                     all(existing[i].get("text_hash") == th for i, th in zip(ids, t_hashes)))
        dup = self.store.find_by_content_hash(doc.content_hash, exclude_source=sid)
        warning = f"identical content also stored under: {', '.join(dup)}" if dup else None
        if unchanged:
            report.files_unchanged += 1
            return FileResult(sid, "unchanged", chunks=len(ids), warning=warning)

        # embed the context-prefixed text; store the clean chunk text as the document
        embed_inputs = [self._embedding_text(doc.title, c.section, c.text) for c in chunks]
        vectors = self.encoder.encode(embed_inputs)
        metadatas = [{
            "source_id": sid, "source_name": p.name, "document_type": doc.document_type,
            "category": doc.category, "title": doc.title, "doc_id": doc_id,
            "content_hash": doc.content_hash, "text_hash": th, "chunk_index": c.index,
            "page": c.page if c.page is not None else -1, "section": c.section,
            "token_count": c.token_count, "chunking_method": CHUNKING_METHOD,
            "chunk_size": self.chunker.chunk_size, "chunk_overlap": self.chunker.chunk_overlap,
            "embedding_model": self.encoder.name,
        } for c, th in zip(chunks, t_hashes)]
        self.store.upsert(ids, vectors, [c.text for c in chunks], metadatas)
        stale = [i for i in existing if i not in set(ids)]
        self.store.delete(stale)  # after upsert: never leaves the source empty on failure

        status = "updated" if existing else "new"
        if existing:
            report.files_updated += 1
        else:
            report.files_new += 1
        report.files_processed += 1
        report.chunks_upserted += len(ids)
        report.chunks_deleted += len(stale)
        return FileResult(sid, status, chunks=len(ids), chunks_deleted=len(stale), warning=warning)

    @staticmethod
    def _embedding_text(title: str, section: str, text: str) -> str:
        """Context-prefixed text used ONLY for embedding (document title + section heading)."""
        prefix = " - ".join(x for x in (title, section) if x)
        return f"{prefix}\n{text}" if prefix else text
