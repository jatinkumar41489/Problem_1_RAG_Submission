"""ChromaDB persistent vector store wrapper.

Distance semantics: the collection uses cosine space, Chroma returns ``distance = 1 - cosine_sim``
(range 0..2). We expose ``score = 1 - distance`` = cosine similarity, higher is better.
Never silently resets or deletes an existing collection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.exceptions import VectorStoreError
from app.vectorstore.filters import build_where

_BATCH = 500


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    distance: float
    metadata: dict[str, Any]


def _clean_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool; drop None and coerce everything else to str."""
    out = {}
    for k, v in md.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


class ChromaStore:
    def __init__(self, persist_dir: Path | str, collection_name: str,
                 embedding_model: str, embedding_dim: int) -> None:
        import chromadb

        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        try:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._col = self._client.get_or_create_collection(
                name=collection_name, embedding_function=None,
                metadata={"hnsw:space": "cosine", "embedding_model": embedding_model,
                          "embedding_dim": embedding_dim})
        except Exception as exc:
            raise VectorStoreError(f"Cannot open ChromaDB at {self.persist_dir}: {exc}") from exc
        meta = self._col.metadata or {}
        if meta.get("embedding_model") not in (None, embedding_model) or \
                int(meta.get("embedding_dim", embedding_dim)) != embedding_dim:
            raise VectorStoreError(
                f"Collection {collection_name!r} was created with embedding model "
                f"{meta.get('embedding_model')!r} ({meta.get('embedding_dim')}-d) but the app is configured "
                f"for {embedding_model!r} ({embedding_dim}-d). Use a different CHROMA_COLLECTION_NAME / "
                "CHROMA_PERSIST_DIRECTORY or re-ingest into a new collection. Nothing was modified.")

    # ------------------------------------------------------------ basic ops
    def count(self) -> int:
        try:
            return int(self._col.count())
        except Exception as exc:
            raise VectorStoreError(f"count() failed: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return {"ok": True, "collection": self.collection_name, "count": self.count(),
                "persist_directory": self.persist_dir, "distance": "cosine",
                "embedding_model": self.embedding_model, "embedding_dim": self.embedding_dim}

    def upsert(self, ids: list[str], embeddings: np.ndarray, documents: list[str],
               metadatas: list[dict[str, Any]]) -> None:
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise VectorStoreError("upsert: ids/embeddings/documents/metadatas length mismatch")
        if len(set(ids)) != len(ids):
            raise VectorStoreError("upsert: duplicate ids within one batch")
        try:
            for i in range(0, len(ids), _BATCH):
                s = slice(i, i + _BATCH)
                self._col.upsert(ids=ids[s], embeddings=embeddings[s].tolist(), documents=documents[s],
                                 metadatas=[_clean_metadata(m) for m in metadatas[s]])
        except Exception as exc:
            raise VectorStoreError(f"upsert failed: {exc}") from exc

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            for i in range(0, len(ids), _BATCH):
                self._col.delete(ids=ids[i:i + _BATCH])
        except Exception as exc:
            raise VectorStoreError(f"delete failed: {exc}") from exc

    def get_source_chunks(self, source_id: str) -> dict[str, dict[str, Any]]:
        """{chunk_id: metadata} for every stored chunk of ``source_id``."""
        try:
            r = self._col.get(where={"source_id": source_id}, include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"get_source_chunks failed: {exc}") from exc
        return dict(zip(r["ids"], r["metadatas"]))

    def find_by_content_hash(self, c_hash: str, exclude_source: str) -> list[str]:
        r = self._col.get(where={"content_hash": c_hash}, include=["metadatas"])
        return sorted({m["source_id"] for m in r["metadatas"] if m.get("source_id") != exclude_source})

    def delete_source(self, source_id: str) -> int:
        ids = list(self.get_source_chunks(source_id))
        self.delete(ids)
        return len(ids)

    def all_chunks(self) -> list[dict[str, Any]]:
        """Every stored chunk (id, text, metadata) - used by evaluation to build relevance labels."""
        r = self._col.get(include=["documents", "metadatas"])
        return [{"chunk_id": i, "text": d, "metadata": m}
                for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])]

    def source_summary(self) -> list[dict[str, Any]]:
        r = self._col.get(include=["metadatas"])
        agg: dict[str, dict[str, Any]] = {}
        for m in r["metadatas"]:
            a = agg.setdefault(m["source_id"], {"source_id": m["source_id"], "chunks": 0,
                                                "category": m.get("category"),
                                                "document_type": m.get("document_type")})
            a["chunks"] += 1
        return sorted(agg.values(), key=lambda a: a["source_id"])

    # ------------------------------------------------------------ query
    def query(self, embedding: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[Hit]:
        where = build_where(filters)  # raises InvalidFilterError (not wrapped)
        total = self.count()
        if total == 0:
            return []
        try:
            r = self._col.query(query_embeddings=[embedding.tolist()], n_results=min(k, total),
                                where=where, include=["documents", "metadatas", "distances"])
        except Exception as exc:
            raise VectorStoreError(f"query failed: {exc}") from exc
        hits = []
        for cid, doc, md, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            hits.append(Hit(chunk_id=cid, text=doc, score=float(1.0 - dist), distance=float(dist), metadata=md))
        return hits

    def close(self) -> None:
        """Release file handles (matters on Windows before deleting the directory)."""
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
