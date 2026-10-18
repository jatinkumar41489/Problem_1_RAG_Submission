"""Embedding backends.

* ``sentence-transformers`` (default, production): all-MiniLM-L6-v2, 384-d, run locally.
* ``hash`` (offline fallback): a deterministic hashed bag-of-words (uni+bigram) projection to the
  same dimensionality. It needs no model download and exists so tests/CI and air-gapped machines can
  run the full pipeline. It is a LEXICAL baseline, NOT a semantic model; results produced with it
  must never be reported as all-MiniLM-L6-v2 results. Every results file records the backend.

All encoders return float32, L2-normalised vectors, so cosine similarity == dot product and the
same preprocessing applies to documents and queries.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from app.config import Settings
from app.exceptions import EmbeddingError


class Encoder(ABC):
    name: str
    dim: int
    backend: str

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray: ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts -> (n, dim) float32, unit-norm. Validates shape and finiteness."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        t0 = time.perf_counter()
        vecs = np.asarray(self._encode(texts), dtype=np.float32)
        self.last_encode_ms = (time.perf_counter() - t0) * 1000.0
        if vecs.ndim != 2 or vecs.shape != (len(texts), self.dim):
            raise EmbeddingError(f"Embedding shape {vecs.shape} != expected {(len(texts), self.dim)} "
                                 f"for model {self.name}.")
        if not np.isfinite(vecs).all():
            raise EmbeddingError("Embedding contains NaN/inf values.")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # all-zero vector (e.g. text of only stop-words) stays zero
        return (vecs / norms).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def verify_dimension(self) -> int:
        """Programmatically verify the output dimensionality (used by verify_setup / tests)."""
        v = self.encode(["dimension probe"])
        if v.shape[1] != self.dim:
            raise EmbeddingError(f"Expected {self.dim}-d embeddings, got {v.shape[1]}-d.")
        return int(v.shape[1])

    @property
    def loaded(self) -> bool:
        return True


class SentenceTransformerEncoder(Encoder):
    backend = "sentence-transformers"

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        self.name, self.dim, self.batch_size = model_name, dim, batch_size
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Run: pip install -r requirements.txt "
                    "(or set EMBEDDING_BACKEND=hash for the offline lexical fallback).") from exc
            try:
                self._model = SentenceTransformer(self.name)
            except Exception as exc:  # network/cache/model-id failures
                raise EmbeddingError(
                    f"Could not load embedding model {self.name!r} ({type(exc).__name__}). The first run "
                    "downloads the model from huggingface.co; check your internet/proxy, or pre-download it. "
                    "Offline fallback: EMBEDDING_BACKEND=hash.") from exc
            actual = self._model.get_sentence_embedding_dimension()
            if actual != self.dim:
                self._model = None
                raise EmbeddingError(f"Model {self.name} produces {actual}-d vectors but EMBEDDING_DIM={self.dim}.")
        return self._model

    def warm_up(self) -> None:
        self._load()

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=False)


class HashingEncoder(Encoder):
    backend = "hash"

    def __init__(self, dim: int = 384) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.name = f"hash-bow-{dim}"
        self._vec = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=True,
                                      norm=None, lowercase=True, stop_words="english")

    def _encode(self, texts: list[str]) -> np.ndarray:
        x = self._vec.transform(texts).astype(np.float32)
        x.data = np.sign(x.data) * np.log1p(np.abs(x.data))  # sub-linear tf
        return x.toarray()


def get_encoder(settings: Settings) -> Encoder:
    if settings.embedding_backend == "hash":
        return HashingEncoder(settings.embedding_dim)
    return SentenceTransformerEncoder(settings.embedding_model, settings.embedding_dim,
                                      settings.embedding_batch_size)
