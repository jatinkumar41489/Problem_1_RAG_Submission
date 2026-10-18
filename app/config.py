"""Central configuration. All values come from environment variables (optionally a .env file).

Relative paths are resolved against the project root, never the current working directory,
so the project has no dependency on absolute paths from the developer's machine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from app.exceptions import ConfigError

BASE_DIR = Path(__file__).resolve().parent.parent
_SECRET_FIELDS = {"groq_api_key"}


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def _int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a number, got {raw!r}.") from exc


def _bool(name: str, default: bool) -> bool:
    raw = _env(name)
    return default if raw is None else raw.lower() in {"1", "true", "yes", "on"}


def resolve_path(p: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths are kept as-is)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


@dataclass(frozen=True)
class Settings:
    # --- LLM (Groq) ---
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    judge_model: str = "openai/gpt-oss-120b"
    groq_timeout: float = 30.0
    groq_max_retries: int = 3
    max_context_tokens: int = 3000
    max_output_tokens: int = 512
    # --- Vector store ---
    chroma_persist_directory: str = "chroma_data"
    chroma_collection_name: str = "rag_chunks"
    # --- Embeddings ---
    embedding_backend: str = "sentence-transformers"  # or "hash" (offline fallback)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    # --- Chunking / retrieval ---
    chunk_size: int = 180
    chunk_overlap: int = 30
    top_k: int = 5
    max_top_k: int = 50
    min_relevance_score: float = 0.30
    # --- Ops ---
    log_level: str = "INFO"
    log_query_text: bool = False
    allowed_ingest_roots: str = "data"
    # --- Evaluation ---
    evaluation_dataset_path: str = "data/evaluation"
    evaluation_output_dir: str = "results"
    sample_documents_dir: str = "data/sample_documents"
    benchmark_warmup: int = 5
    benchmark_repeats: int = 5

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env_file: Optional[str | Path] = None, validate: bool = True) -> "Settings":
        load_dotenv(env_file or BASE_DIR / ".env", override=False)
        s = cls(
            groq_api_key=_env("GROQ_API_KEY"),
            groq_model=_env("GROQ_MODEL", cls.groq_model),
            judge_model=_env("JUDGE_MODEL", cls.judge_model),
            groq_timeout=_float("GROQ_TIMEOUT", cls.groq_timeout),
            groq_max_retries=_int("GROQ_MAX_RETRIES", cls.groq_max_retries),
            max_context_tokens=_int("MAX_CONTEXT_TOKENS", cls.max_context_tokens),
            max_output_tokens=_int("MAX_OUTPUT_TOKENS", cls.max_output_tokens),
            chroma_persist_directory=_env("CHROMA_PERSIST_DIRECTORY", cls.chroma_persist_directory),
            chroma_collection_name=_env("CHROMA_COLLECTION_NAME", cls.chroma_collection_name),
            embedding_backend=_env("EMBEDDING_BACKEND", cls.embedding_backend).lower(),
            embedding_model=_env("EMBEDDING_MODEL", cls.embedding_model),
            embedding_dim=_int("EMBEDDING_DIM", cls.embedding_dim),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", cls.embedding_batch_size),
            chunk_size=_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_int("TOP_K", cls.top_k),
            max_top_k=_int("MAX_TOP_K", cls.max_top_k),
            min_relevance_score=_float("MIN_RELEVANCE_SCORE", cls.min_relevance_score),
            log_level=_env("LOG_LEVEL", cls.log_level).upper(),
            log_query_text=_bool("LOG_QUERY_TEXT", cls.log_query_text),
            allowed_ingest_roots=_env("ALLOWED_INGEST_ROOTS", cls.allowed_ingest_roots),
            evaluation_dataset_path=_env("EVALUATION_DATASET_PATH", cls.evaluation_dataset_path),
            evaluation_output_dir=_env("EVALUATION_OUTPUT_DIR", cls.evaluation_output_dir),
            sample_documents_dir=_env("SAMPLE_DOCUMENTS_DIR", cls.sample_documents_dir),
            benchmark_warmup=_int("BENCHMARK_WARMUP", cls.benchmark_warmup),
            benchmark_repeats=_int("BENCHMARK_REPEATS", cls.benchmark_repeats),
        )
        if validate:
            s.validate()
        return s

    def validate(self) -> None:
        """Validate values that make the application unusable if wrong.

        A missing GROQ_API_KEY is *not* fatal here: ingestion and retrieval work without it, and
        generation reports a clear ``configuration`` error. See ``require_groq``.
        """
        errs = []
        if self.chunk_size <= 0:
            errs.append("CHUNK_SIZE must be > 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            errs.append("CHUNK_OVERLAP must satisfy 0 <= overlap < CHUNK_SIZE")
        if not (1 <= self.top_k <= self.max_top_k):
            errs.append(f"TOP_K must be between 1 and MAX_TOP_K ({self.max_top_k})")
        if not (-1.0 <= self.min_relevance_score <= 1.0):
            errs.append("MIN_RELEVANCE_SCORE must be within [-1, 1] (cosine similarity)")
        if self.embedding_backend not in {"sentence-transformers", "hash"}:
            errs.append("EMBEDDING_BACKEND must be 'sentence-transformers' or 'hash'")
        if self.embedding_dim <= 0 or self.embedding_batch_size <= 0:
            errs.append("EMBEDDING_DIM and EMBEDDING_BATCH_SIZE must be > 0")
        if self.max_context_tokens < 200 or self.max_output_tokens < 16:
            errs.append("MAX_CONTEXT_TOKENS must be >= 200 and MAX_OUTPUT_TOKENS >= 16")
        if self.groq_timeout <= 0 or self.groq_max_retries < 0:
            errs.append("GROQ_TIMEOUT must be > 0 and GROQ_MAX_RETRIES >= 0")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            errs.append("LOG_LEVEL must be DEBUG/INFO/WARNING/ERROR/CRITICAL")
        if errs:
            raise ConfigError("Invalid configuration: " + "; ".join(errs))

    def require_groq(self) -> str:
        if not self.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and set GROQ_API_KEY "
                "(or set it in the current shell, e.g. PowerShell: $env:GROQ_API_KEY='...')."
            )
        return self.groq_api_key

    # --- derived paths ---
    @property
    def persist_path(self) -> Path:
        return resolve_path(self.chroma_persist_directory)

    @property
    def dataset_dir(self) -> Path:
        return resolve_path(self.evaluation_dataset_path)

    @property
    def output_dir(self) -> Path:
        return resolve_path(self.evaluation_output_dir)

    @property
    def sample_dir(self) -> Path:
        return resolve_path(self.sample_documents_dir)

    @property
    def ingest_roots(self) -> list[Path]:
        return [resolve_path(p.strip()) for p in self.allowed_ingest_roots.split(",") if p.strip()]

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to log / return: secrets are replaced by a boolean."""
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = (v is not None) if f.name in _SECRET_FIELDS else v
        d["groq_api_key_configured"] = d.pop("groq_api_key")
        return d

    def __repr__(self) -> str:  # never leak the key through repr()/tracebacks
        return f"Settings({self.public_dict()})"
