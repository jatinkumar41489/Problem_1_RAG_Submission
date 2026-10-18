"""Domain exceptions. Each maps to a safe, stable error code for API clients."""
from __future__ import annotations


class RAGError(Exception):
    """Base class for all application errors."""

    code = "rag_error"
    http_status = 500

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class ConfigError(RAGError):
    code = "configuration_error"
    http_status = 500


class UnsupportedFileType(RAGError):
    code = "unsupported_file_type"
    http_status = 400


class DocumentLoadError(RAGError):
    code = "document_load_error"
    http_status = 422


class PathNotAllowedError(RAGError):
    code = "path_not_allowed"
    http_status = 403


class IngestionError(RAGError):
    code = "ingestion_error"
    http_status = 500


class SourceNotFoundError(IngestionError):
    code = "source_not_found"
    http_status = 404


class EmbeddingError(RAGError):
    code = "embedding_error"
    http_status = 503


class VectorStoreError(RAGError):
    code = "vector_store_error"
    http_status = 503


class InvalidFilterError(RAGError):
    code = "invalid_filter"
    http_status = 422


class RetrievalError(RAGError):
    code = "retrieval_error"
    http_status = 500


class GenerationError(RAGError):
    """LLM call failed. ``category`` is one of: configuration, authentication,
    rate_limit, timeout, connection, model_not_found, bad_request, server_error,
    malformed_response, unknown."""

    code = "generation_error"
    http_status = 503

    def __init__(self, message: str, *, category: str = "unknown",
                 retryable: bool = False, detail: dict | None = None) -> None:
        super().__init__(message, detail=detail)
        self.category = category
        self.retryable = retryable
