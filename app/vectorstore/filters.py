"""Metadata filter validation and translation to ChromaDB ``where`` clauses.

Filters narrow the set of eligible records *inside* the ANN query (Chroma applies the metadata
predicate while searching), so top-k is taken over matching records only.
"""
from __future__ import annotations

from typing import Any

from app.exceptions import InvalidFilterError

ALLOWED_FILTER_FIELDS = ("category", "document_type", "source_id", "source_name")
MAX_VALUES_PER_FIELD = 20


def validate_filters(filters: dict[str, Any] | None) -> dict[str, list[str]]:
    """Normalise to ``{field: [values]}``; raise InvalidFilterError on anything unexpected."""
    if not filters:
        return {}
    if not isinstance(filters, dict):
        raise InvalidFilterError("filters must be an object mapping field -> value or list of values")
    out: dict[str, list[str]] = {}
    for key, val in filters.items():
        if key not in ALLOWED_FILTER_FIELDS:
            raise InvalidFilterError(f"Unknown filter field {key!r}. Allowed: {list(ALLOWED_FILTER_FIELDS)}")
        vals = val if isinstance(val, list) else [val]
        if not vals or len(vals) > MAX_VALUES_PER_FIELD:
            raise InvalidFilterError(f"Filter {key!r} needs 1..{MAX_VALUES_PER_FIELD} values")
        if not all(isinstance(v, str) and v.strip() for v in vals):
            raise InvalidFilterError(f"Filter {key!r} values must be non-empty strings")
        out[key] = [v.strip() for v in vals]
    return out


def build_where(filters: dict[str, Any] | None) -> dict | None:
    norm = validate_filters(filters)
    if not norm:
        return None
    clauses = [({k: vs[0]} if len(vs) == 1 else {k: {"$in": vs}}) for k, vs in norm.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
