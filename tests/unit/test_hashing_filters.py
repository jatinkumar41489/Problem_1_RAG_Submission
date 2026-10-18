import pytest

from app.exceptions import InvalidFilterError
from app.ingestion import hashing
from app.vectorstore.filters import build_where, validate_filters


def test_content_hash_ignores_whitespace_only_changes():
    a = hashing.content_hash([(None, "Hello   world\n")])
    assert a == hashing.content_hash([(None, "Hello world")])
    assert a != hashing.content_hash([(None, "Hello world!")])
    assert a != hashing.content_hash([(1, "Hello world")])   # page structure matters


def test_ids_are_stable_and_unique_per_source_and_content():
    h = hashing.content_hash([(None, "x")])
    assert hashing.document_id("a/r.md", h) == hashing.document_id("a/r.md", h)
    assert hashing.document_id("a/r.md", h) != hashing.document_id("b/r.md", h)
    assert hashing.chunk_id("abc", 3) == "abc-0003"


def test_where_construction():
    assert build_where(None) is None and build_where({}) is None
    assert build_where({"category": "hr"}) == {"category": "hr"}
    assert build_where({"category": ["hr", "finance"]}) == {"category": {"$in": ["hr", "finance"]}}
    assert build_where({"category": "hr", "document_type": "pdf"}) == {"$and": [{"category": "hr"}, {"document_type": "pdf"}]}


@pytest.mark.parametrize("bad", [{"password": "x"}, {"category": 5}, {"category": []}, {"category": ""}, ["category"], {"category": ["a"] * 21}])
def test_invalid_filters(bad):
    with pytest.raises(InvalidFilterError):
        validate_filters(bad)
