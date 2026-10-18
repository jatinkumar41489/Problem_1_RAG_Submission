import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import answer_json


@pytest.fixture
def client(make_services):
    svc = make_services([answer_json(answer="Employees get 25 days [C1].", citations=["C1"])] * 20)
    with TestClient(create_app(services=svc)) as c:
        c.svc = svc
        yield c


def test_health_empty_store_and_no_secrets(client):
    r = client.get("/health")
    assert r.status_code == 200
    b = r.json()
    assert b["vector_store"]["count"] == 0 and b["generation"]["api_key_configured"] is True
    assert "test-key-not-real" not in r.text


def test_ingest_stats_and_idempotent_reingest(client):
    r1 = client.post("/ingest", json={"path": "data/sample_documents"}).json()
    assert r1["files_new"] == 12 and r1["collection_count_before"] == 0 and r1["collection_count_after"] > 0
    r2 = client.post("/ingest", json={"path": "data/sample_documents"}).json()
    assert r2["files_unchanged"] == 12 and r2["chunks_upserted"] == 0 and r2["collection_count_after"] == r1["collection_count_after"]
    s = client.get("/stats").json()
    assert s["vector_count"] == r1["collection_count_after"] and s["source_count"] == 12 and "hr" in s["chunks_by_category"]


@pytest.mark.parametrize("path", ["../..", "/etc", "data/../app", "app", "../problem1_rag/app"])
def test_ingest_path_traversal_blocked(client, path):
    r = client.post("/ingest", json={"path": path})
    assert r.status_code == 403 and r.json()["error"]["code"] == "path_not_allowed"


def test_ingest_missing_path(client):
    r = client.post("/ingest", json={"path": "data/nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "source_not_found"


def test_query_response_structure(client):
    client.post("/ingest", json={"path": "data/sample_documents"})
    r = client.post("/query", json={"query": "How many days of paid annual leave?", "k": 3, "request_id": "abc-1"})
    assert r.status_code == 200
    b = r.json()
    assert b["request_id"] == "abc-1" and b["status"] == "answered" and not b["abstained"]
    assert b["retrieved_chunk_count"] == 3 == len(b["retrieved_chunks"])
    assert b["citations"][0]["chunk_id"] in {c["chunk_id"] for c in b["retrieved_chunks"]}
    assert set(b["latency"]) == {"retrieval_ms", "generation_ms", "end_to_end_ms"}
    assert b["token_usage"]["total_tokens"] == 366 and b["models"]["embedding"] == "hash-bow-384"


def test_query_empty_store_abstains(client):
    b = client.post("/query", json={"query": "anything"}).json()
    assert b["status"] == "no_relevant_context" and b["abstained"] and b["reason"] == "empty_store"
    assert b["token_usage"]["available"] is False and b["latency"]["generation_ms"] is None


def test_query_metadata_filter(client):
    client.post("/ingest", json={"path": "data/sample_documents"})
    b = client.post("/retrieve", json={"query": "price plan", "k": 10, "filters": {"category": "product"}}).json()
    assert b["retrieved_chunks"] and {c["category"] for c in b["retrieved_chunks"]} == {"product"}


@pytest.mark.parametrize("body,code", [({}, 422), ({"query": ""}, 422), ({"query": "   "}, 422), ({"query": "x", "k": 0}, 422),
                                        ({"query": "x", "k": 999}, 422), ({"query": "x", "request_id": "bad id!"}, 422),
                                        ({"query": "x", "filters": {"bogus": "a"}}, 422)])
def test_query_validation(client, body, code):
    r = client.post("/query", json=body)
    assert r.status_code == code and "error" in r.json()


def test_generation_failure_returns_error_status_with_retrieval(make_services):
    import groq, httpx
    err = groq.APITimeoutError(request=httpx.Request("POST", "https://x"))
    svc = make_services([err] * 10)
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    with TestClient(create_app(services=svc)) as c:
        r = c.post("/query", json={"query": "annual leave days", "k": 3})
    b = r.json()
    assert r.status_code == 504 and b["status"] == "generation_error" and b["error"]["category"] == "timeout"
    assert len(b["retrieved_chunks"]) == 3     # retrieval result is still returned


def test_unhandled_error_does_not_leak_internals(client, monkeypatch):
    monkeypatch.setattr(client.svc.answers, "answer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret internal detail /home/x")))
    c = TestClient(client.app, raise_server_exceptions=False)
    r = c.post("/query", json={"query": "x"})
    assert r.status_code == 500 and "secret internal" not in r.text and "error_id" in r.text


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200 and "/query" in client.get("/openapi.json").json()["paths"]
