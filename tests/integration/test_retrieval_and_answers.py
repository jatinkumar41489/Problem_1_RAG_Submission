import pytest

from app.exceptions import InvalidFilterError, RetrievalError
from tests.conftest import answer_json


def test_relevant_chunk_retrieved_with_metadata(ingested):
    # The offline lexical embedder is not reliable at rank 1, so assert presence in the top-5.
    r = ingested.retriever.retrieve("What is the dinner meal limit per person when travelling?", k=5)
    hit = next(c for c in r.chunks if c.source_id == "finance/expense_policy.pdf" and c.page == 1)
    assert "dinner up to $60" in " ".join(hit.text.split())
    assert 0 < hit.score <= 1 and abs(hit.score + hit.distance - 1) < 1e-9
    assert [c.rank for c in r.chunks] == [1, 2, 3, 4, 5]
    assert all(a.score >= b.score for a, b in zip(r.chunks, r.chunks[1:]))
    assert r.embed_ms >= 0 and r.search_ms >= 0 and r.total_ms >= r.search_ms


def test_metadata_filter_returns_only_matching_records(ingested):
    for flt, field in (({"category": "hr"}, "category"), ({"document_type": "pdf"}, "document_type"),
                       ({"source_id": "product/atlas_pricing_and_limits.md"}, "source_id")):
        r = ingested.retriever.retrieve("incident password price leave", k=10, filters=flt)
        assert r.chunks and all(getattr(c, field) == list(flt.values())[0] for c in r.chunks)


def test_filter_excluding_everything_gives_no_matching_chunks(ingested):
    r = ingested.retriever.retrieve("Atlas pricing", k=5, filters={"category": "nonexistent"})
    assert r.chunks == [] and r.decision.reason == "no_matching_chunks" and not r.decision.relevant


def test_k_and_query_validation(ingested):
    for k in (0, -1, 999, True):
        with pytest.raises(RetrievalError):
            ingested.retriever.retrieve("x", k=k)
    with pytest.raises(RetrievalError):
        ingested.retriever.retrieve("   ")
    with pytest.raises(InvalidFilterError):
        ingested.retriever.retrieve("x", filters={"bogus": "a"})
    assert len(ingested.retriever.retrieve("leave", k=2).chunks) == 2


def test_k_larger_than_store_returns_all(ingested):
    assert len(ingested.retriever.retrieve("leave", k=50).chunks) == ingested.store.count()


# ------------------------------------------------------------------ answer service paths
def test_empty_store_abstains_without_calling_llm(make_services):
    svc = make_services([answer_json()])
    r = svc.answers.answer("How many days of leave?")
    assert r.status == "no_relevant_context" and r.reason == "empty_store" and svc.fake_calls.calls == []


def test_unrelated_query_abstains_without_llm(ingested):
    r = ingested.answers.answer("zzzz qqqq xxxx blorp")
    assert r.status == "no_relevant_context" and ingested.fake_calls.calls == []


def test_filter_excluding_relevant_docs_abstains(ingested):
    r = ingested.answers.answer("What is the price of the Atlas Pro plan?", filters={"category": "hr"})
    assert r.status == "no_relevant_context"


def test_low_relevance_threshold_blocks(make_services):
    svc = make_services([answer_json()], min_relevance_score=0.99)
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    r = svc.answers.answer("annual leave days")
    assert r.status == "no_relevant_context" and r.reason == "low_relevance"


def test_grounded_answer_with_valid_citation_and_usage(make_services):
    svc = make_services([answer_json(answer="Full-time employees get 25 days [C1].", citations=["C1"])])
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    r = svc.answers.answer("How many days of paid annual leave do employees get?", k=3)
    assert r.status == "answered" and len(r.citations) == 1
    assert r.citations[0].chunk_id in {c.chunk_id for c in r.retrieval.chunks}      # resolves to a retrieved chunk
    assert r.token_usage == {"available": True, "prompt_tokens": 321, "completion_tokens": 45, "total_tokens": 366}
    assert r.generation_ms is not None and r.total_ms >= r.retrieval_ms
    sys_prompt = svc.fake_calls.calls[0]["messages"][0]["content"]
    assert "untrusted" in sys_prompt and "ONLY" in sys_prompt


def test_invalid_citation_is_reported_not_presented_as_valid(make_services):
    svc = make_services([answer_json(answer="x [C1] [C99]", citations=["C1", "C99"])])
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    r = svc.answers.answer("annual leave days", k=3)
    assert r.status == "answered" and r.invalid_citations == ["C99"] and [c.ref for c in r.citations] == ["C1"]


def test_answer_without_valid_citation_is_withheld(make_services):
    svc = make_services([answer_json(answer="25 days.", citations=[])])
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    r = svc.answers.answer("annual leave days", k=3)
    assert r.status == "insufficient_evidence" and r.reason == "no_valid_citations" and r.citations == []


def test_model_reports_insufficient_evidence(make_services):
    svc = make_services([answer_json(answerable=False, answer="The context does not mention it.", citations=[])])
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    r = svc.answers.answer("annual leave days", k=3)
    assert r.status == "insufficient_evidence" and r.reason == "model_reported_insufficient"


def test_generation_failure_after_retrieval_is_distinct(make_services):
    import groq, httpx
    req = httpx.Request("POST", "https://x")
    err = groq.AuthenticationError("bad", response=httpx.Response(401, request=req), body=None)
    svc = make_services([err])
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    r = svc.answers.answer("annual leave days", k=3)
    assert r.status == "generation_error" and r.error_category == "authentication" and len(r.retrieval.chunks) == 3


def test_malformed_model_output_handled(make_services):
    svc = make_services(["I think the answer is 25 days"])
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    r = svc.answers.answer("annual leave days", k=3)
    assert r.status == "generation_error" and r.error_category == "malformed_response"


def test_json_in_code_fence_is_parsed(make_services):
    svc = make_services(["```json\n" + answer_json() + "\n```"])
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    assert svc.answers.answer("annual leave days", k=3).status == "answered"


def test_oversized_context_is_trimmed(make_services):
    svc = make_services([answer_json()], max_context_tokens=250)
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    r = svc.answers.answer("annual leave days", k=5)
    assert r.context_truncated and 1 <= r.context_chunks_used < 5


def test_query_log_has_required_fields(make_services, caplog):
    import logging
    svc = make_services([answer_json()])
    svc.pipeline.ingest_path(__import__("tests.conftest", fromlist=["x"]).SAMPLE_DIR)
    logging.getLogger("rag").propagate = True
    with caplog.at_level(logging.INFO, logger="rag.query"):
        svc.answers.answer("annual leave days", k=3, request_id="rid-1")
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "query")
    for f in ("request_id", "retrieval_ms", "generation_ms", "total_ms", "retrieved_chunks", "total_tokens", "status", "model"):
        assert hasattr(rec, f)
    assert rec.request_id == "rid-1" and rec.retrieved_chunks == 3 and rec.total_tokens == 366
    assert not hasattr(rec, "query")      # raw query text is not logged by default
