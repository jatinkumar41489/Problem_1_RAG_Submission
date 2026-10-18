import json
from pathlib import Path

import pytest

from app.config import BASE_DIR
from app.evaluation.calibration import calibrate
from app.evaluation.dataset import chunk_grade, load_dataset
from app.evaluation.run_evaluation import build_eval_services, run_all, run_answer_eval, run_retrieval_eval
from app.ingestion.hashing import normalize_text
from app.ingestion.loaders import load_document
from tests.conftest import SAMPLE_DIR, answer_json

DATASET = BASE_DIR / "data" / "evaluation"


def test_dataset_size_ids_and_types():
    qs = load_dataset(DATASET)
    assert 15 <= len(qs) <= 30 and len({q.id for q in qs}) == len(qs)
    types = {q.type for q in qs}
    assert {"factual", "multi_document", "metadata_filter", "pdf_page", "synthesis", "unanswerable", "filter_excludes_evidence"} <= types
    for q in qs:
        assert q.answerable == bool(q.evidence) == bool(q.gold_answers)


def test_every_evidence_phrase_exists_on_its_page_and_is_short_enough_for_overlap():
    for q in load_dataset(DATASET):
        for ev in q.evidence:
            d = load_document(SAMPLE_DIR / ev["source_id"], ev["source_id"])
            segs = [s for s in d.segments if ev.get("page") in (None, s.page)]
            assert any(normalize_text(ev["evidence"]).lower() in normalize_text(s.text).lower() for s in segs), (q.id, ev)
            assert len(ev["evidence"].split()) <= 15     # <= smallest tested overlap => always inside >=1 chunk


def test_chunk_grading_respects_source_page_and_grade():
    ev = [{"source_id": "a.md", "evidence": "25 days", "grade": 2, "page": 2}]
    ok = {"text": "You get 25   DAYS off", "metadata": {"source_id": "a.md", "page": 2}}
    assert chunk_grade(ok, ev) == 2
    assert chunk_grade({**ok, "metadata": {"source_id": "a.md", "page": 3}}, ev) == 0
    assert chunk_grade({**ok, "metadata": {"source_id": "b.md", "page": 2}}, ev) == 0


def test_retrieval_eval_produces_valid_metrics(services):
    svc = build_eval_services(services.settings, Path(services.settings.chroma_persist_directory) / "e")
    try:
        qs = load_dataset(DATASET)
        res = run_retrieval_eval(svc, qs, 5)
        assert res["excluded_queries"] == []
        for kname, m in res["by_k"].items():
            assert m["n_eligible_queries"] == 23
            for key in ("hit_rate_at_k", "mrr_at_k", "ndcg_at_k", "context_precision_at_k", "recall_at_k_chunk"):
                assert 0 <= m[key] <= 1
        assert res["by_k"]["k=10"]["hit_rate_at_k"] >= res["by_k"]["k=1"]["hit_rate_at_k"]      # hit rate is monotone in k
        assert res["by_k"]["k=1"]["hit_rate_at_k"] == pytest.approx(res["by_k"]["k=1"]["mrr_at_k"])   # at k=1 MRR == hit rate
        assert len(res["per_query"]) == 29
    finally:
        svc.close()


def test_filter_aware_ground_truth(services):
    svc = build_eval_services(services.settings, Path(services.settings.chroma_persist_directory) / "f")
    try:
        qs = {q.id: q for q in load_dataset(DATASET)}
        res = run_retrieval_eval(svc, [qs[i] for i in ["Q13", "Q14", "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08", "Q09", "Q10", "Q11", "Q12", "Q15"]], 5)
        row = next(r for r in res["per_query"] if r["id"] == "Q13")
        assert row["relevant_chunks_in_corpus"] >= 1
    finally:
        svc.close()


def test_calibration_separates_classes_and_reports_loo():
    c = calibrate([0.9, 0.8, 0.7, 0.2, 0.1, 0.15], [True, True, True, False, False, False])
    assert c["status"] == "ok" and 0.2 < c["recommended_min_relevance_score"] < 0.7 and c["balanced_accuracy_in_sample"] == 1.0
    assert calibrate([0.5, 0.6], [True, True])["status"] == "not_calibrated"


def test_answer_eval_not_run_without_key_and_never_fabricates(services, tmp_path):
    import dataclasses
    s = dataclasses.replace(services.settings, groq_api_key=None)
    from app.services import Services
    svc = Services(s)
    r = run_answer_eval(svc, load_dataset(DATASET), 5, None)
    svc.close()
    assert r["status"] == "not_run" and "GROQ_API_KEY" in r["reason"] and "aggregate" not in r


def test_answer_eval_with_mocked_llm_and_judge(make_services):
    judge_json = json.dumps({"faithfulness": 5, "relevance": 4, "supported_claims": ["c"], "unsupported_claims": [], "rationale": "ok"})
    svc = make_services([answer_json(answer="25 days [C1]", citations=["C1"]), judge_json] * 40, min_relevance_score=-1.0)
    svc.pipeline.ingest_path(SAMPLE_DIR)
    from app.evaluation.judge import LLMJudge
    judge = LLMJudge(svc.settings, svc.llm)
    qs = [q for q in load_dataset(DATASET) if q.id in ("Q01", "Q24")]
    r = run_answer_eval(svc, qs, 3, judge)
    a = r["aggregate"]
    assert r["status"] == "completed" and a["faithfulness_mean"] == 1.0 and a["answer_relevance_mean"] == 0.75
    assert a["f1"] is not None and a["answered_when_unanswerable"] == 1        # mocked LLM 'hallucinates' on Q24 -> counted
    assert r["judge_model"] != r["generator_model"]


def test_judge_malformed_output_is_unavailable_not_fabricated(make_services):
    from app.evaluation.judge import LLMJudge
    svc = make_services(["not json", json.dumps({"faithfulness": 9, "relevance": 3})])
    j = LLMJudge(svc.settings, svc.llm)
    assert j.judge("q", "a", ["c"]).available is False
    assert j.judge("q", "a", ["c"]).available is False      # out-of-range score rejected


def test_run_all_writes_every_artifact_offline(settings, tmp_path):
    import dataclasses
    s = dataclasses.replace(settings, groq_api_key=None, benchmark_warmup=1, benchmark_repeats=1)
    out = run_all(s, k=5, with_chunking=True)
    d = s.output_dir
    for f in ("retrieval_metrics.json", "retrieval_per_query.csv", "answer_quality.json", "latency_results.json",
              "latency_samples.csv", "chunking_experiment.json", "evaluation_summary.md"):
        assert (d / f).exists(), f
    aq = json.loads((d / "answer_quality.json").read_text())
    assert aq["status"] == "not_run"
    assert "hash" in (d / "evaluation_summary.md").read_text() and "WARNING" in (d / "evaluation_summary.md").read_text()
