"""Evaluation harness: retrieval metrics, abstention calibration, answer quality, chunking
experiment, latency benchmark and summary. Every number is computed from a real run; stages that
cannot run (e.g. no GROQ_API_KEY) are recorded as ``not_run`` with the reason - never filled in.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.evaluation import answer_metrics as am
from app.evaluation import retrieval_metrics as rm
from app.evaluation.calibration import calibrate
from app.evaluation.dataset import Question, chunk_grade, evidence_found, load_dataset
from app.evaluation.judge import LLMJudge
from app.evaluation.latency import environment_info, run_retrieval_benchmark
from app.generation.answer_service import STATUS_ANSWERED
from app.logging_config import get_logger
from app.services import Services
from app.vectorstore.filters import validate_filters

log = get_logger("eval")
K_LIST = (1, 3, 5, 10)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def dump_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- services for eval
def build_eval_services(settings: Settings, workdir: Path, collection: str = "eval_chunks") -> Services:
    """Fresh, isolated store ingesting the sample corpus (never touches the user's main store)."""
    s = dataclasses.replace(settings, chroma_persist_directory=str(workdir), chroma_collection_name=collection)
    svc = Services(s)
    svc.pipeline.ingest_path(settings.sample_dir)
    return svc


def _eligible(meta: dict, filters: dict | None) -> bool:
    for k, vals in validate_filters(filters).items():
        if meta.get(k) not in vals:
            return False
    return True


def _as_chunk(rc) -> dict:
    return {"chunk_id": rc.chunk_id, "text": rc.text,
            "metadata": {"source_id": rc.source_id, "page": rc.page if rc.page is not None else -1}}


# --------------------------------------------------------------------------- retrieval
def run_retrieval_eval(svc: Services, questions: list[Question], primary_k: int) -> dict:
    all_chunks = svc.store.all_chunks()
    kmax = max(max(K_LIST), primary_k)
    per_query, scores_for_calib = [], []
    agg_inputs: dict[int, dict[str, list]] = {k: {m: [] for m in ("recall", "ev_recall", "hit", "rr", "ndcg", "prec")} for k in K_LIST}
    if primary_k not in agg_inputs:
        agg_inputs[primary_k] = {m: [] for m in ("recall", "ev_recall", "hit", "rr", "ndcg", "prec")}
    excluded = []
    for q in questions:
        r = svc.retriever.retrieve(q.question, k=kmax, filters=q.filters)
        top = r.decision.top_score
        scores_for_calib.append((q.id, top if top is not None else -1.0, q.answerable))
        row: dict[str, Any] = {"id": q.id, "type": q.type, "answerable": q.answerable, "filters": json.dumps(q.filters) if q.filters else "",
                               "top1_score": None if top is None else round(top, 4),
                               "gate_abstains_at_threshold": not r.decision.relevant,
                               "retrieved_ids@k": " ".join(c.chunk_id for c in r.chunks[:primary_k])}
        if q.answerable:
            elig = [c for c in all_chunks if _eligible(c["metadata"], q.filters)]
            corpus_grades = [chunk_grade(c, q.evidence) for c in elig]
            total_rel = sum(1 for g in corpus_grades if g >= 1)
            ranked = [chunk_grade(_as_chunk(c), q.evidence) for c in r.chunks]
            row.update(relevant_chunks_in_corpus=total_rel, ranked_grades=" ".join(map(str, ranked[:primary_k])),
                       first_relevant_rank=next((i for i, g in enumerate(ranked, 1) if g >= 1), None))
            if total_rel == 0:
                excluded.append(q.id)
                row["excluded"] = "no relevant chunk under this chunking/filter"
            else:
                for k in agg_inputs:
                    topk = [_as_chunk(c) for c in r.chunks[:k]]
                    agg_inputs[k]["recall"].append(rm.recall_at_k(ranked, total_rel, k))
                    agg_inputs[k]["ev_recall"].append(rm.evidence_recall(evidence_found(topk, q.evidence)))
                    agg_inputs[k]["hit"].append(rm.hit_at_k(ranked, k))
                    agg_inputs[k]["rr"].append(rm.reciprocal_rank(ranked, k))
                    agg_inputs[k]["ndcg"].append(rm.ndcg_at_k(ranked, corpus_grades, k))
                    agg_inputs[k]["prec"].append(rm.context_precision_at_k(ranked, k))
                pk = primary_k
                row.update(hit=rm.hit_at_k(ranked, pk), rr=round(rm.reciprocal_rank(ranked, pk), 4),
                           recall=round(rm.recall_at_k(ranked, total_rel, pk), 4),
                           evidence_recall=round(rm.evidence_recall(evidence_found([_as_chunk(c) for c in r.chunks[:pk]], q.evidence)), 4),
                           ndcg=round(rm.ndcg_at_k(ranked, corpus_grades, pk), 4),
                           context_precision=round(rm.context_precision_at_k(ranked, pk), 4))
        per_query.append(row)
    by_k = {}
    for k, d in agg_inputs.items():
        by_k[f"k={k}"] = {"recall_at_k_chunk": rm.mean(d["recall"]), "evidence_recall_at_k": rm.mean(d["ev_recall"]),
                          "hit_rate_at_k": rm.mean(d["hit"]), "mrr_at_k": rm.mean(d["rr"]), "ndcg_at_k": rm.mean(d["ndcg"]),
                          "context_precision_at_k": rm.mean(d["prec"]), "n_eligible_queries": len(d["hit"])}
    return {"per_query": per_query, "by_k": by_k, "excluded_queries": excluded, "calib_inputs": scores_for_calib,
            "n_chunks_in_store": len(all_chunks)}


# --------------------------------------------------------------------------- answers
def run_answer_eval(svc: Services, questions: list[Question], k: int, judge: LLMJudge | None) -> dict:
    s = svc.settings
    if not s.groq_api_key:
        return {"status": "not_run", "reason": "GROQ_API_KEY is not set; generation and LLM-as-judge need the Groq API. "
                "Re-run `python -m app.cli evaluate` with a key to populate faithfulness, relevance, EM and F1.",
                "timestamp": now_iso()}
    all_chunks = svc.store.all_chunks()
    rows, f_scores, r_scores, em_s, f1_s, cont_s, cite_prec = [], [], [], [], [], [], []
    abst_tp = abst_fp = abst_fn = abst_tn = 0
    ptok = ctok = 0
    for q in questions:
        res = svc.answers.answer(q.question, k=k, filters=q.filters, request_id=f"eval-{q.id}")
        answered = res.status == STATUS_ANSWERED
        row: dict[str, Any] = {"id": q.id, "type": q.type, "answerable": q.answerable, "status": res.status,
                               "reason": res.reason, "answer": res.answer, "citations": " ".join(c.chunk_id for c in res.citations),
                               "invalid_citations": " ".join(res.invalid_citations), "retrieval_ms": round(res.retrieval_ms, 1),
                               "generation_ms": None if res.generation_ms is None else round(res.generation_ms, 1),
                               "prompt_tokens": res.token_usage["prompt_tokens"], "completion_tokens": res.token_usage["completion_tokens"]}
        ptok += res.token_usage["prompt_tokens"] or 0
        ctok += res.token_usage["completion_tokens"] or 0
        if q.answerable and answered:
            abst_tp += 1
        elif q.answerable and not answered:
            abst_fn += 1       # answerable but system abstained / failed
        elif (not q.answerable) and answered:
            abst_fp += 1       # unanswerable but system answered (hallucination risk)
        else:
            abst_tn += 1       # correctly abstained
        if answered:
            ctx = [c.text for c in res.retrieval.chunks[:res.context_chunks_used]]
            if judge is not None:
                jr = judge.judge(q.question, res.answer, ctx)
                row.update(judge_available=jr.available, faithfulness=jr.faithfulness, relevance=jr.relevance,
                           unsupported_claims=" | ".join(jr.unsupported_claims), judge_rationale=jr.rationale, judge_error=jr.error)
                if jr.available:
                    f_scores.append(jr.faithfulness)
                    r_scores.append(jr.relevance)
        if q.answerable and q.gold_answers:
            row.update(exact_match=am.exact_match(res.answer, q.gold_answers), f1=round(am.token_f1(res.answer, q.gold_answers), 4),
                       contains_gold=am.contains_gold(res.answer, q.gold_answers))
            em_s.append(row["exact_match"]); f1_s.append(row["f1"]); cont_s.append(row["contains_gold"])
        if answered and q.answerable:
            good = [chunk_grade({"text": next((c.text for c in res.retrieval.chunks if c.chunk_id == ci.chunk_id), ""),
                                 "metadata": {"source_id": ci.source_id, "page": ci.page if ci.page is not None else -1}}, q.evidence) >= 1
                    for ci in res.citations]
            row["citation_precision"] = round(sum(good) / len(good), 4) if good else 0.0
            cite_prec.append(row["citation_precision"])
        rows.append(row)
    n_gen_err = sum(1 for r in rows if r["status"] == "generation_error")
    return {"status": "completed" if n_gen_err < len(rows) else "failed", "timestamp": now_iso(), "generator_model": s.groq_model,
            "judge_model": judge.model if judge else None,
            "judge_same_model_as_generator_warning": bool(judge and judge.same_as_generator), "k": k, "n_questions": len(rows),
            "generation_errors": n_gen_err,
            "aggregate": {"faithfulness_mean": rm.mean(f_scores), "answer_relevance_mean": rm.mean(r_scores),
                          "n_judged": len(f_scores), "exact_match": rm.mean(em_s), "f1": rm.mean(f1_s),
                          "contains_gold_lenient": rm.mean(cont_s), "n_with_gold": len(em_s),
                          "citation_precision_mean": rm.mean(cite_prec),
                          "answered_when_answerable": abst_tp, "abstained_when_answerable": abst_fn,
                          "answered_when_unanswerable": abst_fp, "abstained_when_unanswerable": abst_tn,
                          "total_prompt_tokens": ptok, "total_completion_tokens": ctok},
            "judge_rubric": "1-5 integer scores normalised to 0..1 via (s-1)/4; see app/evaluation/judge.py",
            "per_question": rows, "all_chunks_in_store": len(all_chunks)}


# --------------------------------------------------------------------------- chunking experiment
def run_chunking_experiment(settings: Settings, questions: list[Question], configs: list[tuple[int, int]], k: int) -> dict:
    results = []
    for size, ov in configs:
        tmp = Path(tempfile.mkdtemp(prefix="chunkexp_"))
        s = dataclasses.replace(settings, chunk_size=size, chunk_overlap=ov)
        svc = build_eval_services(s, tmp, collection=f"exp_{size}_{ov}")
        try:
            ret = run_retrieval_eval(svc, questions, k)
            qs = [{"id": q.id, "question": q.question, "filters": q.filters} for q in questions]
            bench = run_retrieval_benchmark(svc.retriever, qs, k=k, warmup=2, repeats=3)
            m = ret["by_k"][f"k={k}"]
            results.append({"chunk_size": size, "chunk_overlap": ov, "chunks": ret["n_chunks_in_store"],
                            "metrics_at_k": m, "excluded_queries": ret["excluded_queries"],
                            "retrieval_p50_ms": bench["retrieval"]["p50_ms"], "retrieval_p95_ms": bench["retrieval"]["p95_ms"],
                            "avg_words_per_chunk": round(sum(c["metadata"]["token_count"] for c in svc.store.all_chunks()) / max(ret["n_chunks_in_store"], 1), 1)})
        finally:
            svc.close()
            shutil.rmtree(tmp, ignore_errors=True)
    return {"timestamp": now_iso(), "k": k, "embedding_backend": settings.embedding_backend, "embedding_model_effective": svc.encoder.name,
            "dataset_questions": len(questions), "results": results,
            "caveats": ["Only 25-ish chunks exist at the default size, so differences between configs rest on a handful of queries.",
                        "Word-based chunk sizes; the 400-word config would exceed MiniLM's 256 word-piece input limit (silent truncation) with the real "
                        "model, an effect a hashed bag-of-words encoder cannot show.",
                        "Not evidence that one configuration is universally best."]}


# --------------------------------------------------------------------------- orchestration
def run_all(settings: Settings, k: int | None = None, with_chunking: bool = True, with_latency: bool = True,
            with_answers: bool = True, keep_store: bool = False) -> dict:
    k = k or settings.top_k
    out = settings.output_dir
    out.mkdir(parents=True, exist_ok=True)
    questions = load_dataset(settings.dataset_dir)
    work = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    svc = build_eval_services(settings, work)
    meta = {"timestamp": now_iso(), "dataset": str(settings.dataset_dir.name), "n_questions": len(questions),
            "n_answerable": sum(q.answerable for q in questions), "embedding_backend": svc.encoder.backend,
            "embedding_model": svc.encoder.name, "embedding_dim": svc.encoder.dim, "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap, "top_k": k, "min_relevance_score": settings.min_relevance_score,
            "vector_store": "ChromaDB (persistent, cosine)", "environment": environment_info()}
    try:
        ret = run_retrieval_eval(svc, questions, k)
        calib = calibrate([s for _, s, _ in ret["calib_inputs"]], [a for _, _, a in ret["calib_inputs"]])
        # abstention gate performance at the CONFIGURED threshold
        gate = {"threshold": settings.min_relevance_score,
                "answerable_passed": sum(1 for r in ret["per_query"] if r["answerable"] and not r["gate_abstains_at_threshold"]),
                "answerable_blocked": sum(1 for r in ret["per_query"] if r["answerable"] and r["gate_abstains_at_threshold"]),
                "unanswerable_passed_to_llm": sum(1 for r in ret["per_query"] if not r["answerable"] and not r["gate_abstains_at_threshold"]),
                "unanswerable_blocked": sum(1 for r in ret["per_query"] if not r["answerable"] and r["gate_abstains_at_threshold"])}
        dump_json(out / "retrieval_metrics.json", {"meta": meta, "primary_k": k, "by_k": ret["by_k"], "excluded_queries": ret["excluded_queries"],
                  "abstention_gate_at_configured_threshold": gate, "threshold_calibration": calib,
                  "metric_definitions": "see app/evaluation/retrieval_metrics.py and docs/EVALUATION_METHODOLOGY.md",
                  "label_caveat": "relevance labels are author-written evidence phrases, not independently validated ground truth"})
        dump_csv(out / "retrieval_per_query.csv", ret["per_query"])

        answers = {"status": "not_run", "reason": "skipped by flag"}
        if with_answers:
            judge = LLMJudge(settings, svc.llm) if settings.groq_api_key else None
            answers = run_answer_eval(svc, questions, k, judge)
        dump_json(out / "answer_quality.json", {"meta": meta, **{kk: v for kk, v in answers.items() if kk != "per_question"}})
        dump_csv(out / "answer_quality_per_query.csv", answers.get("per_question", []))

        latency = None
        if with_latency:
            qs = [{"id": q.id, "question": q.question, "filters": q.filters} for q in questions]
            latency = run_retrieval_benchmark(svc.retriever, qs, k=k, warmup=settings.benchmark_warmup, repeats=settings.benchmark_repeats)
            samples = latency.pop("samples")
            dump_json(out / "latency_results.json", {"meta": {**meta, "vector_count": svc.store.count()}, **latency,
                      "method": "time.perf_counter; single thread; warm-up passes untimed; cold start reported separately; percentiles use linear interpolation",
                      "caveat": "Tiny local corpus on one machine; says nothing about production-scale latency."})
            dump_csv(out / "latency_samples.csv", samples)
        chunking = None
        if with_chunking:
            chunking = run_chunking_experiment(settings, questions, [(80, 15), (180, 30), (400, 60)], k)
            dump_json(out / "chunking_experiment.json", chunking)
    finally:
        svc.close()
        if not keep_store:
            shutil.rmtree(work, ignore_errors=True)
    summary = write_summary(out, meta, ret, calib, gate, answers, latency, chunking)
    return {"meta": meta, "retrieval": ret["by_k"], "calibration": calib, "answers_status": answers.get("status"),
            "latency": latency and {k2: latency[k2] for k2 in ("retrieval", "cold_start_first_query_ms")}, "summary_path": str(summary)}


def _f(x, nd=3):
    return "N/A" if x is None else f"{x:.{nd}f}"


def write_summary(out: Path, meta, ret, calib, gate, answers, latency, chunking) -> Path:
    k = meta["top_k"]
    m = ret["by_k"][f"k={k}"]
    L = [f"# Evaluation summary", "", f"Generated {meta['timestamp']} | dataset: {meta['n_questions']} questions "
         f"({meta['n_answerable']} answerable) | embedding: **{meta['embedding_backend']} / {meta['embedding_model']}** ({meta['embedding_dim']}-d) | "
         f"chunks: {meta['chunk_size']} words / {meta['chunk_overlap']} overlap | k = {k}", ""]
    if meta["embedding_backend"] == "hash":
        L += ["> **WARNING:** these numbers were produced with the offline `hash` lexical embedder, NOT all-MiniLM-L6-v2. "
              "They validate the pipeline and metric code; they must not be reported as MiniLM results. Re-run with "
              "`EMBEDDING_BACKEND=sentence-transformers` for the real figures.", ""]
    L += ["## Stage status", "", "| Stage | Status |", "|---|---|", "| Retrieval metrics | completed |",
          f"| Abstention calibration | {calib.get('status')} |",
          f"| Answer quality (faithfulness / relevance / EM / F1) | {answers.get('status')}" + (f" - {answers.get('reason')}" if answers.get("reason") else "") + " |",
          f"| Latency benchmark | {'completed' if latency else 'not run'} |", f"| Chunking experiment | {'completed' if chunking else 'not run'} |",
          "| Cost comparison | see cost_comparison.md |", "", f"## Retrieval (k = {k}, {m['n_eligible_queries']} eligible queries)", "",
          "| Metric | Value |", "|---|---|", f"| Hit Rate@{k} | {_f(m['hit_rate_at_k'])} |", f"| Recall@{k} (chunk-level) | {_f(m['recall_at_k_chunk'])} |",
          f"| Evidence recall@{k} | {_f(m['evidence_recall_at_k'])} |", f"| MRR@{k} | {_f(m['mrr_at_k'])} |", f"| nDCG@{k} | {_f(m['ndcg_at_k'])} |",
          f"| Context precision@{k} | {_f(m['context_precision_at_k'])} |", "", "All k:", "", "| k | Hit | Recall | EvRecall | MRR | nDCG | CtxPrec |", "|---|---|---|---|---|---|---|"]
    for kk, d in ret["by_k"].items():
        L.append(f"| {kk[2:]} | {_f(d['hit_rate_at_k'])} | {_f(d['recall_at_k_chunk'])} | {_f(d['evidence_recall_at_k'])} | {_f(d['mrr_at_k'])} | {_f(d['ndcg_at_k'])} | {_f(d['context_precision_at_k'])} |")
    L += ["", "## Abstention", "", f"Configured threshold {gate['threshold']}: answerable passed {gate['answerable_passed']}, answerable wrongly blocked {gate['answerable_blocked']}, "
          f"unanswerable blocked {gate['unanswerable_blocked']}, unanswerable passed to the LLM {gate['unanswerable_passed_to_llm']} (the LLM prompt is the second line of defence).", ""]
    if calib.get("status") == "ok":
        L += [f"Calibration suggests threshold **{calib['recommended_min_relevance_score']}** (in-sample balanced accuracy {calib['balanced_accuracy_in_sample']}, "
              f"leave-one-out accuracy {calib['loo_accuracy']}). {calib['caveat']}", ""]
    if answers.get("status") == "completed":
        a = answers["aggregate"]
        L += ["## Answer quality", "", f"Generator `{answers['generator_model']}`, judge `{answers['judge_model']}`."
              + (" **Warning: judge == generator model.**" if answers.get("judge_same_model_as_generator_warning") else ""), "",
              "| Metric | Value |", "|---|---|", f"| Faithfulness (judge, 0-1) | {_f(a['faithfulness_mean'])} (n={a['n_judged']}) |",
              f"| Answer relevance (judge, 0-1) | {_f(a['answer_relevance_mean'])} |", f"| Exact match | {_f(a['exact_match'])} |", f"| Token F1 | {_f(a['f1'])} |",
              f"| Contains-gold (lenient) | {_f(a['contains_gold_lenient'])} |", f"| Citation precision | {_f(a['citation_precision_mean'])} |", ""]
    if latency:
        r = latency["retrieval"]
        L += ["## Retrieval latency", "", f"p50 **{r['p50_ms']} ms**, p95 **{r['p95_ms']} ms** over {r['n']} samples (k={k}); cold start {latency['cold_start_first_query_ms']} ms. "
              f"Embedding p50 {latency['embed']['p50_ms']} ms, vector search p50 {latency['vector_search']['p50_ms']} ms. " + ("" if r["p95_reliable"] else "p95 is based on < 100 samples and is unreliable."), ""]
    if chunking:
        L += ["## Chunking experiment", "", "| size/overlap | chunks | Hit | MRR | nDCG | CtxPrec | p50 ms |", "|---|---|---|---|---|---|---|"]
        for r in chunking["results"]:
            mm = r["metrics_at_k"]
            L.append(f"| {r['chunk_size']}/{r['chunk_overlap']} | {r['chunks']} | {_f(mm['hit_rate_at_k'])} | {_f(mm['mrr_at_k'])} | {_f(mm['ndcg_at_k'])} | {_f(mm['context_precision_at_k'])} | {r['retrieval_p50_ms']} |")
        L += [""] + [f"- {c}" for c in chunking["caveats"]]
    p = out / "evaluation_summary.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p
