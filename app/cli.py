"""Command-line interface:  python -m app.cli <command> [options]"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from app.config import Settings, resolve_path
from app.exceptions import RAGError
from app.logging_config import setup_logging


def _settings(args) -> Settings:
    s = Settings.from_env()
    over = {}
    for a, f in (("chunk_size", "chunk_size"), ("chunk_overlap", "chunk_overlap"), ("k", "top_k")):
        v = getattr(args, a, None)
        if v is not None:
            over[f] = v
    if over:
        s = dataclasses.replace(s, **over)
        s.validate()
    return s


def _filters(args) -> dict | None:
    f = {}
    for name in ("category", "document_type", "source_id"):
        v = getattr(args, name, None)
        if v:
            f[name] = v
    return f or None


def cmd_ingest(args) -> int:
    from app.services import Services
    s = _settings(args)
    svc = Services(s)
    before = svc.store.count()
    print(f"Vector count BEFORE ingest: {before}")
    rep = svc.pipeline.ingest_path(resolve_path(args.path), prune=args.prune, source_prefix=args.source_prefix)
    for r in rep.results:
        extra = f"  [{r.error or r.warning}]" if (r.error or r.warning) else ""
        print(f"  {r.status:<9} {r.source_id:<45} chunks={r.chunks}{extra}")
    print(f"files: discovered={rep.files_discovered} new={rep.files_new} updated={rep.files_updated} "
          f"unchanged={rep.files_unchanged} failed={rep.files_failed} skipped={rep.files_skipped}")
    print(f"chunks upserted={rep.chunks_upserted} deleted={rep.chunks_deleted} | model={rep.embedding_model} | "
          f"chunk_size={s.chunk_size} overlap={s.chunk_overlap} | {rep.duration_ms} ms")
    print(f"Vector count AFTER ingest:  {rep.count_after}  (delta {rep.count_after - before:+d})")
    svc.close()
    return 1 if rep.files_failed else 0


def cmd_query(args) -> int:
    from app.services import Services
    s = _settings(args)
    svc = Services(s)
    filters = _filters(args)
    if args.retrieve_only:
        r = svc.retriever.retrieve(args.question, k=args.k, filters=filters)
        print(f"retrieval_ms={r.total_ms:.1f} (embed {r.embed_ms:.1f} + search {r.search_ms:.1f}) chunks={len(r.chunks)} "
              f"top_score={r.decision.top_score} relevant={r.decision.relevant} ({r.decision.reason})")
        for c in r.chunks:
            print(f"  #{c.rank} {c.score:.3f} {c.chunk_id} {c.source_id} p={c.page} :: {' '.join(c.text.split())[:90]}")
        svc.close()
        return 0
    res = svc.answers.answer(args.question, k=args.k, filters=filters)
    if args.json:
        print(json.dumps({"status": res.status, "answer": res.answer, "citations": [c.to_dict() for c in res.citations],
                          "token_usage": res.token_usage}, indent=2))
    else:
        print(f"Q: {args.question}\nstatus: {res.status}" + (f" ({res.reason})" if res.reason else ""))
        print(f"A: {res.answer}")
        for c in res.citations:
            loc = f", page {c.page}" if c.page else ""
            print(f"  [{c.ref}] {c.source_name}{loc} | chunk {c.chunk_id} | score {c.score:.3f}")
        if res.invalid_citations:
            print(f"  invalid citations rejected: {res.invalid_citations}")
        if res.error_message:
            print(f"error[{res.error_category}]: {res.error_message}")
        gen = "n/a" if res.generation_ms is None else f"{res.generation_ms:.1f} ms"
        tu = res.token_usage
        toks = f"prompt={tu['prompt_tokens']} completion={tu['completion_tokens']} total={tu['total_tokens']}" if tu["available"] else "unavailable (no LLM call / no usage in response)"
        print(f"latency: retrieval={res.retrieval_ms:.1f} ms | generation={gen} | end-to-end={res.total_ms:.1f} ms")
        print(f"retrieved_chunks={len(res.retrieval.chunks)} (used in prompt: {res.context_chunks_used}) | tokens: {toks} | model={res.model}")
    svc.close()
    return 0 if res.status != "generation_error" else 2


def cmd_stats(args) -> int:
    from app.services import Services
    svc = Services(_settings(args))
    print(json.dumps({"vector_count": svc.store.count(), "sources": svc.store.source_summary(), "health": svc.store.health()}, indent=2))
    svc.close()
    return 0


def cmd_evaluate(args) -> int:
    from app.evaluation.run_evaluation import run_all
    s = _settings(args)
    r = run_all(s, k=args.k, with_chunking=not args.skip_chunking, with_latency=not args.skip_latency, with_answers=not args.skip_answers)
    from app.evaluation.cost_analysis import build_comparison, load_assumptions, write_outputs
    a = load_assumptions(resolve_path("data/cost/assumptions.json"))
    write_outputs(a, build_comparison(a), s.output_dir)
    print(json.dumps({k: v for k, v in r.items() if k != "retrieval"}, indent=2, default=str))
    print(f"Summary written to {r['summary_path']}")
    return 0


def cmd_cost(args) -> int:
    from app.evaluation.cost_analysis import build_comparison, load_assumptions, write_outputs
    s = _settings(args)
    a = load_assumptions(resolve_path(args.assumptions))
    paths = write_outputs(a, build_comparison(a), s.output_dir)
    print("wrote:", *paths.values(), sep="\n  ")
    return 0


def cmd_benchmark(args) -> int:
    from app.evaluation.dataset import load_dataset
    from app.evaluation.latency import run_retrieval_benchmark
    from app.evaluation.run_evaluation import build_eval_services, dump_csv, dump_json
    import tempfile, shutil
    s = _settings(args)
    qs = [{"id": q.id, "question": q.question, "filters": q.filters} for q in load_dataset(s.dataset_dir)]
    tmp = Path(tempfile.mkdtemp(prefix="bench_"))
    svc = build_eval_services(s, tmp)
    try:
        res = run_retrieval_benchmark(svc.retriever, qs, k=s.top_k, warmup=s.benchmark_warmup, repeats=s.benchmark_repeats)
        samples = res.pop("samples")
        dump_json(s.output_dir / "latency_results.json", {"vector_count": svc.store.count(), **res})
        dump_csv(s.output_dir / "latency_samples.csv", samples)
        print(json.dumps({k: res[k] for k in ("cold_start_first_query_ms", "retrieval", "embed", "vector_search")}, indent=2))
    finally:
        svc.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def cmd_calibrate(args) -> int:
    from app.evaluation.calibration import calibrate
    from app.evaluation.dataset import load_dataset
    from app.evaluation.run_evaluation import build_eval_services, run_retrieval_eval
    import tempfile, shutil
    s = _settings(args)
    tmp = Path(tempfile.mkdtemp(prefix="calib_"))
    svc = build_eval_services(s, tmp)
    try:
        ret = run_retrieval_eval(svc, load_dataset(s.dataset_dir), s.top_k)
        print(json.dumps(calibrate([x for _, x, _ in ret["calib_inputs"]], [a for _, _, a in ret["calib_inputs"]]), indent=2))
    finally:
        svc.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def cmd_verify(args) -> int:
    from app.embeddings.encoder import get_encoder
    s = Settings.from_env(validate=False)
    ok = True
    try:
        s.validate()
        print("[ok] configuration valid")
    except RAGError as e:
        print("[FAIL]", e.message); ok = False
    try:
        enc = get_encoder(s)
        print(f"[ok] embedding {enc.backend}:{enc.name} dimension = {enc.verify_dimension()} (expected {s.embedding_dim})")
    except RAGError as e:
        print("[FAIL] embedding:", e.message); ok = False
    print("[ok] GROQ_API_KEY configured" if s.groq_api_key else "[warn] GROQ_API_KEY not set (generation/judge unavailable)")
    if s.groq_api_key and args.list_models:
        try:
            import groq
            ids = sorted(m.id for m in groq.Groq(api_key=s.groq_api_key, timeout=s.groq_timeout).models.list().data)
            print("available models:", *ids, sep="\n  ")
            for name, val in (("GROQ_MODEL", s.groq_model), ("JUDGE_MODEL", s.judge_model)):
                print(f"[{'ok' if val in ids else 'FAIL'}] {name}={val}")
                ok &= val in ids
        except Exception as e:
            print(f"[FAIL] Groq model listing failed ({type(e).__name__})"); ok = False
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.cli", description="Cost-efficient RAG CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="ingest a folder/file of PDF/HTML/MD")
    i.add_argument("--path", default="data/sample_documents")
    i.add_argument("--chunk-size", type=int, dest="chunk_size"); i.add_argument("--chunk-overlap", type=int, dest="chunk_overlap")
    i.add_argument("--prune", action="store_true", help="delete chunks of sources no longer on disk")
    i.add_argument("--source-prefix", default=""); i.set_defaults(fn=cmd_ingest)
    q = sub.add_parser("query", help="ask a question")
    q.add_argument("question"); q.add_argument("-k", type=int)
    q.add_argument("--category"); q.add_argument("--document-type", dest="document_type"); q.add_argument("--source-id", dest="source_id")
    q.add_argument("--retrieve-only", action="store_true"); q.add_argument("--json", action="store_true"); q.set_defaults(fn=cmd_query)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    e = sub.add_parser("evaluate", help="run the full evaluation harness")
    e.add_argument("-k", type=int); e.add_argument("--skip-chunking", action="store_true")
    e.add_argument("--skip-latency", action="store_true"); e.add_argument("--skip-answers", action="store_true"); e.set_defaults(fn=cmd_evaluate)
    b = sub.add_parser("benchmark", help="retrieval latency benchmark"); b.add_argument("-k", type=int); b.set_defaults(fn=cmd_benchmark)
    c = sub.add_parser("cost", help="cost comparison tables"); c.add_argument("--assumptions", default="data/cost/assumptions.json"); c.set_defaults(fn=cmd_cost)
    k = sub.add_parser("calibrate", help="calibrate abstention threshold"); k.add_argument("-k", type=int); k.set_defaults(fn=cmd_calibrate)
    v = sub.add_parser("verify", help="check config, embedding dimension, Groq key"); v.add_argument("--list-models", action="store_true"); v.set_defaults(fn=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        setup_logging(Settings.from_env(validate=False).log_level if args.cmd != "query" else "WARNING")
        return args.fn(args)
    except RAGError as exc:
        print(f"error[{exc.code}]: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
