"""Latency statistics + benchmark runner (monotonic high-resolution timers)."""
from __future__ import annotations

import math
import os
import platform
import sys
import time
from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (identical to numpy's default ``method='linear'``)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not (0 <= p <= 100):
        raise ValueError("p must be within [0, 100]")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def summarize(values: Sequence[float]) -> dict:
    n = len(values)
    return {"n": n, "p50_ms": round(percentile(values, 50), 3), "p95_ms": round(percentile(values, 95), 3),
            "mean_ms": round(sum(values) / n, 3), "min_ms": round(min(values), 3), "max_ms": round(max(values), 3),
            "p95_reliable": n >= 100}


def environment_info() -> dict:
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            mem_gb = round(int(f.readline().split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "ram_gb": mem_gb}


def run_retrieval_benchmark(retriever, queries: list[dict], k: int, warmup: int, repeats: int) -> dict:
    """Measure retrieval latency. ``queries``: [{id, question, filters}].

    * cold start  = encoder load + first query, measured once, reported separately;
    * warm-up     = ``warmup`` untimed passes over the first queries (fills OS/page/HNSW caches);
    * steady state= ``repeats`` sequential passes over ALL queries; single thread, no concurrency;
                    each sample splits query-embedding time and vector-search time.
    """
    t0 = time.perf_counter()
    retriever.retrieve(queries[0]["question"], k=k, filters=queries[0].get("filters"))
    cold_ms = (time.perf_counter() - t0) * 1000
    for i in range(warmup):
        q = queries[i % len(queries)]
        retriever.retrieve(q["question"], k=k, filters=q.get("filters"))
    samples = []
    for rep in range(repeats):
        for q in queries:
            r = retriever.retrieve(q["question"], k=k, filters=q.get("filters"))
            samples.append({"query_id": q["id"], "repeat": rep, "embed_ms": r.embed_ms,
                            "search_ms": r.search_ms, "retrieval_ms": r.total_ms,
                            "chunks": len(r.chunks)})
    tot = [s["retrieval_ms"] for s in samples]
    return {"cold_start_first_query_ms": round(cold_ms, 2), "k": k, "warmup_runs": warmup, "repeats": repeats,
            "queries": len(queries), "retrieval": summarize(tot),
            "embed": summarize([s["embed_ms"] for s in samples]),
            "vector_search": summarize([s["search_ms"] for s in samples]), "samples": samples}
