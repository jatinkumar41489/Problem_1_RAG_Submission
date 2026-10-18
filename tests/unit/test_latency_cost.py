import copy
from pathlib import Path

import numpy as np
import pytest

from app.config import BASE_DIR
from app.evaluation import cost_analysis as ca
from app.evaluation.latency import percentile, summarize

A = ca.load_assumptions(BASE_DIR / "data" / "cost" / "assumptions.json")


@pytest.mark.parametrize("p", [0, 25, 50, 90, 95, 99, 100])
def test_percentile_matches_numpy(p):
    data = list(np.random.default_rng(1).random(57) * 10)
    assert percentile(data, p) == pytest.approx(float(np.percentile(data, p)))


def test_percentile_edge_cases():
    assert percentile([4.0], 95) == 4.0
    with pytest.raises(ValueError):
        percentile([], 50)
    assert summarize([1.0] * 10)["p95_reliable"] is False


def test_vector_and_ram_formulas_hand_calculated():
    sh = ca.self_hosted_cost(A, 1_000_000)
    assert sh["vector_bytes"] == 1_000_000 * 384 * 4
    expected_ram = 2.0 + 1_000_000 * 384 * 4 * 1.5 * 1.3 / 2 ** 30
    assert sh["required_ram_gib"] == pytest.approx(expected_ram)
    assert sh["instance"] == "m6i.large"
    assert sh["compute_usd"] == pytest.approx(0.096 * 730)


def test_managed_minimum_and_usage():
    m = ca.managed_cost(A, 100_000)
    assert m["total_usd"] == 50 and m["storage_gb"] == pytest.approx(100_000 * (1536 + 1550) / 1e9)
    big = ca.managed_cost(A, 10_000_000)
    gb = 10_000_000 * (1536 + 1550) / 1e9
    reads = 300_000 * gb / 1e6 * 16.0
    assert big["reads_usd"] == pytest.approx(reads) and big["total_usd"] > 50


def test_break_even_consistent_with_totals():
    be = ca.break_even_queries(A, 1_000_000)
    sh = ca.self_hosted_cost(A, 1_000_000)
    assert ca.managed_total_at(A, 1_000_000, be) == pytest.approx(sh["infra_usd"], rel=1e-6)


def test_assumptions_change_results():
    b = copy.deepcopy(A)
    b["workload"]["queries_per_month"] = 30_000_000
    assert ca.managed_cost(b, 10_000_000)["total_usd"] > ca.managed_cost(A, 10_000_000)["total_usd"] * 10


def test_outputs_written(tmp_path):
    paths = ca.write_outputs(A, ca.build_comparison(A), tmp_path)
    assert all(Path(p).exists() for p in paths.values())
    assert "100K" in Path(paths["md"]).read_text() and "10M" in Path(paths["md"]).read_text()
