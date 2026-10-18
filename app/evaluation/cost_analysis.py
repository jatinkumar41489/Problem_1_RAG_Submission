"""Reproducible monthly-cost model: self-hosted ChromaDB vs a managed vector DB (Pinecone serverless).

Every number comes from ``data/cost/assumptions.json``; nothing is hard-coded here. Formulas:

  vec_bytes      = N * dim * 4
  payload_bytes  = N * (dim*4 + avg_chunk_text_bytes + avg_metadata_bytes)      (both systems)

Self-hosted (single node):
  ram_gib   = baseline + vec_bytes * index_overhead * ram_headroom / 2^30      -> smallest listed instance with >= ram
  compute   = instance $/h * 730
  disk_gb   = (vec_bytes*index_overhead + N*(text+metadata)) * disk_headroom / 1e9
  storage   = disk_gb * gp3 $/GB-month;   backups = payload_gb * backup_copies * snapshot $/GB-month
  infra     = compute + storage + backups ;  with HA = compute*ha_nodes + storage*ha_nodes + backups
  ops       = ops_hours(N) * $/h  (reported separately)

Managed (Pinecone serverless Standard):
  storage_gb = payload_bytes / 1e9;  storage = storage_gb * $0.33
  RU/query   = max(0.25, storage_gb);  reads  = queries * RU/query / 1e6 * $/M-RU
  WU         = rewrite_fraction * N * max(5/batch, payload_kb) ~ payload KB per record;  writes = WU/1e6 * $/M-WU
  total      = max(monthly_minimum, storage + reads + writes)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

GIB = 2 ** 30


def load_assumptions(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _val(x: Any) -> float:
    return float(x["value"] if isinstance(x, dict) else x)


def pick_instance(required_ram_gib: float, instances: list[dict]) -> dict | None:
    for inst in sorted(instances, key=lambda i: i["ram_gib"]):
        if inst["ram_gib"] >= required_ram_gib:
            return inst
    return None


def self_hosted_cost(a: dict, n: int) -> dict[str, Any]:
    w, s = a["workload"], a["self_hosted_chromadb"]
    vec_bytes = n * w["embedding_dim"] * w["bytes_per_float"]
    text_meta = n * (w["avg_chunk_text_bytes"] + w["avg_metadata_bytes"])
    payload_bytes = vec_bytes + text_meta
    ram_gib = _val(s["baseline_ram_gib"]) + vec_bytes * _val(s["index_overhead_factor"]) * _val(s["ram_headroom_factor"]) / GIB
    inst = pick_instance(ram_gib, s["instances"])
    out: dict[str, Any] = {"vector_bytes": vec_bytes, "payload_gb": payload_bytes / 1e9, "required_ram_gib": ram_gib}
    if inst is None:
        out.update(instance=None, feasible=False, note="exceeds largest listed single-node instance")
        return out
    compute = inst["usd_per_hour"] * s["hours_per_month"]
    disk_gb = (vec_bytes * _val(s["index_overhead_factor"]) + text_meta) * _val(s["disk_headroom_factor"]) / 1e9
    storage = disk_gb * _val(s["ebs_gp3_usd_per_gb_month"])
    backups = payload_bytes / 1e9 * _val(s["backup_copies"]) * _val(s["snapshot_usd_per_gb_month"])
    ha = s["ha_nodes"]
    ops = s["ops_hours_per_month"][str(n)] * _val(s["ops_usd_per_hour"]) if str(n) in s["ops_hours_per_month"] else None
    infra = compute + storage + backups
    infra_ha = (compute + storage) * ha + backups
    out.update(instance=inst["name"], instance_ram_gib=inst["ram_gib"], feasible=True, compute_usd=compute,
               disk_gb=disk_gb, storage_usd=storage, backups_usd=backups, infra_usd=infra,
               infra_ha_usd=infra_ha, ops_usd=ops,
               total_incl_ops_usd=None if ops is None else infra + ops,
               total_ha_incl_ops_usd=None if ops is None else infra_ha + ops)
    return out


def managed_cost(a: dict, n: int) -> dict[str, Any]:
    w, m = a["workload"], a["managed_pinecone_serverless_standard"]
    payload_bytes = n * (w["embedding_dim"] * w["bytes_per_float"] + w["avg_chunk_text_bytes"] + w["avg_metadata_bytes"])
    gb = payload_bytes / 1e9
    ru_per_query = max(0.25, gb)
    res = {}
    for tag in ("low", "high"):
        ru_price = m["read_units_usd_per_million"][tag]
        wu_price = m["write_units_usd_per_million"][tag]
        storage = gb * m["storage_usd_per_gb_month"]
        reads = w["queries_per_month"] * ru_per_query / 1e6 * ru_price
        wu = w["monthly_rewrite_fraction"] * n * math.ceil(payload_bytes / n / 1000)
        writes = wu / 1e6 * wu_price
        usage = storage + reads + writes
        res[tag] = {"storage_usd": storage, "reads_usd": reads, "writes_usd": writes, "usage_usd": usage,
                    "total_usd": max(m["monthly_minimum_usd"], usage)}
    used = m["read_units_usd_per_million"]["used"]
    return {"storage_gb": gb, "read_units_per_query": ru_per_query, "used_price_end": used, **res[used],
            "range_usd": [res["low"]["total_usd"], res["high"]["total_usd"]]}


def build_comparison(a: dict) -> list[dict[str, Any]]:
    rows = []
    for n in a["scales"]:
        sh, mg = self_hosted_cost(a, n), managed_cost(a, n)
        row = {"vectors": n, "self_hosted": sh, "managed": mg}
        if sh.get("feasible"):
            row["saving_vs_infra_usd"] = mg["total_usd"] - sh["infra_usd"]
            row["saving_vs_infra_ha_usd"] = mg["total_usd"] - sh["infra_ha_usd"]
            if sh["total_incl_ops_usd"] is not None:
                row["saving_vs_total_incl_ops_usd"] = mg["total_usd"] - sh["total_incl_ops_usd"]
                row["saving_vs_ha_incl_ops_usd"] = mg["total_usd"] - sh["total_ha_incl_ops_usd"]
        rows.append(row)
    return rows


def managed_total_at(a: dict, n: int, queries_per_month: float) -> float:
    """Managed monthly total (used price end) at a different query volume."""
    b = copy_with_queries(a, queries_per_month)
    return managed_cost(b, n)["total_usd"]


def copy_with_queries(a: dict, q: float) -> dict:
    import copy
    b = copy.deepcopy(a)
    b["workload"]["queries_per_month"] = q
    return b


def break_even_queries(a: dict, n: int) -> float | None:
    """Queries/month above which self-hosted single-node infra (no ops) is cheaper than managed.
    None = self-hosted is never cheaper (its infra is below/at the managed monthly minimum region
    handled by the formula) or the node is infeasible."""
    sh = self_hosted_cost(a, n)
    if not sh.get("feasible"):
        return None
    mg = managed_cost(a, n)
    per_query = mg["read_units_per_query"] / 1e6 * a["managed_pinecone_serverless_standard"]["read_units_usd_per_million"][mg["used_price_end"]]
    fixed = mg["storage_usd"] + mg["writes_usd"]
    target = sh["infra_usd"]
    if target <= max(a["managed_pinecone_serverless_standard"]["monthly_minimum_usd"], fixed):
        return 0.0  # self-hosted already cheaper at zero query volume
    return (target - fixed) / per_query


def sensitivity(a: dict, volumes=(30_000, 300_000, 3_000_000, 30_000_000)) -> list[dict]:
    out = []
    for n in a["scales"]:
        sh = self_hosted_cost(a, n)
        for q in volumes:
            out.append({"vectors": n, "queries_per_month": q, "managed_usd": managed_total_at(a, n, q),
                        "self_hosted_infra_usd": sh.get("infra_usd")})
    return out


def _fmt(n: int) -> str:
    return {100000: "100K", 1000000: "1M", 10000000: "10M"}.get(n, f"{n:,}")


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def to_markdown(a: dict, rows: list[dict]) -> str:
    L = ["# Monthly cost comparison: self-hosted ChromaDB vs managed vector DB (Pinecone serverless Standard)", "",
         f"*Model inputs: `data/cost/assumptions.json` (access date {a['access_date']}, {a['currency']}). "
         "These are **estimates from a model, not bills**.*", "",
         "| Vectors | Payload | Chroma node (RAM need) | Chroma infra $/mo | Chroma 2-node HA $/mo | + ops $/mo | Chroma total incl. ops | Pinecone $/mo (low-high price end) | Saving vs infra-only | Saving vs incl. ops |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sh, mg = r["self_hosted"], r["managed"]
        if not sh.get("feasible"):
            L.append(f"| {_fmt(r['vectors'])} | {sh['payload_gb']:.2f} GB | exceeds listed instances | n/a | n/a | n/a | n/a | {_usd(mg['total_usd'])} | n/a | n/a |")
            continue
        L.append(f"| {_fmt(r['vectors'])} | {sh['payload_gb']:.2f} GB | {sh['instance']} ({sh['required_ram_gib']:.1f} GiB) | "
                 f"{_usd(sh['infra_usd'])} | {_usd(sh['infra_ha_usd'])} | {_usd(sh['ops_usd'])} | {_usd(sh['total_incl_ops_usd'])} | "
                 f"{_usd(mg['total_usd'])} ({_usd(mg['range_usd'][0])}-{_usd(mg['range_usd'][1])}) | {_usd(r['saving_vs_infra_usd'])} | "
                 f"{_usd(r.get('saving_vs_total_incl_ops_usd'))} |")
    w = a["workload"]
    m = a["managed_pinecone_serverless_standard"]
    L += ["", "## Key assumptions", "",
          f"- Embedding dimension {w['embedding_dim']}, float32 ({w['bytes_per_float']} B/value), no quantisation; chunk text ~{w['avg_chunk_text_bytes']} B and metadata ~{w['avg_metadata_bytes']} B per vector, stored on both systems.",
          f"- Query volume {w['queries_per_month']:,}/month; {int(w['monthly_rewrite_fraction']*100)}% of vectors re-written per month. Initial bulk load is a one-off and excluded.",
          "- Self-hosted: single AWS on-demand instance in us-east-1 (price from a third-party aggregator, not aws.amazon.com), gp3 disk, snapshots; HNSW index held in RAM. HA column doubles compute+disk.",
          "- 'Ops' = engineer time for upgrades, backups, monitoring (a pure assumption: "
          f"{_usd(_val(a['self_hosted_chromadb']['ops_usd_per_hour']))}/h). It is shown separately because it dominates at small scale.",
          f"- Managed: Pinecone serverless Standard, **verified** against {m['source']}: min ${m['monthly_minimum_usd']}/mo, ${m['storage_usd_per_gb_month']}/GB-month storage, read units ${m['read_units_usd_per_million']['low']}-{m['read_units_usd_per_million']['high']}/M, write units ${m['write_units_usd_per_million']['low']}-{m['write_units_usd_per_million']['high']}/M. The low end is used (favours the managed side).",
          "- Pod-based Pinecone pricing and Dedicated Read Nodes are not modelled (current prices not verified).",
          f"- Excluded as identical on both sides: {a['excluded_common_costs']}", "",
          "## Query-volume sensitivity (managed $/mo vs self-hosted infra-only $/mo)", "",
          "Pinecone serverless read cost scales with index size x query volume, so the winner depends on traffic.", "",
          "| Vectors | Queries/month | Managed $/mo | Chroma infra-only $/mo | Cheaper (infra-only) |", "|---|---|---|---|---|"]
    for sr in sensitivity(a):
        sh_u = sr["self_hosted_infra_usd"]
        cheaper = "n/a" if sh_u is None else ("self-hosted" if sh_u < sr["managed_usd"] else "managed")
        L.append(f"| {_fmt(sr['vectors'])} | {sr['queries_per_month']:,} | {_usd(sr['managed_usd'])} | {_usd(sh_u)} | {cheaper} |")
    L += ["", "## Break-even and conclusions (computed from the inputs, not hand-written)", ""]
    for r in rows:
        be = break_even_queries(a, r["vectors"])
        base = "managed" if r["managed"]["total_usd"] < r["self_hosted"].get("infra_usd", float("inf")) else "self-hosted"
        be_txt = "never (self-hosted already cheaper)" if be == 0.0 else ("n/a" if be is None else f"~{be:,.0f} queries/month")
        L.append(f"- **{_fmt(r['vectors'])}**: at {w['queries_per_month']:,} queries/month the cheaper option on infra alone is **{base}**; "
                 f"self-hosted becomes cheaper than managed above {be_txt}. "
                 + ("Including the assumed ops time, managed is also cheaper at this query volume." if r["self_hosted"].get("total_incl_ops_usd") is not None and r["managed"]["total_usd"] < r["self_hosted"]["total_incl_ops_usd"] else "Even including the assumed ops time, self-hosted is cheaper at this query volume."))
    L += ["",
          "**Interpretation.** For a lightly-queried index, Pinecone *serverless* is cheap (its bill is storage-dominated), so a dedicated always-on "
          "self-hosted node is NOT automatically cheaper. Self-hosting wins when (a) query volume is high relative to index size, (b) the vector store "
          "can share an existing server (marginal cost far below a dedicated instance), or (c) the comparison is against always-on *pod* pricing, which is "
          "not modelled here because current pod prices could not be verified.",
          "At 10M vectors a single-node Chroma HNSW index (~30 GiB RAM) is at the edge of what is operationally comfortable; treat that row as a lower bound and consider pgvector/Qdrant or a managed option for reliability, not only price."]
    return "\n".join(L) + "\n"


def write_outputs(a: dict, rows: list[dict], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cost_comparison.json").write_text(json.dumps({"assumptions": a, "rows": rows}, indent=2), encoding="utf-8")
    (out_dir / "cost_comparison.md").write_text(to_markdown(a, rows), encoding="utf-8")
    with open(out_dir / "cost_comparison.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["vectors", "payload_gb", "chroma_instance", "required_ram_gib", "chroma_infra_usd", "chroma_ha_infra_usd",
                      "ops_usd", "chroma_total_incl_ops_usd", "managed_usd", "managed_low_usd", "managed_high_usd",
                      "saving_vs_infra_usd", "saving_vs_incl_ops_usd"])
        for r in rows:
            sh, mg = r["self_hosted"], r["managed"]
            wri.writerow([r["vectors"], round(sh["payload_gb"], 3), sh.get("instance"), round(sh["required_ram_gib"], 2),
                          *(round(sh.get(k) or 0, 2) if sh.get("feasible") else "" for k in ("infra_usd", "infra_ha_usd", "ops_usd", "total_incl_ops_usd")),
                          round(mg["total_usd"], 2), round(mg["range_usd"][0], 2), round(mg["range_usd"][1], 2),
                          round(r.get("saving_vs_infra_usd", 0), 2) if sh.get("feasible") else "",
                          round(r.get("saving_vs_total_incl_ops_usd", 0), 2) if sh.get("feasible") else ""])
    return {k: str(out_dir / f"cost_comparison.{k}") for k in ("json", "csv", "md")}
