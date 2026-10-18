# Cost assumptions

Single source of truth: `data/cost/assumptions.json` (every input carries a `verified` note). Regenerate: `python -m app.cli cost` -> `results/cost_comparison.{json,csv,md}`. Formulas are in the docstring of `app/evaluation/cost_analysis.py` and unit-tested.

## Verification status (checked 2026-10-18)
| Input | Status |
|---|---|
| Pinecone serverless Standard: $50/mo minimum, $0.33/GB-month, read units $16-18/M, write units $4-4.50/M | **Verified** on pinecone.io/pricing (pricing.md) and corroborated by third-party trackers |
| Pinecone read-unit rule (~1 RU per GB of namespace per query, min 0.25) and write-unit rule (~1 WU/KB) | From third-party summaries of Pinecone docs - **verify before relying on it** |
| EC2 on-demand prices (m6i.large $0.096/h, m6i.xlarge $0.192, r6i.xlarge $0.252, r6i.2xlarge $0.504, r6i.4xlarge $1.008; us-east-1) | Third-party aggregators (doit.com, holori.com); **not** aws.amazon.com |
| EBS gp3 $0.08/GB-month | Weak source (blog) - re-check |
| Snapshot $/GB, HNSW overhead 1.5x, RAM headroom 1.3x, baseline 2 GiB, disk headroom 1.5x, ops hours and $75/h | **Unverified estimates / assumptions** |
| Pod-based Pinecone pricing, Dedicated Read Nodes | **Not modelled** (prices not verified) |

## Scenario
384-d float32; 1.1 KB chunk text + 0.45 KB metadata per vector stored on both sides; 300K queries/month; 10% of vectors rewritten monthly; one region; embedding and LLM costs excluded because identical on both sides (local MiniLM has $0 API cost).

## Result in one paragraph
At the stated scenario the **managed serverless option is as cheap or cheaper on infrastructure alone at all three scales**, because Pinecone's bill is storage-dominated at low query volume and the self-hosted node is an always-on instance. Self-hosting becomes cheaper above a computed break-even query volume (~14M queries/month at 100K vectors, ~1.4M at 1M, ~0.33M at 10M). Counting ops time (assumption), managed wins at the baseline. The assignment premise ("always-on pods make large lightly-queried indexes expensive") applies to pod-style pricing, which we could not verify; this repo therefore does **not** claim a cost win, only a quantified break-even. Change any assumption in the JSON and re-run.

## Trade-offs accepted by self-hosting
Operational effort (patching, backups, monitoring), single-node availability (HA column doubles cost), no managed scaling, local-disk durability, single-writer Chroma, and at 10M vectors a ~30 GiB in-memory HNSW index that is at the practical edge of single-node Chroma.
