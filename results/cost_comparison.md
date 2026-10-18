# Monthly cost comparison: self-hosted ChromaDB vs managed vector DB (Pinecone serverless Standard)

*Model inputs: `data/cost/assumptions.json` (access date 2026-10-18, USD). These are **estimates from a model, not bills**.*

| Vectors | Payload | Chroma node (RAM need) | Chroma infra $/mo | Chroma 2-node HA $/mo | + ops $/mo | Chroma total incl. ops | Pinecone $/mo (low-high price end) | Saving vs infra-only | Saving vs incl. ops |
|---|---|---|---|---|---|---|---|---|---|
| 100K | 0.31 GB | m6i.large (2.3 GiB) | $70.14 | $140.27 | $150.00 | $220.14 | $50.00 ($50.00-$50.00) | $-20.14 | $-170.14 |
| 1M | 3.09 GB | m6i.large (4.8 GiB) | $70.70 | $141.24 | $300.00 | $370.70 | $50.00 ($50.00-$50.00) | $-20.70 | $-320.70 |
| 10M | 30.86 GB | r6i.xlarge (29.9 GiB) | $190.13 | $378.71 | $900.00 | $1,090.13 | $174.31 ($174.31-$194.83) | $-15.82 | $-915.82 |

## Key assumptions

- Embedding dimension 384, float32 (4 B/value), no quantisation; chunk text ~1100 B and metadata ~450 B per vector, stored on both systems.
- Query volume 300,000/month; 10% of vectors re-written per month. Initial bulk load is a one-off and excluded.
- Self-hosted: single AWS on-demand instance in us-east-1 (price from a third-party aggregator, not aws.amazon.com), gp3 disk, snapshots; HNSW index held in RAM. HA column doubles compute+disk.
- 'Ops' = engineer time for upgrades, backups, monitoring (a pure assumption: $75.00/h). It is shown separately because it dominates at small scale.
- Managed: Pinecone serverless Standard, **verified** against https://www.pinecone.io/pricing.md (Pricing Dimensions table) - accessed 2026-10-18; corroborated by third-party trackers: min $50/mo, $0.33/GB-month storage, read units $16.0-18.0/M, write units $4.0-4.5/M. The low end is used (favours the managed side).
- Pod-based Pinecone pricing and Dedicated Read Nodes are not modelled (current prices not verified).
- Excluded as identical on both sides: Embedding generation (local MiniLM: $0 API cost, CPU time only), LLM inference (Groq tokens) and the application server are identical on both sides and therefore excluded from the comparison; they would be added equally to both columns.

## Query-volume sensitivity (managed $/mo vs self-hosted infra-only $/mo)

Pinecone serverless read cost scales with index size x query volume, so the winner depends on traffic.

| Vectors | Queries/month | Managed $/mo | Chroma infra-only $/mo | Cheaper (infra-only) |
|---|---|---|---|---|
| 100K | 30,000 | $50.00 | $70.14 | managed |
| 100K | 300,000 | $50.00 | $70.14 | managed |
| 100K | 3,000,000 | $50.00 | $70.14 | managed |
| 100K | 30,000,000 | $148.39 | $70.14 | self-hosted |
| 1M | 30,000 | $50.00 | $70.70 | managed |
| 1M | 300,000 | $50.00 | $70.70 | managed |
| 1M | 3,000,000 | $150.75 | $70.70 | self-hosted |
| 1M | 30,000,000 | $1,483.90 | $70.70 | self-hosted |
| 10M | 30,000 | $50.00 | $190.13 | managed |
| 10M | 300,000 | $174.31 | $190.13 | managed |
| 10M | 3,000,000 | $1,507.46 | $190.13 | self-hosted |
| 10M | 30,000,000 | $14,838.98 | $190.13 | self-hosted |

## Break-even and conclusions (computed from the inputs, not hand-written)

- **100K**: at 300,000 queries/month the cheaper option on infra alone is **managed**; self-hosted becomes cheaper than managed above ~14,152,592 queries/month. Including the assumed ops time, managed is also cheaper at this query volume.
- **1M**: at 300,000 queries/month the cheaper option on infra alone is **managed**; self-hosted becomes cheaper than managed above ~1,378,775 queries/month. Including the assumed ops time, managed is also cheaper at this query volume.
- **10M**: at 300,000 queries/month the cheaper option on infra alone is **managed**; self-hosted becomes cheaper than managed above ~332,032 queries/month. Including the assumed ops time, managed is also cheaper at this query volume.

**Interpretation.** For a lightly-queried index, Pinecone *serverless* is cheap (its bill is storage-dominated), so a dedicated always-on self-hosted node is NOT automatically cheaper. Self-hosting wins when (a) query volume is high relative to index size, (b) the vector store can share an existing server (marginal cost far below a dedicated instance), or (c) the comparison is against always-on *pod* pricing, which is not modelled here because current pod prices could not be verified.
At 10M vectors a single-node Chroma HNSW index (~30 GiB RAM) is at the edge of what is operationally comfortable; treat that row as a lower bound and consider pgvector/Qdrant or a managed option for reliability, not only price.
