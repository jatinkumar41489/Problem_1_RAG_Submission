# Atlas Plans, Pricing and Limits

Atlas is available in three plans. All prices are in US dollars and exclude applicable taxes. A 14-day free trial is available on the Starter and Pro plans without a credit card.

## Plans

| Plan | Price | Storage included | Max connections |
| --- | --- | --- | --- |
| Starter | $29 per month | 10 GB | 100 |
| Pro | $199 per month | 500 GB | 2,000 |
| Enterprise | Custom pricing | Custom | Custom |

The Pro plan adds point-in-time recovery and multi-zone replication. The Enterprise plan adds a dedicated cluster, private networking and a named technical account manager.

## Overage and data transfer

Storage beyond the included amount is billed at $0.12 per GB-month. Data transfer into Atlas is free. Data transfer out is free for the first 100 GB each month and then costs $0.08 per GB. Overage is billed at the end of each monthly billing period together with the plan fee.

## Service limits

The maximum size of a single document is 16 MB. A Pro cluster accepts up to 5,000 requests per second, and requests above this rate are throttled with an HTTP 429 response. The Starter plan is limited to 500 requests per second. A single collection may hold up to 2 billion documents, and a cluster supports at most 64 TB of storage.

## Plan changes and cancellation

Customers may upgrade at any time; the new plan takes effect immediately and the fee is prorated. Downgrades take effect at the next billing period, and a downgrade is rejected if the stored data exceeds the target plan's limits. Cancelled clusters are retained in a suspended state for 14 days before their data is deleted permanently.
