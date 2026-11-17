# Advanced RAG ablation (T7)

Retrieval only (no generation), all answerable questions. Each row adds one step to
the row above it. A hit is a retrieved chunk from a gold (set_id, section).

| Row | Pipeline | recall@10 | MRR | Mean latency |
|---|---|---|---|---|
| `dense` | Fixed chunks, dense top-10 (Naive retrieval) | 0.659 | 0.497 | 0.42 s |
| `adv-section` | + section-based chunks with drug \| section header | 0.700 | 0.535 | 0.60 s |
| `adv-rewrite` | + query rewriting | 0.809 | 0.766 | 2.06 s |
| `adv-hybrid` | + BM25 + dense, RRF fusion (k = 60) | 0.784 | 0.716 | 1.13 s |
| `advanced-retrieval` | + bge-reranker-base over the top 50 (full Advanced) | 0.781 | 0.718 | 3.00 s |

n = 160 answerable questions.

Latency is the mean over every question run (answerable + refusal). Rewrites are cached, so only the first row run with rewriting pays for the Haiku calls; later rows reuse them and understate that cost.

## Reranker time per question

`advanced-retrieval`, 296 questions, up to 50 (question, chunk) pairs each: mean 1.69 s, median 1.65 s, max 12.43 s.

## Which step helped most

TODO: a few lines on which step moved recall@10 / MRR the most, and at what latency.
