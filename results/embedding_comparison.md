# Embedding model comparison (T5)

Plain dense top-10 retrieval over the fixed-size chunks (23,921 chunks, at most 1,600 characters each, 200 overlap, never crossing a section).
Scored on all 160 answerable questions. A hit is a retrieved chunk from a gold (set_id, section).

| Model | Vectors | recall@10 | MRR | Embed corpus | Query time |
|---|---|---|---|---|---|
| OpenAI `text-embedding-3-small` | 512 | **0.659** | **0.498** | 6.9 min (API) | 0.23 s |
| `BAAI/bge-base-en-v1.5` | 768 | 0.581 | 0.410 | 12.3 min (local, M2) | 0.04 s |
| `NeuML/pubmedbert-base-embeddings` | 768 | 0.250 | 0.183 | 11.2 min (local, M2) | 0.04 s |

Reference points: oracle 1.0 / 1.0, random sections ≈ 0.

Notes
- Local models cut text longer than 512 tokens: 207 chunks (0.9%) for bge-base and 98 (0.4%) for pubmedbert. OpenAI has no cut at this size.
- Query time for OpenAI includes the API round trip.
- Raw results: `results/dense-<model>_full.json`. Rerun with `uv run python scripts/run_eval.py --arch dense-<model> --full --no-judge`.

## Choice

**OpenAI `text-embedding-3-small`, 512 numbers per vector.**

- Best on both metrics: recall@10 is 0.078 above bge-base, well past the 0.03 treated as noise for 160 questions, and MRR is 0.088 higher.
- Small vectors (512, stored as `halfvec`) keep both planned chunk sets well inside the Supabase free tier (500 MB).
- No 512-token cut, so long chunks are embedded in full.
- Costs: an API call per question (~0.2 s) and an OpenAI key. Embedding the corpus once cost about $0.15.
- pubmedbert likely scored low because it is tuned for sentence similarity, not for matching consumer questions to passages.

This embedder is used for every architecture (T6–T9).
