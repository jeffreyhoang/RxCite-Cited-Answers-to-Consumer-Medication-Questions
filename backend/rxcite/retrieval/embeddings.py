# Text embeddings for the three candidate embedding models
#       Every vector is unit length, so cosine similarity is a plain dot product.

from dataclasses import dataclass
from functools import cache

import numpy as np

from rxcite.config import require_env


@dataclass(frozen=True)
class EmbeddingModel:
    name: str  # Hugging Face or OpenAI model id
    dim: int  # vector size we use and store
    query_prefix: str = ""  # some models expect an instruction before search queries
    api: bool = False  # True for OpenAI, False for a local sentence-transformers model


MODELS = {
    "bge-base": EmbeddingModel(
        "BAAI/bge-base-en-v1.5",
        768,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "pubmedbert": EmbeddingModel("NeuML/pubmedbert-base-embeddings", 768),
    "openai-small": EmbeddingModel("text-embedding-3-small", 512, api=True),  # API shortens it
}

OPENAI_BATCH = 256  # ~400-token chunks -> ~100k tokens per request (limit 300k)


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@cache
def local_model(name: str):
    from sentence_transformers import SentenceTransformer  # slow import, only when needed

    return SentenceTransformer(name)


@cache
def openai_client():
    from openai import OpenAI

    return OpenAI(api_key=require_env("OPENAI_API_KEY"))


# Embed texts as "query" (a user question) or "passage" (a chunk); returns float32 unit vectors
def embed(
    texts: list[str], model: str, kind: str = "passage", show_progress: bool = False
) -> np.ndarray:
    m = MODELS[model]
    if kind == "query":
        texts = [m.query_prefix + t for t in texts]
    if m.api:
        out = []
        for i in range(0, len(texts), OPENAI_BATCH):
            batch = texts[i : i + OPENAI_BATCH]
            resp = openai_client().embeddings.create(model=m.name, input=batch, dimensions=m.dim)
            out.extend(d.embedding for d in resp.data)
            if show_progress:
                print(f"  {min(i + OPENAI_BATCH, len(texts))}/{len(texts)}", end="\r")
        vectors = np.array(out, dtype=np.float32)
    else:
        vectors = local_model(m.name).encode(
            texts, batch_size=32, show_progress_bar=show_progress, convert_to_numpy=True
        )
    return normalize(vectors.astype(np.float32))


# Token count of each text with the model's own tokenizer (local models only)
def token_counts(texts: list[str], model: str) -> list[int]:
    tokenizer = local_model(MODELS[model].name).tokenizer
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]]
