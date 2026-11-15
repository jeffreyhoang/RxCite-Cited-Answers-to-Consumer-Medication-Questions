# Registry of RAG architectures the eval runner can run, by name: name -> function that builds it
#       Filled in as architectures are added (dummy/oracle in T4, dense in T5, naive in T6, ...)

from collections.abc import Callable

from rxcite.embeddings import MODELS
from rxcite.pipelines.base import RAGPipeline


def _oracle() -> RAGPipeline:
    from rxcite.pipelines.baselines import OraclePipeline

    return OraclePipeline()


def _dummy() -> RAGPipeline:
    from rxcite.pipelines.baselines import DummyPipeline

    return DummyPipeline()


def _pgvector_dense() -> RAGPipeline:
    from rxcite.pipelines.dense import PgvectorDensePipeline

    return PgvectorDensePipeline()


def _naive() -> RAGPipeline:
    from rxcite.pipelines.naive import NaivePipeline

    return NaivePipeline()


def _dense(model: str) -> Callable[[], RAGPipeline]:
    def build() -> RAGPipeline:
        from rxcite.pipelines.dense import DensePipeline

        return DensePipeline(model)

    return build


PIPELINES: dict[str, Callable[[], RAGPipeline]] = {
    "oracle": _oracle,
    "dummy": _dummy,
    "naive": _naive,  # T6: dense top-10 -> shared generator
    "dense": _pgvector_dense,  # chosen embedder, pgvector in Supabase
    # T5 embedder comparison (in memory): dense-bge-base, dense-pubmedbert, dense-openai-small
    **{f"dense-{m}": _dense(m) for m in MODELS},
}
