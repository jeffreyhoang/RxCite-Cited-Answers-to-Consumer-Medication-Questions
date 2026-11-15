# Registry of RAG architectures the eval runner can run, by name: name -> function that builds it
#       Added per ticket: dummy/oracle (T4), dense (T5), naive (T6), hierarchical (T8), ...

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


def _hierarchical() -> RAGPipeline:
    from rxcite.pipelines.hierarchical import HierarchicalPipeline

    return HierarchicalPipeline()


# Retrieval only (placeholder answers, no Haiku): for full-set recall without generation cost
def _hierarchical_retrieval() -> RAGPipeline:
    from rxcite.pipelines.baselines import dummy_generator
    from rxcite.pipelines.hierarchical import HierarchicalPipeline

    return HierarchicalPipeline(dummy_generator, name="hierarchical-retrieval")


def _dense(model: str) -> Callable[[], RAGPipeline]:
    def build() -> RAGPipeline:
        from rxcite.pipelines.dense import DensePipeline

        return DensePipeline(model)

    return build


PIPELINES: dict[str, Callable[[], RAGPipeline]] = {
    "oracle": _oracle,
    "dummy": _dummy,
    "naive": _naive,  # T6: dense top-10 -> shared generator
    "hierarchical": _hierarchical,  # T8: drug filter -> child chunks -> parent sections
    "hierarchical-retrieval": _hierarchical_retrieval,
    "dense": _pgvector_dense,  # chosen embedder, pgvector in Supabase
    # T5 embedder comparison (in memory): dense-bge-base, dense-pubmedbert, dense-openai-small
    **{f"dense-{m}": _dense(m) for m in MODELS},
}
