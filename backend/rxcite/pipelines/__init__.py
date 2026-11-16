# Registry of RAG architectures the eval runner can run, by name: name -> function that builds it
#       By ticket: dummy/oracle (T4), dense (T5), naive (T6), advanced (T7), hierarchical (T8),
#       decomposition (T9). Each variant needs its own name: the runner caches answers per name.

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


def _advanced() -> RAGPipeline:
    from rxcite.pipelines.advanced import AdvancedPipeline

    return AdvancedPipeline()


# Retrieval-only Advanced RAG ablation rows (T7): each adds one step to the row above it
def _advanced_retrieval(name: str, **steps: bool) -> Callable[[], RAGPipeline]:
    def build() -> RAGPipeline:
        from rxcite.pipelines.advanced import AdvancedPipeline
        from rxcite.pipelines.baselines import dummy_generator

        return AdvancedPipeline(dummy_generator, name=name, **steps)

    return build


def _decomposition() -> RAGPipeline:
    from rxcite.pipelines.decomposition import DecompositionPipeline

    return DecompositionPipeline()


def _decomposition_retrieval() -> RAGPipeline:
    from rxcite.pipelines.baselines import dummy_generator
    from rxcite.pipelines.decomposition import DecompositionPipeline

    return DecompositionPipeline(dummy_generator, name="decomposition-retrieval")


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
    "advanced": _advanced,  # T7: rewrite -> BM25 + dense -> RRF -> rerank -> shared generator
    "adv-section": _advanced_retrieval("adv-section", rewrite=False, hybrid=False, rerank=False),
    "adv-rewrite": _advanced_retrieval("adv-rewrite", hybrid=False, rerank=False),
    "adv-hybrid": _advanced_retrieval("adv-hybrid", rerank=False),
    "advanced-retrieval": _advanced_retrieval("advanced-retrieval"),
    "decomposition": _decomposition,  # T9: sub-questions -> dense search each -> merge
    "decomposition-retrieval": _decomposition_retrieval,
    "dense": _pgvector_dense,  # chosen embedder, pgvector in Supabase
    # T5 embedder comparison (in memory): dense-bge-base, dense-pubmedbert, dense-openai-small
    **{f"dense-{m}": _dense(m) for m in MODELS},
}
