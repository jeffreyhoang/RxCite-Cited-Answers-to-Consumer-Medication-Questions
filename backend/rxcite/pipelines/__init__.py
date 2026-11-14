# Registry of RAG architectures the eval runner can run, by name: name -> function that builds it
#       Filled in as architectures are added (dummy/oracle in T4, naive in T6, ...)

from collections.abc import Callable

from rxcite.pipelines.base import RAGPipeline


def _oracle() -> RAGPipeline:
    from rxcite.pipelines.baselines import OraclePipeline

    return OraclePipeline()


def _dummy() -> RAGPipeline:
    from rxcite.pipelines.baselines import DummyPipeline

    return DummyPipeline()


PIPELINES: dict[str, Callable[[], RAGPipeline]] = {"oracle": _oracle, "dummy": _dummy}
