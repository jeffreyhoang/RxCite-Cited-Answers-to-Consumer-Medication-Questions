# Shared class/interface content for the RAG architectures
#       -all architectures need RAGPipeline and retrieve()

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from rxcite.config import TOP_K  # set in config.py


# A piece of an openFDA label returned by retrieval
@dataclass
class Chunk:
    chunk_id: str
    set_id: str  # openFDA label id (used to build the citation link)
    drug_name: str
    section: str  # e.g. "dosage_and_administration"
    text: str
    score: float = 0.0  # retrieval/rerank score; higher is better
    parent_id: str | None = None  # parent section id (Hierarchical RAG)


# Points at label section that the answer relied on
@dataclass
class Citation:
    set_id: str
    section: str


# What the generation returns
@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False  # True when context did not support an answer


# Everything harness + API need for one question
@dataclass
class RAGResult:
    question: str
    architecture: str
    answer: str
    citations: list[Citation]
    chunks: list[Chunk]  # retrieved chunks, best first
    refused: bool = False
    latency_s: float = 0.0


# Generator takes the question and retrieved chunks and returns an Answer
Generator = Callable[[str, list[Chunk]], Answer]


# Base RAG class: subclasses (architectures) implement retrieve(); answer() is shared
class RAGPipeline(ABC):
    name: str = "base"  # set in each subclass, e.g. "naive", "hierarchical", "decomposition"

    def __init__(self, generator: Generator, top_k: int = TOP_K):
        self.generator = generator
        self.top_k = top_k

    # Return up to top_k chunks, best first (implement per architecture)
    @abstractmethod
    def retrieve(self, query: str) -> list[Chunk]:
        """Return up to top_k chunks, best first."""
        pass

    # Retrieve, generate, and time the pipeline (usually not overridden)
    def answer(self, query: str) -> RAGResult:
        start = time.perf_counter()
        chunks = self.retrieve(query)
        generated = self.generator(query, chunks)
        return RAGResult(
            question=query,
            architecture=self.name,
            answer=generated.text,
            citations=generated.citations,
            chunks=chunks,
            refused=generated.refused,
            latency_s=time.perf_counter() - start,
        )
