# Web API (T11): serves the four RAG architectures to the React frontend
#       Run from backend/: uv run uvicorn api.main:app --reload --port 8000
#       Docs while running: http://localhost:8000/docs
#
#       Endpoints are plain `def`, so FastAPI runs each request in a worker thread. Requests for
#       the same architecture take turns (one lock each); different architectures run at the
#       same time, so the frontend's compare-all mode (four /ask calls at once) runs in parallel.
#       Only Advanced uses the reranker, so the model is never called from two threads at once.

import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import anthropic
import openai
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, StringConstraints

from rxcite.pipelines import PIPELINES
from rxcite.pipelines.base import RAGPipeline, RAGResult, citation_url, section_name

log = logging.getLogger("rxcite.api")

# The four architectures the app offers, in display order (the registry also has eval-only
# variants such as adv-section and dummy, which the API doesn't expose)
ARCHITECTURES = [
    {
        "id": "naive",
        "name": "Naive RAG",
        "description": "Embeds the question and answers from the 10 most similar label chunks.",
    },
    {
        "id": "advanced",
        "name": "Advanced RAG",
        "description": "Rewrites the question, combines keyword (BM25) and vector search over "
        "section-based chunks, and reranks the results with a cross-encoder.",
    },
    {
        "id": "hierarchical",
        "name": "Hierarchical RAG",
        "description": "Detects the drug, searches only its label, and answers from the "
        "full label sections that matched.",
    },
    {
        "id": "decomposition",
        "name": "Decomposition RAG",
        "description": "Splits multi-part questions into sub-questions, searches for each, "
        "and merges the results.",
    },
]
ArchitectureId = Literal["naive", "advanced", "hierarchical", "decomposition"]

# Every architecture embeds the question (OpenAI), searches Supabase, and answers with Claude
REQUIRED_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SUPABASE_DB_URL")

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"  # Vite dev server
MAX_QUESTION_CHARS = 1000

# Errors from the services a pipeline calls -> 502, named so the message says which one failed
UPSTREAM_ERRORS: tuple[tuple[type[Exception], str], ...] = (
    (anthropic.APIError, "Claude API"),
    (openai.APIError, "OpenAI API"),
    (psycopg.Error, "Supabase database"),
)


class AskRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]
    architecture: ArchitectureId


class CitationOut(BaseModel):
    n: int  # the answer's [n] marker
    set_id: str
    section: str
    section_label: str  # e.g. "Drug interactions"
    drug_name: str
    label: str  # e.g. "Lisinopril — Drug interactions"
    url: str  # DailyMed page for the label


class SourceOut(BaseModel):
    rank: int  # 1 = best
    chunk_id: str
    set_id: str
    drug_name: str
    section: str
    section_label: str
    text: str
    score: float
    url: str
    cited: bool  # True if the answer cites this source's (label, section)


class AskResponse(BaseModel):
    question: str
    architecture: str
    answer: str  # contains [n] markers matching citations[].n (none when refused)
    refused: bool
    citations: list[CitationOut]
    sources: list[SourceOut]
    info: dict[str, Any]  # intermediate steps; keys depend on the architecture
    latency_s: float


def to_response(result: RAGResult, architecture: str) -> AskResponse:
    cited = {(c.set_id, c.section) for c in result.citations}
    return AskResponse(
        question=result.question,
        architecture=architecture,
        answer=result.answer,
        refused=result.refused,
        citations=[
            CitationOut(
                n=i,
                set_id=c.set_id,
                section=c.section,
                section_label=section_name(c.section),
                drug_name=c.drug_name,
                label=c.label,
                url=c.url,
            )
            for i, c in enumerate(result.citations, 1)
        ],
        sources=[
            SourceOut(
                rank=i,
                chunk_id=c.chunk_id,
                set_id=c.set_id,
                drug_name=c.drug_name,
                section=c.section,
                section_label=section_name(c.section),
                text=c.text,
                score=c.score,
                url=citation_url(c.set_id),
                cited=(c.set_id, c.section) in cited,
            )
            for i, c in enumerate(result.chunks, 1)
        ],
        info=result.info,
        latency_s=result.latency_s,
    )


def missing_env(required: Sequence[str]) -> list[str]:
    return [name for name in required if not os.environ.get(name)]


# Load the BM25 index and the reranker now, so the first Advanced question isn't slow
def warm_up() -> None:
    from rxcite.bm25 import index
    from rxcite.rerank import reranker

    for step, load in (("BM25 index", lambda: index("section")), ("reranker", reranker)):
        try:
            load()
            log.info("Warm-up: %s loaded", step)
        except Exception:  # not fatal: it loads on the first Advanced question instead
            log.exception("Warm-up: %s failed to load", step)


def create_app(
    builders: Mapping[str, Callable[[], RAGPipeline]] | None = None,
    required_env: Sequence[str] = REQUIRED_ENV,
    warm: bool | None = None,
) -> FastAPI:
    """builders: architecture id -> function that builds its pipeline (tests pass fakes).
    warm: preload the BM25 index and reranker (default on; RXCITE_WARMUP=0 turns it off)."""
    if builders is None:
        builders = {a["id"]: PIPELINES[a["id"]] for a in ARCHITECTURES}
    if warm is None:
        warm = os.environ.get("RXCITE_WARMUP", "1") != "0"

    pipelines: dict[str, RAGPipeline] = {}
    load_errors: dict[str, str] = {}
    locks = {arch: threading.Lock() for arch in builders}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for arch, build in builders.items():
            try:
                pipelines[arch] = build()
            except Exception as e:
                log.exception("Could not build the %s pipeline", arch)
                load_errors[arch] = f"{type(e).__name__}: {e}"
        if missing := missing_env(required_env):
            log.warning("Missing settings in backend/.env: %s", ", ".join(missing))
        if warm and "advanced" in pipelines and not missing_env(required_env):
            warm_up()
        yield

    app = FastAPI(title="RxCite API", version="0.1.0", lifespan=lifespan)
    origins = os.environ.get("RXCITE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict:
        missing = missing_env(required_env)
        return {
            "status": "degraded" if missing or load_errors else "ok",
            "architectures": [a["id"] for a in ARCHITECTURES if a["id"] in pipelines],
            "unavailable": load_errors,
            "missing_env": missing,
        }

    @app.get("/architectures")
    def architectures() -> list[dict]:
        return [a for a in ARCHITECTURES if a["id"] in builders]

    @app.post("/ask")
    def ask(body: AskRequest) -> AskResponse:
        arch = body.architecture
        if arch not in pipelines:
            reason = load_errors.get(arch, "not enabled on this server")
            raise HTTPException(503, f"{arch} is unavailable: {reason}")
        if missing := missing_env(required_env):
            raise HTTPException(
                503, f"Server is missing settings: {', '.join(missing)} (see backend/.env)"
            )
        try:
            with locks[arch]:
                result = pipelines[arch].answer(body.question)
        except Exception as e:
            for error_type, service in UPSTREAM_ERRORS:
                if isinstance(e, error_type):
                    log.exception("%s failed while answering with %s", service, arch)
                    raise HTTPException(502, f"{service} request failed; try again") from e
            raise
        return to_response(result, arch)

    return app


app = create_app()
