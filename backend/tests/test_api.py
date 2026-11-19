import psycopg
import pytest
from fastapi.testclient import TestClient

from api.main import ARCHITECTURES, create_app
from rxcite.pipelines.base import Chunk, Citation, RAGPipeline, RAGResult

QUESTION = "can i drink alcohol with ibuprofen?"


# Fixed result or error, no network calls
class Fake(RAGPipeline):
    def __init__(self, name, refused=False, error=None):
        super().__init__(generator=None)
        self.name, self.refused, self.error = name, refused, error

    def retrieve(self, query):
        return []

    def answer(self, query):
        if self.error:
            raise self.error
        chunks = [
            Chunk("s1:warnings:0", "s1", "ibuprofen", "warnings", "Stomach bleeding...", 0.9),
            Chunk("s2:drug_interactions:0", "s2", "aspirin", "drug_interactions", "...", 0.5),
        ]
        citations = [] if self.refused else [Citation("s1", "warnings", "ibuprofen")]
        text = "The labels don't cover this." if self.refused else "Bleeding risk rises [1]."
        info = {"search_query": "ibuprofen alcohol"}
        return RAGResult(query, self.name, text, citations, chunks, self.refused, 1.5, info)


def client(builders=None, required_env=()):
    builders = builders or {a["id"]: (lambda a=a: Fake(a["id"])) for a in ARCHITECTURES}
    return TestClient(create_app(builders, required_env=required_env, warm=False))


def ask(c, architecture="naive", question=QUESTION):
    return c.post("/ask", json={"question": question, "architecture": architecture})


def test_health_and_architectures():
    with client() as c:
        health = c.get("/health").json()
        assert health["status"] == "ok"
        assert health["architectures"] == ["naive", "advanced", "hierarchical", "decomposition"]
        archs = c.get("/architectures").json()
        assert [a["id"] for a in archs] == health["architectures"]
        assert all(a["name"] and a["description"] for a in archs)


def test_ask_returns_numbered_citations_and_marks_cited_sources():
    with client() as c:
        r = ask(c, "advanced")
    assert r.status_code == 200
    body = r.json()
    assert body["architecture"] == "advanced"
    assert body["answer"] == "Bleeding risk rises [1]."
    assert body["refused"] is False
    [cite] = body["citations"]
    assert cite["n"] == 1
    assert cite["label"] == "Ibuprofen — Warnings"
    assert cite["url"].endswith("setid=s1")
    assert [s["rank"] for s in body["sources"]] == [1, 2]
    assert [s["cited"] for s in body["sources"]] == [True, False]
    assert body["sources"][1]["section_label"] == "Drug interactions"
    assert body["info"] == {"search_query": "ibuprofen alcohol"}


def test_refused_answer_has_no_citations():
    with client({"naive": lambda: Fake("naive", refused=True)}) as c:
        body = ask(c).json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert not any(s["cited"] for s in body["sources"])


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   ", "architecture": "naive"},
        {"question": "x" * 1001, "architecture": "naive"},
        {"question": QUESTION, "architecture": "adv-section"},  # eval-only variant
        {"question": QUESTION},
    ],
)
def test_invalid_requests_are_422(payload):
    with client() as c:
        assert c.post("/ask", json=payload).status_code == 422


def test_question_is_trimmed():
    with client() as c:
        assert ask(c, question=f"  {QUESTION}\n").json()["question"] == QUESTION


def test_pipeline_that_failed_to_build_is_503():
    def broken():
        raise RuntimeError("model download failed")

    with client({"naive": lambda: Fake("naive"), "advanced": broken}) as c:
        health = c.get("/health").json()
        r = ask(c, "advanced")
    assert health["status"] == "degraded"
    assert health["architectures"] == ["naive"]
    assert "model download failed" in health["unavailable"]["advanced"]
    assert r.status_code == 503
    assert "advanced is unavailable" in r.json()["detail"]


def test_missing_env_is_503(monkeypatch):
    monkeypatch.delenv("RXCITE_TEST_KEY", raising=False)
    with client(required_env=["RXCITE_TEST_KEY"]) as c:
        assert c.get("/health").json()["missing_env"] == ["RXCITE_TEST_KEY"]
        r = ask(c)
    assert r.status_code == 503
    assert "RXCITE_TEST_KEY" in r.json()["detail"]


def test_upstream_failure_is_502():
    error = psycopg.OperationalError("connection refused")
    with client({"naive": lambda: Fake("naive", error=error)}) as c:
        r = ask(c)
    assert r.status_code == 502
    assert "Supabase" in r.json()["detail"]


# Without CORS headers the frontend would report the server as unreachable
def test_unexpected_error_is_500_with_cors():
    with client({"naive": lambda: Fake("naive", error=ValueError("bug"))}) as c:
        r = c.post(
            "/ask",
            json={"question": QUESTION, "architecture": "naive"},
            headers={"Origin": "http://localhost:5173"},
        )
    assert r.status_code == 500
    assert "naive" in r.json()["detail"]
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_allows_vite_dev_server():
    with client() as c:
        r = c.options(
            "/ask",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
