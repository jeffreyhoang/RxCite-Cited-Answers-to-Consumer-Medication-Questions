import json

import pytest

from rxcite import rewrite
from rxcite.rewrite import rewrite_query

QUESTION = "can i drink while on advil?"


class FakeResponse:
    def __init__(self, text):
        self.text = text


def fake_call(text, seen=None):
    def call(prompt, model, **kwargs):
        if seen is not None:
            seen.append((prompt, model, kwargs))
        return FakeResponse(text)

    return call


def test_uses_rewritten_query(monkeypatch):
    seen = []
    reply = json.dumps({"query": "  Advil ibuprofen alcohol drug interactions "})
    monkeypatch.setattr(rewrite.llm, "call", fake_call(reply, seen))
    assert rewrite_query(QUESTION) == "Advil ibuprofen alcohol drug interactions"
    [(prompt, model, kwargs)] = seen
    assert prompt == QUESTION and model == rewrite.GENERATOR_MODEL
    assert kwargs["schema"]["required"] == ["query"]


@pytest.mark.parametrize("text", ["", '{"query": ', '{"query": "   "}', "{}"])
def test_falls_back_to_original_question(monkeypatch, text):
    monkeypatch.setattr(rewrite.llm, "call", fake_call(text))
    assert rewrite_query(QUESTION) == QUESTION
