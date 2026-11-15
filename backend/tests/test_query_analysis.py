import json

from rxcite import query_analysis
from rxcite.query_analysis import SECTION_GROUPS, analyze, labels_for


class FakeResponse:
    def __init__(self, data):
        self.text = json.dumps(data)


def test_analyze_parses_reply(monkeypatch):
    monkeypatch.setattr(query_analysis, "drug_names", lambda: ["metformin hcl", "silodosin"])
    reply = {"drugs": ["silodosin", "silodosin"], "section_type": "dosage"}
    monkeypatch.setattr(query_analysis.llm, "call", lambda *a, **k: FakeResponse(reply))
    info = analyze("can i eat after taking rapaflo?")
    assert info.drugs == ["silodosin"] and info.section_type == "dosage"


def test_no_drugs_means_no_labels():
    assert labels_for([]) == []  # no database call needed


def test_section_groups_use_real_section_names():
    real = {"adverse_reactions", "dosage_and_administration", "drug_interactions", "warnings"}
    names = {s for group in SECTION_GROUPS.values() for s in group}
    assert real <= names and "other" in SECTION_GROUPS


def test_unreadable_reply_means_no_drug(monkeypatch):
    monkeypatch.setattr(query_analysis, "drug_names", lambda: ["silodosin"])
    fake = FakeResponse({})
    fake.text = ""  # reply cut off before any JSON
    monkeypatch.setattr(query_analysis.llm, "call", lambda *a, **k: fake)
    info = analyze("anything")
    assert info.drugs == [] and info.section_type == "other"
