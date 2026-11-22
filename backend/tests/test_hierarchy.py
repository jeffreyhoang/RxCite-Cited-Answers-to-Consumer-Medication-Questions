from rxcite.pipelines.base import Chunk
from rxcite.retrieval.hierarchy import GAP, TOPIC_BONUS, parent_text, rank_parents


def child(set_id, section, n, score):
    return Chunk(f"{set_id}:{section}:{n}", set_id, "drug", section, "...", score)


def test_rank_groups_by_section_and_keeps_best_score():
    children = [
        child("a", "warnings", 0, 0.60),
        child("a", "warnings", 3, 0.50),
        child("a", "overdosage", 0, 0.55),
        child("b", "description", 1, 0.40),
    ]
    parents = rank_parents(children, "other", n=2)
    assert [(p.set_id, p.section) for p in parents] == [("a", "warnings"), ("a", "overdosage")]
    assert parents[0].score == 0.60 and parents[0].matched == {0: 0.60, 3: 0.50}


def test_topic_bonus_reorders_only_close_scores():
    children = [child("a", "warnings", 0, 0.62), child("a", "adverse_reactions", 0, 0.58)]
    parents = rank_parents(children, "side_effects")  # both sections are in this group
    assert parents[0].section == "warnings"  # 0.67 vs 0.63
    children = [child("a", "overdosage", 0, 0.62), child("a", "adverse_reactions", 0, 0.58)]
    parents = rank_parents(children, "side_effects")  # only adverse_reactions gets the bonus
    assert parents[0].section == "adverse_reactions"
    assert abs(parents[0].score - (0.58 + TOPIC_BONUS)) < 1e-9


def test_short_section_is_returned_whole():
    assert parent_text("short section", {}, {0: 0.5}) == "short section"


def test_long_section_keeps_matches_and_neighbors_under_cap():
    chunks = {i: f"<{i}>" + "x" * 895 for i in range(20)}  # 20 chunks of 900 chars
    text = parent_text("y" * 20_000, chunks, {10: 0.9, 2: 0.5}, max_chars=6000)
    assert len(text) <= 6000
    assert all(f"<{i}>" in text for i in (1, 2, 3, 9, 10, 11))  # both windows fit
    assert "<0>" not in text and "<12>" not in text and GAP in text
    assert text.index("<1>") < text.index("<10>")  # kept in label order


def test_best_match_always_kept_even_if_over_cap():
    chunks = {0: "a" * 7000, 1: "b" * 100}
    assert parent_text("z" * 8000, chunks, {0: 0.9}, max_chars=6000) == "a" * 7000
