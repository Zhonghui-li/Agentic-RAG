"""L1 (every-PR) deterministic tests for the eval graders.

These test CODE, not the agent: no LLM, no network, fully reproducible.
They MUST pass 100% — a failure is a real bug, not model variance.
"""
from eval.graders import (
    fact_matched, answer_correct, tool_selection_ok, context_recall_hit,
)


# --- format normalization: the bugs the first eval run exposed ---
def test_day_code_matches_natural_language():
    assert fact_matched("TuTh", "It meets on Tuesdays and Thursdays.")
    assert fact_matched("MWF", "Mondays, Wednesdays, and Fridays")
    assert not fact_matched("TuTh", "It meets on Mondays and Wednesdays.")


def test_time_12h_24h_equivalence():
    assert fact_matched("16:00", "from 4:00 PM to 5:05 PM")
    assert fact_matched("10:40", "10:40 AM")
    assert not fact_matched("16:00", "from 9:20 to 10:25")


def test_date_iso_and_natural_equivalence():
    assert fact_matched("January 5", "Winter 2026 begins on January 5, 2026.")
    assert fact_matched("2026-01-05", "Winter 2026 begins on January 5, 2026.")


def test_course_code_word_boundary():
    # "CSE 12" must NOT match "CSE 120"
    assert fact_matched("CSE 12", "Prerequisite: CSE 12 and CSE 16.")
    assert not fact_matched("CSE 12", "Prerequisite: CSE 120 only.")


def test_bare_number_word_boundary():
    assert fact_matched("5", "CSE 3 is a 5-credit course.")
    assert not fact_matched("5", "There are 15 seats and 55 enrolled.")


# --- AND / OR-group semantics ---
def test_answer_correct_all_required():
    assert answer_correct("needs CSE 12 and CSE 16", ["CSE 12", "CSE 16"]) == []
    assert answer_correct("needs CSE 12 only", ["CSE 12", "CSE 16"]) == ["CSE 16"]


def test_or_group_any_alternative():
    facts = [["CSE 140", "CSE 142", "CSE 144"]]  # any one is acceptable
    assert answer_correct("I recommend CSE 140 Artificial Intelligence.", facts) == []
    assert answer_correct("I recommend CSE 999.", facts) == facts  # none matched


# --- tool selection ---
def test_tool_selection():
    assert tool_selection_ok(["search_catalog"], "search_catalog")
    assert not tool_selection_ok(["search_catalog"], "web_search")
    assert tool_selection_ok(["search_catalog", "lookup_schedule"], "multi")
    assert not tool_selection_ok(["search_catalog"], "multi")


# --- context recall ---
def test_context_recall():
    out = "[CSE 142] Machine Learning ... [CSE 101] Data Structures ..."
    assert context_recall_hit(out, ["CSE 142"])
    assert context_recall_hit(out, ["CSE 142", "CSE 101"])
    assert not context_recall_hit(out, ["CSE 130"])
