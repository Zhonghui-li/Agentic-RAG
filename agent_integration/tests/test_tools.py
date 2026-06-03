"""L1 (every-PR) deterministic tests for the structured tools.

lookup_schedule and get_academic_calendar are pure functions over the JSON
data (no LLM, no network), so their outputs are fixed and assertable.
web_search hits the network -> excluded from L1 (belongs to L2 / integration).
"""
from agents.slug_tools import lookup_schedule, get_academic_calendar


def test_lookup_schedule_known_course():
    out = lookup_schedule("CSE 142", "Fall 2025")
    assert "CSE 142" in out
    assert "Fall 2025" in out
    # synthesized but deterministic (seed=42): MWF 16:00, Open
    assert "16:00" in out and "MWF" in out


def test_lookup_schedule_code_normalization():
    # 'cse142' should normalize to 'CSE 142'
    assert "CSE 142" in lookup_schedule("cse142", "Fall 2025")


def test_lookup_schedule_not_found():
    out = lookup_schedule("CSE 999")
    assert "no scheduled offering" in out.lower()


def test_calendar_known_quarter():
    out = get_academic_calendar("Winter 2026")
    assert "January 5" in out          # instruction begins
    assert "2026-01-05" in out         # ISO form also present


def test_calendar_lists_all_when_no_term():
    out = get_academic_calendar()
    assert "Fall 2025" in out and "Winter 2026" in out and "Spring 2026" in out
