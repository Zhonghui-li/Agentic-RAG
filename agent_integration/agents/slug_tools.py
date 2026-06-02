"""Structured tools for the UCSC Slug Advisor agent.

These three tools do *structured / dynamic / computational* lookups that
semantic retrieval (search_catalog) can't do reliably:

  - lookup_schedule        : exact per-term class-offering lookup
  - get_academic_calendar  : key quarter dates (+ today, for date math)
  - web_search             : keyless fallback for out-of-corpus questions

Each function is a plain, unit-testable Python function. The agent layer
(slug_advisor_agent.py) wraps them as LangChain tools.
"""
import os
import json
import datetime
from typing import Optional

_DATA_DIR = os.environ.get(
    "UCSC_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data-ucsc"),
)


def _load(name: str) -> dict:
    with open(os.path.join(_DATA_DIR, name)) as f:
        return json.load(f)


# Loaded once at import (small files).
_SCHEDULE = _load("schedule.json")
_CALENDAR = _load("calendar.json")


def _norm_code(code: str) -> str:
    """Normalize 'cse142', 'CSE  142' -> 'CSE 142'."""
    import re
    code = (code or "").upper().strip()
    m = re.match(r"([A-Z]{2,4})\s*0*(\d{1,3}[A-Z]?)", code)
    return f"{m.group(1)} {m.group(2)}" if m else code


def _iso_to_human(iso: str) -> str:
    """'2026-01-05' -> 'January 5, 2026'. Pass ranges 'a/b' through both ends."""
    try:
        d = datetime.date.fromisoformat(iso)
        return d.strftime("%B %-d, %Y")
    except ValueError:
        return iso


# ----------------------------------------------------------------------------
# Tool 1: schedule lookup (structured, exact match)
# ----------------------------------------------------------------------------
def lookup_schedule(course_code: str, term: Optional[str] = None) -> str:
    """Look up when a course is offered: term, meeting days/time, location,
    instructor, and seat availability. `term` is optional (e.g. 'Fall 2025');
    omit it to see all terms the course is offered."""
    code = _norm_code(course_code)
    hits = [o for o in _SCHEDULE["offerings"] if o["course_code"] == code]
    if term:
        t = term.lower()
        hits = [o for o in hits if t in o["term"].lower()]

    if not hits:
        scope = f" in {term}" if term else ""
        return (f"{code} has no scheduled offering{scope}. "
                f"(Terms covered: {', '.join(_SCHEDULE['term_list'])}.)")

    lines = []
    for o in hits:
        seats_left = o["capacity"] - o["enrolled"]
        avail = (f"{o['enrolled']}/{o['capacity']} enrolled, {seats_left} seats left"
                 if seats_left > 0 else
                 f"full ({o['enrolled']}/{o['capacity']}), {o['waitlisted']} waitlisted")
        lines.append(
            f"{code} — {o['title']} [{o['term']}]: {o['days']} "
            f"{o['start_time']}–{o['end_time']}, {o['location']}, "
            f"instructor {o['instructor']}. {avail}. Status: {o['status']}."
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Tool 2: academic calendar (structured + date math support)
# ----------------------------------------------------------------------------
def get_academic_calendar(term: Optional[str] = None) -> str:
    """Get key UCSC academic-calendar dates (instruction begins/ends, final
    exams, quarter ends, priority enrollment). `term` is optional (e.g.
    'Winter 2026'); omit it to get all quarters. Today's date is included so
    you can compute 'days until' a date."""
    quarters = _CALENDAR["quarters"]
    today = datetime.date.today()

    def fmt(name: str, q: dict) -> str:
        def show(key, label):
            v = q.get(key, "")
            if "/" in v:  # a date range like exams
                a, b = v.split("/")
                return f"  {label}: {_iso_to_human(a)} – {_iso_to_human(b)}"
            return f"  {label}: {_iso_to_human(v)} ({v})"
        return "\n".join([
            f"{name} ({_CALENDAR['academic_year']}):",
            show("instruction_begins", "Instruction begins"),
            show("instruction_ends", "Instruction ends"),
            show("final_exams", "Final exams"),
            show("quarter_ends", "Quarter ends"),
            show("priority_enrollment_begins", "Priority enrollment begins"),
        ])

    if term:
        match = next((k for k in quarters if term.lower() in k.lower()), None)
        if not match:
            return (f"No calendar found for '{term}'. "
                    f"Quarters: {', '.join(quarters)}.")
        body = fmt(match, quarters[match])
    else:
        body = "\n\n".join(fmt(k, v) for k, v in quarters.items())

    return f"{body}\n\n(Today is {today.isoformat()}.)"


# ----------------------------------------------------------------------------
# Tool 3: web search (keyless fallback for out-of-corpus questions)
# ----------------------------------------------------------------------------
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for information not in the UCSC course catalog. Use only
    as a fallback when the catalog and schedule tools can't answer."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:  # network / lib issues shouldn't crash the agent
        return f"web_search unavailable ({type(e).__name__}: {e})."

    if not results:
        return f"No web results for '{query}'."
    return "\n".join(
        f"- {r.get('title', '')}: {r.get('body', '')[:200]} ({r.get('href', '')})"
        for r in results
    )
