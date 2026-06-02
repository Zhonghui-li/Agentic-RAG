"""Deterministic graders for the Slug Advisor eval (mirrors promptfoo's
`contains-all` pattern, with format normalization so natural-language answers
aren't unfairly failed).

Three checks, all reference-based (use the gold fields baked into the eval set):
  - answer_correct      : every answer_fact is present in the answer (after
                          canonicalizing day codes / times / dates)
  - tool_selection_ok   : the expected tool was actually called
  - context_recall_hit  : every gold_source course was retrieved by search_catalog

Why normalization: "TuTh" ≡ "Tuesdays and Thursdays", "16:00" ≡ "4:00 PM",
"2026-01-05" ≡ "January 5, 2026". Plain substring matching fails these even
though the answer is correct (see the first 44-question run).
"""
import re

# ----------------------------------------------------------------------------
# Canonicalizers: turn a piece of text into a normalized set of facts
# ----------------------------------------------------------------------------
_DAYCODE = {"M": "mon", "Tu": "tue", "W": "wed", "Th": "thu", "F": "fri", "Sa": "sat", "Su": "sun"}
_DAYNAME = {"monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu",
            "friday": "fri", "saturday": "sat", "sunday": "sun"}
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}


def _is_daycode(s):
    return re.fullmatch(r"(?:M|Tu|W|Th|F|Sa|Su)+", s.strip()) is not None


def _daycode_days(s):
    return {_DAYCODE[t] for t in re.findall(r"M|Tu|W|Th|F|Sa|Su", s)}


def _text_days(text):
    t = text.lower()
    return {v for name, v in _DAYNAME.items() if name in t}


def _text_times(text):
    """Set of 'HH:MM' (24h) times found, parsing both 24h and 12h+am/pm."""
    out = set()
    for m in re.finditer(r"(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?", text, re.I):
        h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
        if ap.startswith("p") and h != 12:
            h += 12
        if ap.startswith("a") and h == 12:
            h = 0
        out.add(f"{h:02d}:{mi:02d}")
    return out


def _text_dates(text):
    """Set of (month, day) pairs found, parsing ISO and 'Month D'."""
    out = set()
    t = text.lower()
    for _y, mo, d in re.findall(r"(\d{4})-(\d{2})-(\d{2})", t):
        out.add((int(mo), int(d)))
    months = "|".join(_MONTHS)
    for mo, d in re.findall(rf"({months})\s+(\d{{1,2}})", t):
        out.add((_MONTHS[mo], int(d)))
    return out


def _is_course_code(s):
    return re.fullmatch(r"[A-Z]{2,4}\s?\d{1,3}[A-Z]?", s.strip()) is not None


# ----------------------------------------------------------------------------
# Fact matching
# ----------------------------------------------------------------------------
def fact_matched(fact, answer):
    fact = str(fact).strip()

    if _is_daycode(fact):
        return _daycode_days(fact).issubset(_text_days(answer))

    if re.fullmatch(r"\d{1,2}:\d{2}", fact):
        return fact in _text_times(answer)

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", fact) or re.fullmatch(
            rf"(?:{'|'.join(_MONTHS)})\s+\d{{1,2}}", fact.lower()):
        return bool(_text_dates(fact) & _text_dates(answer))

    if _is_course_code(fact):
        return re.search(rf"\b{re.escape(fact)}\b", answer, re.I) is not None

    if re.fullmatch(r"\d+", fact):  # bare number (e.g. credits) -> word-boundary
        return re.search(rf"\b{fact}\b", answer) is not None

    return fact.lower() in answer.lower()


def answer_correct(answer, facts):
    """Return list of MISSING facts ([] means fully correct)."""
    return [f for f in facts if not fact_matched(f, answer)]


def tool_selection_ok(tools_used, expected_tool):
    if expected_tool == "multi":
        return len(set(tools_used)) >= 2
    return expected_tool in tools_used


def context_recall_hit(search_catalog_output, gold_source):
    """All gold_source course codes appear in the retrieved passages."""
    return all(
        re.search(rf"\b{re.escape(code)}\b", search_catalog_output, re.I)
        for code in gold_source
    )
