"""Generate a grounded eval set for the UCSC Slug Advisor agent.

Questions are derived directly from the real scraped data, so gold answers are
guaranteed correct. Each item is tagged with the tool the agent *should* call,
which lets the eval harness score both answer correctness and tool-selection
accuracy.

Output: data-ucsc/ucsc_eval.jsonl
Schema per line:
  {
    "id", "question", "category",
    "expected_tool",        # search_catalog | lookup_schedule | get_academic_calendar | multi
    "answer_facts": [...],  # substrings that must appear in a correct answer
    "gold_source": [...],   # course code(s) that are the gold retrieval source for
                            # this question; enables true context-recall scoring in
                            # the eval harness. Empty for calendar-only questions.
  }
"""
import os
import re
import json
import argparse

# A readable, stable subset of well-known CSE courses to build questions around.
FOCUS = ["CSE 3", "CSE 20", "CSE 30", "CSE 12", "CSE 13S", "CSE 16",
         "CSE 101", "CSE 102", "CSE 107", "CSE 120", "CSE 130", "CSE 142"]


def _norm_prereq_codes(requirements: str):
    """Pull course codes out of a requirements string for fact-checking."""
    return sorted(set(re.findall(r"[A-Z]{2,4}\s?\d{1,3}[A-Z]?", requirements or "")))


def gen_course_content(courses):
    items = []
    for code in FOCUS:
        c = next((x for x in courses if x["course_code"] == code), None)
        if not c:
            continue
        items.append({
            "question": f"How many credits is {code}?",
            "category": "course_content",
            "expected_tool": "search_catalog",
            "answer_facts": [c["credits"]],
            "gold_source": [code],
        })
    return items


def gen_prerequisites(courses):
    items = []
    for code in FOCUS:
        c = next((x for x in courses if x["course_code"] == code), None)
        if not c or not c["requirements"]:
            continue
        codes = _norm_prereq_codes(c["requirements"])
        if not codes:
            continue
        items.append({
            "question": f"What are the prerequisites for {code}?",
            "category": "prerequisites",
            "expected_tool": "search_catalog",
            # require at least the first 2 prereq codes to appear
            "answer_facts": codes[:2],
            "gold_source": [code],
        })
    return items


def gen_schedule(schedule):
    items, seen = [], set()
    for o in schedule["offerings"]:
        code = o["course_code"]
        if code not in FOCUS or code in seen:
            continue
        seen.add(code)
        items.append({
            "question": f"Is {code} offered in {o['term']}, and what time does it meet?",
            "category": "schedule",
            "expected_tool": "lookup_schedule",
            "answer_facts": [o["start_time"], o["days"]],
            "gold_source": [code],
        })
    return items


def gen_calendar(calendar):
    items = []
    q = calendar["quarters"]
    # Dates use natural "Month D" form; the grader canonicalizes dates, so it
    # also matches ISO ("2026-01-05") surface forms in the answer.
    items.append({
        "question": "When does Winter 2026 instruction begin?",
        "category": "calendar",
        "expected_tool": "get_academic_calendar",
        "answer_facts": ["January 5"],
        "gold_source": [],
    })
    items.append({
        "question": "When are Fall 2025 final exams?",
        "category": "calendar",
        "expected_tool": "get_academic_calendar",
        "answer_facts": ["December 8", "December 12"],
        "gold_source": [],
    })
    items.append({
        "question": "When does priority enrollment for Spring 2026 begin?",
        "category": "calendar",
        "expected_tool": "get_academic_calendar",
        "answer_facts": ["February 26"],
        "gold_source": [],
    })
    return items


# Hand-written natural-language / multi-tool questions.
HANDWRITTEN = [
    {"question": "I want to learn machine learning. Which CSE course should I take and what are its prerequisites?",
     "category": "course_content", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 142", "CSE 101"], "gold_source": ["CSE 142"]},
    {"question": "What's the prerequisite for the machine learning course, and is it offered this fall?",
     "category": "multi_tool", "expected_tool": "multi",
     "answer_facts": ["CSE 101"], "gold_source": ["CSE 142"]},
    {"question": "Which lower-division course introduces programming in Python?",
     "category": "course_content", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 20"], "gold_source": ["CSE 20"]},
    {"question": "Does CSE 130 require operating systems background? What are its prerequisites?",
     "category": "prerequisites", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 101"], "gold_source": ["CSE 130"]},
    {"question": "How long is winter break between Fall 2025 and Winter 2026 instruction?",
     "category": "calendar", "expected_tool": "get_academic_calendar",
     "answer_facts": ["December 12", "January 5"], "gold_source": []},
    {"question": "Is there a machine learning course at the graduate level?",
     "category": "course_content", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 242"], "gold_source": ["CSE 242"]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="../data-ucsc")
    ap.add_argument("--out", default="../data-ucsc/ucsc_eval.jsonl")
    args = ap.parse_args()

    courses = json.load(open(os.path.join(args.datadir, "cse_courses.json")))
    schedule = json.load(open(os.path.join(args.datadir, "schedule.json")))
    calendar = json.load(open(os.path.join(args.datadir, "calendar.json")))

    items = []
    items += gen_course_content(courses)
    items += gen_prerequisites(courses)
    items += gen_schedule(schedule)
    items += gen_calendar(calendar)
    items += HANDWRITTEN

    for i, it in enumerate(items):
        it["id"] = f"ucsc_{i:03d}"

    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    by_cat, by_tool = {}, {}
    for it in items:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
        by_tool[it["expected_tool"]] = by_tool.get(it["expected_tool"], 0) + 1
    print(f"Wrote {len(items)} eval items to {args.out}")
    print("By category:", by_cat)
    print("By expected_tool:", by_tool)


if __name__ == "__main__":
    main()
