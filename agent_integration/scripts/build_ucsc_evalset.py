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

# For "abstain" items (policy / subjective / uncomputable): a correct answer
# honestly admits the catalog doesn't cover it and redirects. One OR-group =
# any honesty/redirect phrase suffices.
HONESTY = [["not in", "does not", "doesn't", "do not have", "don't have",
            "couldn't find", "could not find", "consult", "recommend checking",
            "isn't available", "not available", "advising", "registrar",
            "admissions", "no information"]]


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

# Harder, discriminating questions (verified against cse_courses.json). These
# stress the system where the easy templated set doesn't: questions that DON'T
# name the course (so retrieval must actually find it -> context recall becomes
# meaningful), prerequisite chains, comparisons, out-of-scope -> web_search, and
# date arithmetic. Empty answer_facts => answer graded N/A (tool-selection only).
HARD = [
    # --- find-the-course: no course code given -> real retrieval ---
    {"question": "Which course is an introduction to computer graphics?",
     "category": "find_course", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 160"], "gold_source": ["CSE 160"]},
    {"question": "Which lower-division course covers machine learning basics and data analysis?",
     "category": "find_course", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 40"], "gold_source": ["CSE 40"]},
    {"question": "Which upper-division course teaches computer architecture?",
     "category": "find_course", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 120"], "gold_source": ["CSE 120"]},
    {"question": "Which course teaches computer vision?",
     "category": "find_course", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 164"], "gold_source": ["CSE 164"]},
    {"question": "Which course covers deep learning?",
     "category": "find_course", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 144"], "gold_source": ["CSE 144"]},
    {"question": "Which course is about computer security?",
     "category": "find_course", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 132"], "gold_source": ["CSE 132"]},
    # --- comparison: retrieve two courses and compare ---
    {"question": "Which has more credits, CSE 13S or CSE 101?",
     "category": "comparison", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 13S"], "gold_source": ["CSE 13S", "CSE 101"]},
    {"question": "Which is worth more credits: CSE 160 or CSE 162?",
     "category": "comparison", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 160"], "gold_source": ["CSE 160", "CSE 162"]},
    # --- multi-hop prerequisite chain: repeated search_catalog ---
    {"question": "I've taken no CS courses yet. Trace the prerequisite chain back from CSE 142 to the earliest course I'd need.",
     "category": "multi_hop", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 101", "CSE 12"], "gold_source": ["CSE 142", "CSE 101"]},
    {"question": "What is the prerequisite of the prerequisite of CSE 130?",
     "category": "multi_hop", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 12"], "gold_source": ["CSE 130", "CSE 101"]},
    # --- out-of-scope -> should escalate to web_search (answer graded N/A) ---
    {"question": "Who is the current chair of the UCSC Computer Science and Engineering department?",
     "category": "out_of_scope", "expected_tool": "web_search",
     "answer_facts": [], "gold_source": []},
    {"question": "What is the acceptance rate for UCSC's computer science program?",
     "category": "out_of_scope", "expected_tool": "web_search",
     "answer_facts": [], "gold_source": []},
    # --- date arithmetic (Dec 5 2025 -> Jan 5 2026 = 31 days) ---
    {"question": "How many days are there between the end of Fall 2025 instruction and the start of Winter 2026 instruction?",
     "category": "date_math", "expected_tool": "get_academic_calendar",
     "answer_facts": ["31"], "gold_source": []},
    # --- negative / honesty: CSE 999 does not exist (answer graded N/A) ---
    {"question": "Is CSE 999 offered in Winter 2026?",
     "category": "negative", "expected_tool": "lookup_schedule",
     "answer_facts": [], "gold_source": []},
]

# Multi-hop questions generated by Ragas TestsetGenerator (eval/gen_ragas_testset.py),
# then HUMAN-VERIFIED against cse_courses.json and converted to answer_facts.
# (Of 12 raw generations, these 4 were kept; the rest were dropped for typo-injection
# or referencing deprecated course names like "CMPS 10".)
# Kept 1 of the 4 generated as a representative (the others were the same
# "prereqs of X, relate to Y" pattern -> redundant once DIVERSE filled coverage).
RAGAS_CURATED = [
    {"question": "What are the prerequisites for CSE 30, and how do they compare to the prerequisites for CSE 122?",
     "category": "multi_hop", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 20", "CSE 100"], "gold_source": ["CSE 30", "CSE 122"]},
]

# Coverage-matrix questions: deliberately fill empty cells (advising / eligibility /
# ambiguity / policy / aggregation / subjective) to make the eval REALISTIC. Several
# of these are expected to be hard or expose limitations (e.g. aggregation has no
# supporting tool) — that's the point: a realistic eval must have headroom.
# All answers verified against cse_courses.json / schedule.json.
DIVERSE = [
    # recommendation / advising (must surface a relevant course)
    # open-ended -> answer_facts use OR-groups (any valid course passes); no single
    # gold_source (several courses are legitimately relevant).
    {"question": "I want to specialize in artificial intelligence. Which CSE courses should I take?",
     "category": "recommendation", "expected_tool": "search_catalog",
     "answer_facts": [["CSE 140", "CSE 142", "CSE 144", "CSE 164"]], "gold_source": []},
    {"question": "I'm interested in cybersecurity. Which course should I take?",
     "category": "recommendation", "expected_tool": "search_catalog",
     "answer_facts": [["CSE 132", "CSE 108", "CSE 235", "CSE 233"]], "gold_source": []},
    # eligibility (reason about completed courses vs prerequisites)
    {"question": "I've completed CSE 12 and CSE 30. Am I eligible to enroll in CSE 101?",
     "category": "eligibility", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 16"], "gold_source": ["CSE 101"]},
    {"question": "I've finished CSE 20. Can I take CSE 30 next, or is something else required?",
     "category": "eligibility", "expected_tool": "search_catalog",
     "answer_facts": ["MATH"], "gold_source": ["CSE 30"]},
    # ambiguity (several courses match -> pick the right intro)
    {"question": "Which course should I take to start learning about databases?",
     "category": "ambiguity", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 180"], "gold_source": ["CSE 180"]},
    {"question": "I want an introductory algorithms course. Which one should I take?",
     "category": "ambiguity", "expected_tool": "search_catalog",
     "answer_facts": ["CSE 101"], "gold_source": ["CSE 101"]},
    # policy (waitlist is in schedule data; P/NP is NOT in our data -> escalate)
    {"question": "CSE 142 is full for Winter 2026. Is there a waitlist?",
     "category": "policy", "expected_tool": "lookup_schedule",
     "answer_facts": ["wait"], "gold_source": []},
    # policy -> abstain (catalog has no P/NP info; admit + redirect, don't web_search)
    {"question": "Can I take CSE 3 on a pass/no-pass (P/NP) basis?",
     "category": "policy", "expected_tool": "abstain",
     "answer_facts": HONESTY, "gold_source": []},
    # aggregation -> dual-track: GATE on honest abstain (current correct behavior,
    # no counting tool), but keep ideal_facts=["96"] as a non-gating DIAGNOSTIC so
    # the capability gap stays visible. When a counting tool is added, capability
    # coverage rises and ideal_facts can be promoted to answer_facts.
    {"question": "How many CSE courses are offered in Fall 2025?",
     "category": "aggregation", "expected_tool": "abstain",
     "answer_facts": HONESTY, "ideal_facts": ["96"],
     "capability_gap": "no aggregation/counting tool", "gold_source": []},
    # subjective -> abstain (catalog has no difficulty ratings; admit + redirect)
    {"question": "Do students consider CSE 142 a difficult course?",
     "category": "subjective", "expected_tool": "abstain",
     "answer_facts": HONESTY, "gold_source": []},
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
    items += HARD
    items += RAGAS_CURATED
    items += DIVERSE

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
