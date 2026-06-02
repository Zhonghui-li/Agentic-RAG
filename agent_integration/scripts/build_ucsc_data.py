"""Build the structured data files that the agent's tools query:

  - calendar.json : real UCSC 2025-26 academic calendar (from registrar.ucsc.edu)
  - schedule.json : per-term class offerings.

NOTE on schedule.json: the live Schedule of Classes (pisa.ucsc.edu) is an
ASP.NET form-post system that changes every term and is not stably scrapable.
To keep the `lookup_schedule` tool functional and deterministic, offerings here
are *synthesized from the real scraped catalog courses* (meeting days/times,
room, enrollment). This is representative data, flagged via the "_note" field,
not live registrar data. The course identities, titles, and instructors are real.
"""
import os
import re
import json
import random
import argparse

# Real UCSC 2025-26 academic calendar (registrar.ucsc.edu/calendar)
CALENDAR = {
    "academic_year": "2025-26",
    "_source": "registrar.ucsc.edu/calendar",
    "quarters": {
        "Fall 2025": {
            "instruction_begins": "2025-09-25",
            "instruction_ends": "2025-12-05",
            "final_exams": "2025-12-08/2025-12-12",
            "quarter_ends": "2025-12-12",
            "priority_enrollment_begins": "2025-05-19",
        },
        "Winter 2026": {
            "instruction_begins": "2026-01-05",
            "instruction_ends": "2026-03-13",
            "final_exams": "2026-03-16/2026-03-20",
            "quarter_ends": "2026-03-20",
            "priority_enrollment_begins": "2025-11-13",
        },
        "Spring 2026": {
            "instruction_begins": "2026-03-30",
            "instruction_ends": "2026-06-05",
            "final_exams": "2026-06-08/2026-06-11",
            "quarter_ends": "2026-06-11",
            "priority_enrollment_begins": "2026-02-26",
        },
    },
}

TERMS = ["Fall 2025", "Winter 2026", "Spring 2026"]
DAY_PATTERNS = ["MWF", "TuTh", "MW", "TuTh", "MWF"]
TIME_SLOTS = [
    ("09:20", "10:25"), ("10:40", "11:45"), ("12:00", "13:05"),
    ("13:20", "14:25"), ("14:40", "15:45"), ("16:00", "17:05"),
]
ROOMS = [
    "Engineering 2 180", "Baskin Engr 152", "Physical Sciences 110",
    "Thimann Lecture 003", "Social Sciences 2 075", "Engineering 2 192",
]


def synthesize_schedule(courses, seed=42):
    rng = random.Random(seed)
    offerings = []
    for c in courses:
        code = c["course_code"]
        # Each course is offered in 1-2 terms (grad courses less often).
        n_terms = 1 if c["division"] == "graduate" else rng.choice([1, 1, 2])
        terms = rng.sample(TERMS, n_terms)
        for term in terms:
            cap = rng.choice([30, 40, 60, 90, 120, 150])
            enrolled = rng.randint(int(cap * 0.4), int(cap * 1.1))
            enrolled = min(enrolled, int(cap * 1.1))
            if enrolled >= cap:
                status = "Wait List" if enrolled <= cap + 8 else "Closed"
            else:
                status = "Open"
            start, end = rng.choice(TIME_SLOTS)
            # catalog instructor is sometimes comma-junk (", , ,") when TBA
            raw_instr = c.get("instructor") or ""
            instructor = raw_instr if re.sub(r"[,\s]", "", raw_instr) else "Staff"
            offerings.append({
                "course_code": code,
                "title": c["title"],
                "term": term,
                "days": rng.choice(DAY_PATTERNS),
                "start_time": start,
                "end_time": end,
                "location": rng.choice(ROOMS),
                "instructor": instructor if instructor else "Staff",
                "credits": c["credits"],
                "capacity": cap,
                "enrolled": min(enrolled, cap),
                "waitlisted": max(0, enrolled - cap),
                "status": status,
            })
    return {
        "_note": "Representative offerings synthesized from real catalog courses; "
                 "not live pisa.ucsc.edu data. Course identities/titles are real.",
        "term_list": TERMS,
        "offerings": offerings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", default="../data-ucsc/cse_courses.json")
    ap.add_argument("--outdir", default="../data-ucsc")
    args = ap.parse_args()

    with open(args.courses) as f:
        courses = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    cal_path = os.path.join(args.outdir, "calendar.json")
    with open(cal_path, "w") as f:
        json.dump(CALENDAR, f, indent=2)
    print(f"Wrote {cal_path}")

    schedule = synthesize_schedule(courses)
    sch_path = os.path.join(args.outdir, "schedule.json")
    with open(sch_path, "w") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)
    print(f"Wrote {sch_path}: {len(schedule['offerings'])} offerings "
          f"across {len(courses)} courses")
    # quick sanity summary
    by_term = {}
    for o in schedule["offerings"]:
        by_term[o["term"]] = by_term.get(o["term"], 0) + 1
    print("Offerings by term:", by_term)


if __name__ == "__main__":
    main()
