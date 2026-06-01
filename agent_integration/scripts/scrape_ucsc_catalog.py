"""Scrape the UC Santa Cruz General Catalog (catalog.ucsc.edu) for a department's
courses and emit structured JSON.

The catalog is server-rendered HTML. Each course is an ``<h2 class="course-name">``
header followed by sibling blocks (description, credits, instructor, requirements,
GE code) until the next course header.

Usage:
    python scrape_ucsc_catalog.py \
        --dept cse-computer-science-and-engineering \
        --out ../data-ucsc/cse_courses.json
"""
import os
import re
import json
import time
import argparse

import requests
from bs4 import BeautifulSoup

BASE = "https://catalog.ucsc.edu"
LISTING = "/en/current/general-catalog/courses/{dept}/"
HEADERS = {"User-Agent": "Mozilla/5.0 (UCSC Slug Advisor course-data fetcher)"}


def _clean(text: str) -> str:
    """Collapse whitespace; turn empty/comma-only strings into ''."""
    text = re.sub(r"\s+", " ", text or "").strip()
    # instructor fields are often just stray commas when staff is TBA
    if not re.sub(r"[,\s]", "", text):
        return ""
    return text


def _division_from_href(href: str) -> str:
    for div in ("lower-division", "upper-division", "graduate"):
        if div in (href or ""):
            return div
    return "unknown"


def _siblings_until_next_course(h2):
    """Yield element siblings after ``h2`` up to (excluding) the next course header."""
    sib = h2.next_sibling
    while sib is not None:
        name = getattr(sib, "name", None)
        if name == "h2" and "course-name" in (sib.get("class") or []):
            return
        if name:
            yield sib
        sib = sib.next_sibling


def parse_course(h2):
    """Build a structured course dict from a course-name <h2> and its siblings."""
    link = h2.find("a")
    href = link.get("href", "") if link else ""

    # code lives in <span>, title is the remaining header text
    span = h2.find("span")
    code = _clean(span.get_text()) if span else ""
    full = _clean(h2.get_text(" "))
    title = _clean(full[len(code):]) if code and full.startswith(code) else full

    description, credits, requirements, ge_code, instructor = "", "", "", "", ""
    for sib in _siblings_until_next_course(h2):
        classes = sib.get("class") or []
        if "desc" in classes and not description:
            description = _clean(sib.get_text(" "))
        elif "sc-credithours" in classes:
            cred = sib.find(class_="credits")
            credits = _clean(cred.get_text()) if cred else ""
        elif "extraFields" in classes:
            txt = _clean(sib.get_text(" "))
            requirements = re.sub(r"^Requirements\s*", "", txt).strip()
        elif "genEd" in classes:
            txt = _clean(sib.get_text(" "))
            ge_code = re.sub(r"^General Education Code\s*", "", txt).strip()
        elif "instructor" in classes:
            txt = _clean(sib.get_text(" "))
            instructor = re.sub(r"^Instructor\s*", "", txt).strip()

    return {
        "course_code": code,
        "title": title,
        "description": description,
        "credits": credits,
        "requirements": requirements,
        "ge_code": ge_code,
        "instructor": instructor,
        "division": _division_from_href(href),
        "url": BASE + href if href else "",
    }


def scrape_department(dept: str):
    url = BASE + LISTING.format(dept=dept)
    print(f"Fetching {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    headers = soup.find_all("h2", class_="course-name")
    print(f"Found {len(headers)} course-name blocks.")

    courses, seen = [], set()
    for h2 in headers:
        course = parse_course(h2)
        code = course["course_code"]
        if not code or not course["description"]:
            continue
        if code in seen:  # cross-listed duplicates
            continue
        seen.add(code)
        courses.append(course)
    return courses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="cse-computer-science-and-engineering")
    ap.add_argument("--out", default="../data-ucsc/cse_courses.json")
    args = ap.parse_args()

    courses = scrape_department(args.dept)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(courses, f, indent=2, ensure_ascii=False)

    by_div = {}
    for c in courses:
        by_div[c["division"]] = by_div.get(c["division"], 0) + 1
    print(f"\nSaved {len(courses)} unique courses to {args.out}")
    print("By division:", by_div)
    print("With prerequisites:", sum(1 for c in courses if c["requirements"]))
    print("\nSample:")
    for c in courses[:3]:
        print(f"  {c['course_code']} — {c['title']} ({c['credits']} cr) [{c['division']}]")
        if c["requirements"]:
            print(f"     req: {c['requirements'][:80]}")


if __name__ == "__main__":
    main()
