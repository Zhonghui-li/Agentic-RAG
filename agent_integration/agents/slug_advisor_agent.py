"""UCSC Slug Advisor — a tool-calling agent (LangGraph ReAct).

Outer orchestrator that decides which tool to call for a student's question:

  - search_catalog        : course descriptions / prerequisites / credits (RAG)
  - lookup_schedule       : when/where a course meets, seat availability
  - get_academic_calendar : quarter dates, enrollment dates, date math
  - web_search            : fallback for out-of-catalog questions

The agent does the final answer composition (it reads tool outputs and writes
the answer). Multi-hop prerequisite questions are handled by calling
search_catalog repeatedly.
"""
import os
from typing import Dict, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

# Safety valve: cap how many agent<->tool super-steps a single question may take
# (one tool-call round ≈ 2 super-steps). Prevents runaway loops / cost blowups.
DEFAULT_RECURSION_LIMIT = 15

from agents.slug_retrieval import search_catalog as _search_catalog
from agents.slug_tools import (
    lookup_schedule as _lookup_schedule,
    get_academic_calendar as _get_academic_calendar,
    web_search as _web_search,
    _SCHEDULE,
)

_TERMS = ", ".join(_SCHEDULE["term_list"])

# Wrap the plain functions as LangChain tools (name + docstring -> tool schema).
TOOLS = [
    tool(_search_catalog),
    tool(_lookup_schedule),
    tool(_get_academic_calendar),
    tool(_web_search),
]

SYSTEM_PROMPT = """You are Slug Advisor, a UC Santa Cruz student assistant. Your scope \
is defined by your tools (below), not a fixed topic — currently that is CSE course \
advising. As tools are added your scope grows; you don't need new rules for that.

Use the tools to answer; do not rely on memory for facts:
- search_catalog: course content, prerequisites, credits, requirements.
- lookup_schedule: whether/when a course is offered, meeting time, seats.
- get_academic_calendar: quarter start/end, finals, enrollment dates, date math.
- web_search: ONLY when the catalog/schedule/calendar tools cannot answer.

Guidelines:
- STAY IN SCOPE — scope = what your tools can answer (not a hardcoded topic):
  - About UCSC and a tool can help → use it.
  - About UCSC but no tool covers it → say you don't have that and point them to
the right office (advising, registrar, admissions).
  - Unrelated to being a UCSC student (general chit-chat, writing/coding tasks,
other universities, personal opinions) → politely decline in one sentence and
remind them what you can help with. Never answer off-topic requests (e.g. "write
me a poem"), even if you could.
- For prerequisite-chain questions, call search_catalog repeatedly (e.g. look up \
the course, see its prereqs, then look those up).
- When a question needs both course info and scheduling, FIRST find the exact \
course code with search_catalog, THEN pass that exact code to lookup_schedule. \
Do not guess course codes.
- The schedule/calendar only cover these terms: {terms}. Map relative terms like \
"this fall" to the matching term in that list; never use a term outside it.
- Combine results from multiple tools into one clear answer.
- Always cite course codes (e.g. "CSE 101"). Be concise.
- If a tool says something isn't found, say so honestly; don't invent facts.
- DO NOT FABRICATE when something is outside the catalog/schedule/calendar data \
(which only cover course descriptions, prerequisites, credits, meeting times, \
seats, and academic dates). Handle two cases differently:
  - Factual info about the university that's simply elsewhere (department staff/\
chair, admission/acceptance rates, news): USE web_search to try to answer.
  - Policy (e.g. pass/no-pass grading), subjective opinions (course difficulty/\
quality), or totals you can't compute from per-course lookups (e.g. "how many \
courses are offered"): do NOT web_search and do NOT guess — say the course \
catalog doesn't cover it and point them to the right office (advising, registrar, \
admissions). Never invent a yes/no, a number, a rate, or a policy.""".format(terms=_TERMS)


def build_agent(model: str = None, temperature: float = 0.0):
    model = model or os.environ.get("GEN_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=temperature)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def _extract_trace(messages) -> List[Dict]:
    """Pull the tool-call trace (tool name + args) out of the message list."""
    trace = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            trace.append({"tool": tc["name"], "args": tc["args"]})
    return trace


def run_advisor(question: str, agent=None, verbose: bool = False,
                recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> Dict:
    """Run the agent on a question. Returns {answer, trace, tools_used}."""
    agent = agent or build_agent()
    try:
        result = agent.invoke(
            {"messages": [("user", question)]},
            {"recursion_limit": recursion_limit},
        )
        messages = result["messages"]
        answer = messages[-1].content
        trace = _extract_trace(messages)
    except GraphRecursionError:
        answer = ("I wasn't able to resolve this within the step limit. "
                  "Please try rephrasing or narrowing your question.")
        trace = []

    if verbose:
        for step in trace:
            print(f"  🔧 {step['tool']}({step['args']})")

    return {
        "answer": answer,
        "trace": trace,
        "tools_used": [t["tool"] for t in trace],
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the prerequisites for CSE 142?"
    print(f"Q: {q}\n")
    out = run_advisor(q, verbose=True)
    print("\nA:", out["answer"])
