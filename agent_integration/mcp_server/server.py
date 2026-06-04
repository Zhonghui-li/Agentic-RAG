"""MCP server exposing the UCSC course tools (Model Context Protocol).

This makes the *capability* — semantic+keyword search over the real UCSC CSE
catalog, plus structured schedule/calendar lookups — available to ANY MCP client
(Claude Desktop, Cursor, ...) so the client's own LLM can call them. The
vectorstore, data, and retrieval logic stay server-side; only the query action
is exposed.

It reuses the exact same functions the in-app agent uses (agents.slug_retrieval /
agents.slug_tools) — the value is the RAG over real data, MCP is just the standard
doorway to it.

Run (stdio transport, for Claude Desktop):
    OPENAI_API_KEY=... EMB_MODEL=text-embedding-3-small python mcp_server/server.py
"""
import os
import sys

# default to no reranker so a local Claude-Desktop launch stays light (no torch);
# set RERANK=1 for the higher-quality CrossEncoder reranking.
os.environ.setdefault("RERANK", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP
from agents.slug_retrieval import search_catalog as _search_catalog
from agents.slug_tools import (
    lookup_schedule as _lookup_schedule,
    get_academic_calendar as _get_academic_calendar,
)

mcp = FastMCP("UCSC Slug Advisor")


@mcp.tool()
def search_catalog(query: str) -> str:
    """Search the UC Santa Cruz CSE course catalog for course descriptions,
    prerequisites, credits, and requirements. Returns the most relevant course
    entries. Pass a topic (e.g. "machine learning") or a course code (e.g.
    "CSE 101"). For prerequisite chains, call again with the prerequisite course."""
    return _search_catalog(query)


@mcp.tool()
def lookup_schedule(course_code: str, term: str = "") -> str:
    """Look up when a UCSC CSE course is offered: term, meeting days/time,
    location, instructor, and seat availability. `course_code` like "CSE 142";
    `term` is optional (e.g. "Fall 2025") — omit it to see all terms offered."""
    return _lookup_schedule(course_code, term or None)


@mcp.tool()
def get_academic_calendar(term: str = "") -> str:
    """Get UC Santa Cruz academic-calendar dates for 2025-26 (instruction
    begins/ends, finals, quarter end, priority enrollment). `term` is optional
    (e.g. "Winter 2026") — omit it to get all quarters."""
    return _get_academic_calendar(term or None)


if __name__ == "__main__":
    mcp.run()  # stdio transport
