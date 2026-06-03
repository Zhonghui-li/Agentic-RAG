"""Minimal FastAPI service for the UCSC Slug Advisor agent — a clickable live
demo that exposes the agent over HTTP and surfaces its tool-call trace.

Run (from agent_integration/, with OPENAI_API_KEY + EMB_MODEL set):
    uvicorn slug_service.app:app --reload --port 8100
Then open http://localhost:8100
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.slug_advisor_agent import build_agent, run_advisor

_STATIC = os.path.join(os.path.dirname(__file__), "static")
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = build_agent()  # loads vectorstore + reranker once, on startup
    yield


app = FastAPI(title="UCSC Slug Advisor", lifespan=lifespan)


class Query(BaseModel):
    question: str


@app.post("/advisor")
def advisor(q: Query):
    # sync endpoint -> FastAPI runs it in a threadpool, so the blocking
    # LangGraph/LLM calls don't fight the event loop (the old service's async bug).
    return run_advisor(q.question, agent=_agent)


@app.get("/health")
def health():
    return {"status": "ok", "agent_loaded": _agent is not None}


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))
