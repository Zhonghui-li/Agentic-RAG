"""Minimal FastAPI service for the UCSC Slug Advisor agent — a clickable live
demo that exposes the agent over HTTP and surfaces its tool-call trace.

Run (from agent_integration/, with OPENAI_API_KEY + EMB_MODEL set):
    uvicorn slug_service.app:app --reload --port 8100
Then open http://localhost:8100
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.slug_advisor_agent import build_agent, run_advisor
from slug_service.limits import (
    MAX_INPUT_CHARS, check_rate_limit, check_daily_quota,
)


def _client_ip(request: Request) -> str:
    # behind Cloud Run the real client IP is the first X-Forwarded-For entry
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

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
def advisor(q: Query, request: Request):
    # abuse/cost controls (OpenAI + GCP are shared lab accounts)
    if len(q.question) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"Question too long (max {MAX_INPUT_CHARS} characters).")
    if not check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Too many requests — please wait a minute and try again.")
    if not check_daily_quota():
        raise HTTPException(429, "This demo has reached its daily limit. Please try again tomorrow.")
    # sync endpoint -> FastAPI runs it in a threadpool, so the blocking
    # LangGraph/LLM calls don't fight the event loop (the old service's async bug).
    try:
        return run_advisor(q.question, agent=_agent)
    except Exception as e:
        # public testing: an LLM/timeout error should be a friendly message, not a 500
        print(f"[/advisor] agent error: {type(e).__name__}: {e}")
        return {"answer": "Sorry — I hit an error answering that. Please try again in a moment.",
                "trace": [], "tools_used": [], "trace_id": None}


class Feedback(BaseModel):
    trace_id: str
    value: int                       # 1 = thumbs up, 0 = thumbs down
    question: str | None = None      # carried so the score row is self-explanatory in Langfuse
    answer: str | None = None


@app.post("/feedback")
def feedback(fb: Feedback):
    """Capture a user's thumbs up/down on an answer as a Langfuse score (name=user_feedback),
    so community testing yields LABELED eval data, not just unlabeled traces. The question/answer
    are attached as the score's metadata so a reviewer can read what was rated."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return {"ok": False, "reason": "observability disabled"}
    try:
        from langfuse import get_client
        lf = get_client()
        meta = {k: v[:500] for k, v in (("question", fb.question), ("answer", fb.answer)) if v}
        lf.create_score(name="user_feedback", value=float(1 if fb.value >= 1 else 0),
                        trace_id=fb.trace_id, data_type="NUMERIC", metadata=(meta or None))
        lf.flush()
    except Exception as e:
        print(f"[/feedback] {type(e).__name__}: {e}")
        return {"ok": False}
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "agent_loaded": _agent is not None}


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))
