"""Optional Langfuse tracing for the agent (LLM observability).

Active only when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set; otherwise a
no-op, so the agent runs unchanged without an account.

Each agent run is logged as one "agent" observation: question -> answer, plus
tools used, latency, and token usage (for cost). A per-LLM-call breakdown would
use Langfuse's LangChain callback handler, currently blocked by a
langchain / langchain-core version skew, so we trace at the run level.
"""
import os

LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)


def sum_usage(messages):
    """Sum input/output tokens across the run's messages (for cost tracking)."""
    inp = out = 0
    for m in messages:
        u = getattr(m, "usage_metadata", None) or {}
        inp += u.get("input_tokens", 0)
        out += u.get("output_tokens", 0)
    return {"input": inp, "output": out} if (inp or out) else None


def log_trace(question, answer, tools_used, latency_s, usage=None):
    """Record one agent run to Langfuse. No-op unless keys are configured."""
    if not LANGFUSE_ENABLED:
        return
    try:
        from langfuse import get_client
        lf = get_client()
        with lf.start_as_current_observation(
            name="slug-advisor",
            as_type="agent",
            input=question,
            output=answer,
            metadata={
                "tools_used": tools_used,
                "n_tool_calls": len(tools_used or []),
                "latency_s": latency_s,
                "usage": usage,
            },
        ):
            lf.set_current_trace_io(input=question, output=answer)
        lf.flush()
    except Exception as e:  # tracing must never break the agent
        print(f"[langfuse] trace skipped: {type(e).__name__}: {e}")
