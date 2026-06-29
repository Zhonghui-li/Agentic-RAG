"""Optional Langfuse tracing for the agent (LLM observability).

Active only when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set; otherwise a
no-op, so the agent runs unchanged without an account.

`trace_agent(question)` is a context manager that WRAPS the agent run, so the
span's duration is the real latency. Inside, call the yielded `record(...)` to
attach the answer / tools / token usage. A per-LLM-call breakdown would use
Langfuse's LangChain callback handler, currently blocked by a langchain /
langchain-core version skew, so we trace at the run level.
"""
import os
from contextlib import contextmanager

LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)
_MODEL = os.environ.get("GEN_LLM_MODEL", "gpt-4o-mini")


def sum_usage(messages):
    """Sum input/output tokens across the run's messages (for cost tracking)."""
    inp = out = 0
    for m in messages:
        u = getattr(m, "usage_metadata", None) or {}
        inp += u.get("input_tokens", 0)
        out += u.get("output_tokens", 0)
    return {"input": inp, "output": out} if (inp or out) else None


def _noop(**_):
    return None


@contextmanager
def trace_agent(question):
    """Wrap an agent run in a Langfuse span (no-op unless keys set). Yields a
    `record(answer, tools_used, usage)` callback to set the span output."""
    if not LANGFUSE_ENABLED:
        yield _noop
        return
    try:
        from langfuse import get_client
        lf = get_client()
    except Exception as e:  # never break the agent
        print(f"[langfuse] disabled ({type(e).__name__}: {e})")
        yield _noop
        return

    # one "generation" observation wraps the run: its duration is the real
    # latency, and model + usage_details let Langfuse compute $ cost — both on
    # the same row (no separate child, so no apparent "duplication").
    with lf.start_as_current_observation(name="slug-advisor", as_type="generation", input=question):
        try:
            trace_id = lf.get_current_trace_id()   # so user feedback can be scored onto this trace
        except Exception:
            trace_id = None

        def record(answer=None, tools_used=None, usage=None, contexts=None):
            try:
                lf.update_current_generation(
                    output=answer,
                    model=_MODEL,
                    usage_details=({"input": usage["input"], "output": usage["output"]}
                                   if usage else None),
                    # contexts stored so the offline scorer (eval/score_traces.py)
                    # can grade faithfulness against what the tools returned
                    metadata={"tools_used": tools_used,
                              "n_tool_calls": len(tools_used or []),
                              "contexts": contexts},
                )
                lf.set_current_trace_io(input=question, output=answer)
            except Exception as e:
                print(f"[langfuse] record skipped: {e}")
            return trace_id
        yield record
    try:
        lf.flush()
    except Exception:
        pass
