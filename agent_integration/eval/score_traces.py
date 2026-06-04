"""Offline scorer: the P2-eval <-> P3-observability loop.

Fetches recent Langfuse traces (real production runs), scores each with the eval
suite's reference-free Ragas metrics (faithfulness + answer relevancy), and pushes
the scores back to Langfuse. Run periodically (e.g. nightly) to monitor production
QUALITY over time — without adding judge-call latency/cost to user requests.

Reference-free on purpose: production questions have no gold labels, so we use the
metrics that need none (is the answer grounded in what the tools returned? is it
on-topic?). The deterministic graders (answer_facts / gold_source) stay in the
offline eval set, which does have gold.

Usage (with OPENAI_API_KEY + EMB_MODEL + LANGFUSE_* set):
    python -m eval.score_traces
"""
import argparse

from langfuse import get_client
from eval.quality import score_quality


def main(limit=50):
    lf = get_client()
    summaries = lf.api.trace.list(limit=limit).data

    items, ids = [], []
    for s in summaries:
        t = lf.api.trace.get(s.id)  # full detail (observations + scores)
        already = {sc.name for sc in (getattr(t, "scores", None) or [])}
        if "faithfulness" in already:
            continue
        contexts = None
        for o in (t.observations or []):
            md = getattr(o, "metadata", None) or {}
            if md.get("contexts"):
                contexts = md["contexts"]
                break
        if not contexts or not t.output:
            continue  # nothing to ground against (e.g. abstain answers)
        items.append({"question": t.input, "answer": t.output,
                      "contexts": contexts if isinstance(contexts, list) else [contexts]})
        ids.append(s.id)

    if not items:
        print("no new scorable traces.")
        return 0

    print(f"scoring {len(items)} traces (Ragas faithfulness + answer_relevancy)...")
    scores = score_quality(items)

    pushed = 0
    for tid, sc in zip(ids, scores):
        for name in ("faithfulness", "answer_relevancy"):
            if sc.get(name) is not None:
                lf.create_score(name=name, value=float(sc[name]),
                                trace_id=tid, data_type="NUMERIC")
                pushed += 1
    lf.flush()
    print(f"pushed {pushed} scores across {len(items)} traces -> Langfuse (Scores).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    raise SystemExit(main(ap.parse_args().limit))
