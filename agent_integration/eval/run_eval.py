"""Run the Slug Advisor agent over the eval set and score:
  - deterministic: answer correctness, tool-selection accuracy, context recall
  - (--quality) LLM-judged: Ragas faithfulness + answer relevancy

Writes a metrics baseline (eval/baseline.json) and per-item results
(eval/results.json). Runs the full agent -> costs API calls (P2 Layer-2;
nightly/on-demand, not every PR).

Per-category n is small (2-15), so per-category numbers are NOISY -> reported
as DIAGNOSTIC only. Gate/track on the coarser BUCKET and overall numbers, which
have enough samples to be stable.

Usage (from agent_integration/, with OPENAI_API_KEY + EMB_MODEL set):
    python -m eval.run_eval --quality
"""
import sys
import json
import time
import argparse

from langchain_core.messages import ToolMessage
from agents.slug_advisor_agent import build_agent
from eval.graders import answer_correct, tool_selection_ok, context_recall_hit

# Coarser buckets for stable reporting (fine categories kept as diagnostic tags).
BUCKETS = {
    "catalog_qa": {"course_content", "prerequisites", "find_course", "comparison",
                   "ambiguity", "multi_hop"},
    "advising": {"recommendation", "eligibility"},
    "scheduling_calendar": {"schedule", "multi_tool", "calendar", "date_math"},
    "escalation_honesty": {"out_of_scope", "subjective", "policy", "negative", "aggregation"},
}


def bucket_of(category):
    for b, cats in BUCKETS.items():
        if category in cats:
            return b
    return "other"


def run_one(agent, question):
    res = agent.invoke({"messages": [("user", question)]}, {"recursion_limit": 15})
    msgs = res["messages"]
    answer = msgs[-1].content
    tools = [tc["name"] for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    sc_out = " ".join(m.content for m in tool_msgs if getattr(m, "name", "") == "search_catalog")
    all_ctx = [m.content for m in tool_msgs]  # everything the tools returned
    return answer, tools, sc_out, all_ctx


def pct(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data-ucsc/ucsc_eval.jsonl")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--baseline", default="eval/baseline.json")
    ap.add_argument("--quality", action="store_true", help="also compute Ragas faithfulness/relevancy")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.eval)]
    agent = build_agent()
    rows = []
    t0 = time.time()

    for i, it in enumerate(items):
        try:
            answer, tools, sc_out, all_ctx = run_one(agent, it["question"])
            facts = it.get("answer_facts") or []
            missing = answer_correct(answer, facts) if facts else []
            a_ok = (not missing) if facts else None
            t_ok = tool_selection_ok(tools, it["expected_tool"])
            gs = it.get("gold_source") or []
            c_ok = context_recall_hit(sc_out, gs) if (gs and "search_catalog" in tools) else None
            rows.append({"id": it["id"], "category": it["category"], "bucket": bucket_of(it["category"]),
                         "question": it["question"], "ans": a_ok, "tool": t_ok, "ctx": c_ok,
                         "missing": missing, "tools": tools, "answer": answer, "contexts": all_ctx})
        except Exception as e:
            rows.append({"id": it["id"], "category": it["category"], "bucket": bucket_of(it["category"]),
                         "question": it["question"], "ans": False, "tool": False, "ctx": None,
                         "_err": f"{type(e).__name__}: {e}"})
        r = rows[-1]
        failed = (r.get("ans") is False) or (r.get("tool") is False)
        print(f"[{i+1}/{len(items)}] {r['id']} ans={r['ans']} tool={r['tool']} ctx={r['ctx']}"
              + ("  <-- FAIL" if failed else ""))

    # --- optional LLM-judged quality (Ragas faithfulness + relevancy) ---
    if args.quality:
        from eval.quality import score_quality
        idx = [j for j, r in enumerate(rows) if r.get("answer") and r.get("contexts")]
        q_items = [{"question": rows[j]["question"], "answer": rows[j]["answer"],
                    "contexts": rows[j]["contexts"]} for j in idx]
        print(f"\nScoring quality (Ragas) on {len(q_items)} items...")
        for j, sc in zip(idx, score_quality(q_items)):
            rows[j]["faithfulness"] = sc["faithfulness"]
            rows[j]["answer_relevancy"] = sc["answer_relevancy"]

    elapsed = round(time.time() - t0, 1)
    metrics = {
        "answer_correctness": pct([r.get("ans") for r in rows]),
        "tool_selection_accuracy": pct([r.get("tool") for r in rows]),
        "context_recall": pct([r.get("ctx") for r in rows]),
        "n": len(rows), "elapsed_s": elapsed,
    }
    if args.quality:
        metrics["faithfulness"] = mean([r.get("faithfulness") for r in rows])
        metrics["answer_relevancy"] = mean([r.get("answer_relevancy") for r in rows])

    by_bucket = {}
    for b in list(BUCKETS) + ["other"]:
        sub = [r for r in rows if r["bucket"] == b]
        if sub:
            by_bucket[b] = {"n": len(sub), "answer": pct([r.get("ans") for r in sub]),
                            "tool": pct([r.get("tool") for r in sub]),
                            "faithfulness": mean([r.get("faithfulness") for r in sub])}

    json.dump({"metrics": metrics, "by_bucket": by_bucket, "rows": rows},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    json.dump(metrics, open(args.baseline, "w"), indent=2)

    print(f"\n=== METRICS ({elapsed}s) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\n=== BY BUCKET (answer / tool / faithfulness) — gate/track here, not per-category ===")
    for b, m in by_bucket.items():
        print(f"  {b:20} n={m['n']:2}  answer={m['answer']}%  tool={m['tool']}%  faith={m['faithfulness']}")

    fails = [r for r in rows if (r.get("ans") is False) or (r.get("tool") is False)]
    low_faith = [r for r in rows if isinstance(r.get("faithfulness"), float) and r["faithfulness"] < 0.5]
    if fails:
        print(f"\n=== {len(fails)} FAILURES (deterministic) ===")
        for r in fails:
            print(f"- {r['id']} [{r['category']}] {r['question']}")
            print(f"    missing={r.get('missing')} tools={r.get('tools')} err={r.get('_err','')}")
    if low_faith:
        print(f"\n=== {len(low_faith)} LOW-FAITHFULNESS (possible hallucination) ===")
        for r in low_faith:
            print(f"- {r['id']} [{r['category']}] faith={r['faithfulness']} | {r['question']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
