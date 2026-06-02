"""Run the Slug Advisor agent over the eval set and score three deterministic
metrics: answer correctness, tool-selection accuracy, context recall.

Writes a metrics baseline (eval/baseline.json) and per-item results
(eval/results.json). This is the P2 Layer-2 evaluation (it runs the full
agent, so it costs API calls — meant for nightly/on-demand, not every PR).

Usage (from agent_integration/, with OPENAI_API_KEY + EMB_MODEL set):
    python -m eval.run_eval
"""
import sys
import json
import time
import argparse

from langchain_core.messages import ToolMessage
from agents.slug_advisor_agent import build_agent
from eval.graders import answer_correct, tool_selection_ok, context_recall_hit


def run_one(agent, question):
    res = agent.invoke({"messages": [("user", question)]}, {"recursion_limit": 15})
    msgs = res["messages"]
    answer = msgs[-1].content
    tools = [tc["name"] for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
    sc_out = " ".join(
        m.content for m in msgs
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "search_catalog"
    )
    return answer, tools, sc_out


def pct(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data-ucsc/ucsc_eval.jsonl")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--baseline", default="eval/baseline.json")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.eval)]
    agent = build_agent()
    rows = []
    t0 = time.time()

    for i, it in enumerate(items):
        try:
            answer, tools, sc_out = run_one(agent, it["question"])
            facts = it.get("answer_facts") or []
            # empty answer_facts (e.g. out-of-scope/web_search items) -> answer N/A
            missing = answer_correct(answer, facts) if facts else []
            a_ok = (not missing) if facts else None
            t_ok = tool_selection_ok(tools, it["expected_tool"])
            gs = it.get("gold_source") or []
            c_ok = context_recall_hit(sc_out, gs) if (gs and "search_catalog" in tools) else None
            rows.append({"id": it["id"], "category": it["category"], "question": it["question"],
                         "ans": a_ok, "tool": t_ok, "ctx": c_ok, "missing": missing,
                         "tools": tools, "answer": answer})
        except Exception as e:
            rows.append({"id": it["id"], "category": it["category"], "question": it["question"],
                         "ans": False, "tool": False, "ctx": None, "_err": f"{type(e).__name__}: {e}"})
        r = rows[-1]
        failed = (r.get("ans") is False) or (r.get("tool") is False)
        print(f"[{i+1}/{len(items)}] {r['id']} ans={r['ans']} tool={r['tool']} ctx={r['ctx']}"
              + ("  <-- FAIL" if failed else ""))

    elapsed = round(time.time() - t0, 1)
    metrics = {
        "answer_correctness": pct([r["ans"] for r in rows]),
        "tool_selection_accuracy": pct([r["tool"] for r in rows]),
        "context_recall": pct([r["ctx"] for r in rows]),
        "n": len(rows), "elapsed_s": elapsed,
    }
    by_cat = {}
    for c in sorted(set(r["category"] for r in rows)):
        sub = [r for r in rows if r["category"] == c]
        by_cat[c] = {"n": len(sub), "answer": pct([r["ans"] for r in sub]),
                     "tool": pct([r["tool"] for r in sub])}

    json.dump({"metrics": metrics, "by_category": by_cat, "rows": rows},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    json.dump(metrics, open(args.baseline, "w"), indent=2)

    print(f"\n=== METRICS ({elapsed}s) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\n=== BY CATEGORY (answer / tool) ===")
    for c, m in by_cat.items():
        print(f"  {c:14} n={m['n']:2}  answer={m['answer']}%  tool={m['tool']}%")

    fails = [r for r in rows if (r.get("ans") is False) or (r.get("tool") is False)]
    if fails:
        print(f"\n=== {len(fails)} FAILURES ===")
        for r in fails:
            print(f"- {r['id']} [{r['category']}] {r['question']}")
            print(f"    missing={r.get('missing')} tools={r.get('tools')} err={r.get('_err','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
