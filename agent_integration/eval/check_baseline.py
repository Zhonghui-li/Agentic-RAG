"""L2 regression gate: compare the latest run (eval/results.json) against the
committed baseline (eval/baseline.json) and FAIL (exit 1) if any metric drops
more than its tolerance.

Tolerances reflect each metric's noise (LLM non-determinism ~2pp measured;
LLM-judged metrics drift more, so a wider band). This is REGRESSION detection
relative to baseline — not an absolute accuracy target.

Usage (after `python -m eval.run_eval --quality`):
    python -m eval.check_baseline
"""
import sys
import json
import argparse

# metric -> allowed drop below baseline before it counts as a regression
TOLERANCE = {
    "answer_correctness": 5.0,        # percentage points
    "tool_selection_accuracy": 5.0,
    "context_recall": 5.0,
    "faithfulness": 0.10,             # 0-1 scale, LLM-judged -> wider band
    "answer_relevancy": 0.10,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="eval/baseline.json")
    ap.add_argument("--results", default="eval/results.json")
    args = ap.parse_args()

    baseline = json.load(open(args.baseline))
    results = json.load(open(args.results))["metrics"]

    print(f"{'metric':26} {'baseline':>9} {'current':>9} {'min':>9}  status")
    regressions = []
    for metric, tol in TOLERANCE.items():
        b, c = baseline.get(metric), results.get(metric)
        if b is None or c is None:
            print(f"{metric:26} {str(b):>9} {str(c):>9} {'-':>9}  skip (missing)")
            continue
        floor = round(b - tol, 3)
        ok = c >= floor
        print(f"{metric:26} {b:>9} {c:>9} {floor:>9}  {'OK' if ok else 'REGRESSION'}")
        if not ok:
            regressions.append((metric, b, c, floor))

    if regressions:
        print(f"\nFAIL: {len(regressions)} metric(s) regressed beyond tolerance.")
        return 1
    print("\nPASS: no metric regressed beyond tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
