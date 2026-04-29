"""
recover_from_traj.py — 从个人轨迹文件恢复 ERROR 条目

对 traj.jsonl 里 answer=='ERROR' 的题目，尝试从对应的 <qid>.jsonl 轨迹文件
里提取第一次尝试的 final_answer 和 eval 指标，重新写入 traj.jsonl。

用法:
  python scripts/recover_from_traj.py \
      --run_dir runs/trajectories_4omini_oracle_rl_500
"""

import os, sys, json, argparse, re, string
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.evaluate_dataset_real import compute_and_write_stats


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

def _token_f1(pred: str, gt: str) -> float:
    p, g = _normalize_answer(pred).split(), _normalize_answer(gt).split()
    if not p or not g:
        return float(p == g)
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec  = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)

def _exact_match(pred: str, gt: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gt))

def _num(x, default=0.0) -> float:
    try:
        return float(x if not isinstance(x, list) else x[0])
    except Exception:
        return float(default)


def recover(run_dir: str):
    traj_path = os.path.join(run_dir, "traj.jsonl")
    rows = [json.loads(l) for l in open(traj_path, encoding="utf-8") if l.strip()]

    recovered = 0
    still_error = 0

    for row in rows:
        if row.get("answer") != "ERROR":
            continue

        qid = row.get("qid", "")
        individual_path = os.path.join(run_dir, f"{qid}.jsonl")

        if not os.path.exists(individual_path):
            print(f"  [skip] {qid[:8]}: no traj file")
            still_error += 1
            continue

        traj_entries = [json.loads(l) for l in open(individual_path, encoding="utf-8") if l.strip()]
        if not traj_entries:
            print(f"  [skip] {qid[:8]}: empty traj file")
            still_error += 1
            continue

        # 取第一次尝试的数据
        first = traj_entries[0]
        answer = first.get("final_answer", "")
        ev = first.get("eval", {})

        ref_str = row.get("reference", "")
        if isinstance(ref_str, list):
            ref_str = ref_str[0] if ref_str else ""

        # 恢复 eval 指标
        row["answer"] = answer
        row["faithfulness"]        = _num(ev.get("faith", ev.get("faithfulness", 0.0)))
        row["response_relevancy"]  = _num(ev.get("response_relevancy", 0.0))
        row["noise_sensitivity"]   = _num(ev.get("noise_sensitivity", 1.0))
        row["semantic_f1"]         = _num(ev.get("semantic_f1", ev.get("semantic_f1_score", 0.0)))

        # ctxP / ctxR: 从 state 里提取（轨迹末尾可能有）
        row["context_precision"]   = _num(first.get("context_precision", row.get("context_precision", 0.0)))
        row["context_recall"]      = _num(first.get("context_recall",    row.get("context_recall",    0.0)))

        # 重新计算 EM / offF1
        row["em"]         = _exact_match(answer, ref_str) if ref_str else 0.0
        row["official_f1"] = _token_f1(answer, ref_str) if ref_str else 0.0

        print(f"  [ok] {qid[:8]}: answer={answer!r:.40} EM={row['em']:.0f} F1={row['official_f1']:.2f}")
        recovered += 1

    print(f"\n[recover] {recovered} recovered, {still_error} still ERROR (no traj file)")

    # 重写 traj.jsonl
    with open(traj_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[recover] traj.jsonl rewritten with {len(rows)} entries")

    # 重新计算 summary_stats.csv
    compute_and_write_stats(rows, run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.run_dir, "traj.jsonl")):
        print(f"[error] traj.jsonl not found: {args.run_dir}")
        sys.exit(1)

    recover(args.run_dir)


if __name__ == "__main__":
    main()
