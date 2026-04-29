"""
retry_errors.py — 对 traj.jsonl 里 answer=='ERROR' 的题目重新跑一遍

用法:
  # IRCoT baseline
  python scripts/retry_errors.py \
      --run_dir runs/trajectories_4omini_ircot_500 \
      --use_router 0 --use_adaptive_retrieval 0

  # Oracle RL Router
  python scripts/retry_errors.py \
      --run_dir runs/trajectories_4omini_oracle_rl_500 \
      --use_router 1 --use_adaptive_retrieval 1

跑完之后 traj.jsonl 会多出 retry 的条目（同一题可能有两条）。
用 --merge_only 可以单独执行去重并重新计算 summary_stats.csv，不重新跑模型。
"""

import os, sys, json, argparse, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.evaluate_dataset_real import compute_and_write_stats


def find_errors(traj_path: str):
    """返回 traj.jsonl 里所有 answer=='ERROR' 的条目。"""
    errors = []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("answer") == "ERROR":
                errors.append(row)
    return errors


def dedup_and_recompute(run_dir: str):
    """
    对 traj.jsonl 去重：同一 qid 有多条时，
      - 优先保留最后一条非 ERROR 的
      - 若全是 ERROR，保留最后一条
    然后重写 summary_stats.csv。
    """
    traj_path = os.path.join(run_dir, "traj.jsonl")
    rows_by_qid = {}          # qid -> 最佳row
    order = []                # 保留原始顺序（first-seen）

    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("qid", row.get("question", ""))

            if qid not in rows_by_qid:
                order.append(qid)

            existing = rows_by_qid.get(qid)
            if existing is None:
                rows_by_qid[qid] = row
            elif existing.get("answer") == "ERROR" and row.get("answer") != "ERROR":
                # 新的成功了，替换
                rows_by_qid[qid] = row
            elif row.get("answer") != "ERROR":
                # 两条都成功，取更新的
                rows_by_qid[qid] = row
            # 否则保留已有的（旧的成功 / 新旧都失败保留旧的）

    deduped = [rows_by_qid[qid] for qid in order]

    errors_remaining = sum(1 for r in deduped if r.get("answer") == "ERROR")
    print(f"[dedup] {len(deduped)} unique questions, {errors_remaining} still ERROR after retry")

    # 重写 traj.jsonl（去重版）
    with open(traj_path, "w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[dedup] traj.jsonl rewritten with {len(deduped)} entries")

    # 重新计算 summary_stats.csv
    compute_and_write_stats(deduped, run_dir)
    return deduped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. runs/trajectories_4omini_oracle_rl_500")
    ap.add_argument("--use_router", type=int, default=0)
    ap.add_argument("--use_adaptive_retrieval", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--gen_max_attempts", type=int, default=2)
    ap.add_argument("--merge_only", action="store_true",
                    help="只做去重+重新计算stats，不重新跑模型")
    args = ap.parse_args()

    traj_path = os.path.join(args.run_dir, "traj.jsonl")
    if not os.path.exists(traj_path):
        print(f"[error] traj.jsonl not found: {traj_path}")
        sys.exit(1)

    if args.merge_only:
        print("[merge_only] 跳过retry，直接去重+重算stats")
        dedup_and_recompute(args.run_dir)
        return

    # --- 找出 ERROR 条目 ---
    errors = find_errors(traj_path)
    if not errors:
        print("[retry] 没有 ERROR 条目，无需 retry。")
        dedup_and_recompute(args.run_dir)
        return

    print(f"[retry] 找到 {len(errors)} 条 ERROR，准备重新跑")
    for e in errors:
        print(f"  - {e.get('question', '')[:70]}")

    # --- 生成临时 dataset ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        for e in errors:
            tmp.write(json.dumps({
                "question": e["question"],
                "reference": e.get("reference") or e.get("answer", ""),
                "qid": e["qid"],
            }, ensure_ascii=False) + "\n")
        tmp_path = tmp.name
    print(f"[retry] 临时 dataset 写入: {tmp_path}")

    # --- 调用 evaluate_dataset_real，结果 append 到同一 run_dir ---
    cmd = [
        sys.executable, "-m", "agents.evaluate_dataset_real",
        "--dataset", tmp_path,
        "--out_dir", args.run_dir,
        "--top_k", str(args.top_k),
        "--gen_max_attempts", str(args.gen_max_attempts),
        "--use_router", str(args.use_router),
        "--use_adaptive_retrieval", str(args.use_adaptive_retrieval),
    ]
    env = os.environ.copy()
    print(f"[retry] 运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)

    os.unlink(tmp_path)

    if result.returncode != 0:
        print(f"[retry] 子进程退出码 {result.returncode}，检查输出")
        sys.exit(result.returncode)

    print("\n[retry] 完成，开始去重+重算stats")
    dedup_and_recompute(args.run_dir)


if __name__ == "__main__":
    main()
