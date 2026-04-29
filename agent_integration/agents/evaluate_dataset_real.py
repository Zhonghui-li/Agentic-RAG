# ===== PASTE INTO agents/evaluate_dataset_real.py =====
import os, json, csv, uuid, argparse, traceback

# Load .env from project root so OPENAI_API_KEY is available
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
from typing import Any, Dict, List, Optional
import statistics as st
import math

import functools

from agents.langgraph_rag import run_rag_pipeline
from agents.retrieval_agent import RetrievalAgent
from agents.generation_agent import GenerationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reasoning_agent import ReasoningAgent
from agents.hybrid_retriever import HybridRetriever
from agents.reranker import create_cross_encoder_reranker
from agents.multi_query import generate_query_variants
from utils.trajectory_logger import TrajectoryLogger

# --- 用 FAISS 加载你的向量库 ---
def get_vectorstore() -> Any:
    """
    从环境变量 FAISS_PATH_OPENAI 加载 FAISS 索引。
    需要: langchain-community>=0.2, langchain-openai
    """
    faiss_dir = os.getenv("FAISS_PATH_OPENAI", "vectorstore-hotpot/hotpotqa_faiss")
    if not os.path.isdir(faiss_dir):
        raise FileNotFoundError(
            f"FAISS index not found at: {faiss_dir}\n"
            "Set env FAISS_PATH_OPENAI to your FAISS dir."
        )
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        emb_model = os.getenv("EMB_MODEL", "text-embedding-3-large")
        embeddings = OpenAIEmbeddings(model=emb_model)
        # allow_dangerous_deserialization=True 是新版本需要的加载开关
        vs = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ Loaded FAISS from: {faiss_dir}  (emb={emb_model})")
        return vs
    except Exception as e:
        raise RuntimeError(f"[FAISS.load_local error] {e}")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

# ===== Official HotpotQA Evaluation Metrics =====
# Adapted from the official HotpotQA evaluation script (Yang et al. 2018)
import re, string

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s

def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens   = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))
# =================================================

def _num(x, default=0.0) -> float:
    try:
        return float(x if not isinstance(x, list) else x[0])
    except Exception:
        return float(default)

def _round(x: float, nd=3) -> float:
    try:
        return round(float(x), nd)
    except Exception:
        return x

def _quantiles_q25_q75(arr: List[float]):
    """兼容小样本：statistics.quantiles 在 n<2 时会报错，这里兜底。"""
    if not arr:
        return (None, None)
    if len(arr) < 2:
        return (arr[0], arr[0])
    try:
        q = st.quantiles(arr, n=4, method="inclusive")
        return (q[0], q[2])  # p25, p75
    except Exception:
        # 旧版Python无method参数
        q = st.quantiles(arr, n=4)
        return (q[0], q[2])

def compute_and_write_stats(rows: List[Dict[str, Any]], out_dir: str):
    """
    统计指标写入 summary_stats.csv，并打印到控制台。
    自动过滤 NaN / inf，避免出现 nan 统计结果。
    """
    metrics_keys = [
        ("faithfulness", "faith"),
        ("response_relevancy", "rel"),
        ("noise_sensitivity", "noise"),
        ("semantic_f1", "semf1"),
        ("context_precision", "ctxP"),
        ("context_recall", "ctxR"),
        ("em", "EM"),
        ("official_f1", "offF1"),
    ]
    stats_rows = []
    total_n = len(rows)

    def arr_for(key: str) -> List[float]:
        vals = []
        for r in rows:
            v = r.get(key, None)
            if v is None:
                continue
            try:
                # 支持 list 结构（取第一个）
                x = float(v if not isinstance(v, list) else v[0])
                # 关键：过滤 NaN / +/-inf
                if math.isnan(x) or math.isinf(x):
                    continue
                vals.append(x)
            except Exception:
                # 无法转成 float 的直接跳过
                continue
        return vals

    print("\n===== SUMMARY STATS =====")
    print(f"N = {total_n}")

    for full_key, short in metrics_keys:
        arr = arr_for(full_key)
        if not arr:
            stats_rows.append({
                "metric": short, "n": 0, "mean": "", "median": "", "p25": "",
                "p75": "", "ge_0.8_pct": "", "le_0.2_pct": ""
            })
            print(f"{short}: (no data)")
            continue

        mean_v = _round(st.mean(arr))
        median_v = _round(st.median(arr))
        p25, p75 = _quantiles_q25_q75(arr)
        p25_v = "" if p25 is None else _round(p25)
        p75_v = "" if p75 is None else _round(p75)

        ge_08 = _round(sum(x >= 0.8 for x in arr) / len(arr))
        # 对噪声灵敏度（越小越好）额外提供 le_0.2
        le_02 = _round(sum(x <= 0.2 for x in arr) / len(arr)) if full_key == "noise_sensitivity" else ""

        print(
            f"{short}: mean={mean_v}  median={median_v}  p25={p25_v}  p75={p75_v}  >=0.8%={ge_08}"
            + (f"  <=0.2%={le_02}" if le_02 != "" else "")
        )

        stats_rows.append({
            "metric": short,
            "n": len(arr),
            "mean": mean_v,
            "median": median_v,
            "p25": p25_v,
            "p75": p75_v,
            "ge_0.8_pct": ge_08,
            "le_0.2_pct": le_02,
        })

    # 写 stats 到 CSV
    os.makedirs(out_dir, exist_ok=True)
    stats_csv = os.path.join(out_dir, "summary_stats.csv")
    with open(stats_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "n", "mean", "median", "p25", "p75", "ge_0.8_pct", "le_0.2_pct"],
        )
        w.writeheader()
        for r in stats_rows:
            w.writerow(r)

    print(f"📊 stats saved to: {stats_csv}")
    print("===== END STATS =====\n")

def run_dataset_and_collect(
    dataset: List[Dict[str, Any]],
    vectorstore: Any,
    out_dir: str = "runs/trajectories",
    retriever_top_k: int = 5,
    gen_max_attempts: int = 2,
    use_router: bool = False,
    use_par2: bool = False,        # PAR2-RAG two-stage retrieval (pure)
    use_adaptive_retrieval: bool = False,  # IRCoT + PAR2 fallback router
    debug: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "traj.jsonl")
    csv_path   = os.path.join(out_dir, "summary.csv")

    # Agents 初始化（兼容不同签名）
    evaluation_agent = EvaluationAgent()  # 如需 llm 后面再注入
    reasoning_agent  = ReasoningAgent()
    generation_agent = GenerationAgent()  # 如需 llm 后面再注入

    # --- 混合检索 + 重排 + 多查询 ---
    hybrid_retriever = HybridRetriever(vectorstore)
    reranker = create_cross_encoder_reranker(
        model_name=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_n=int(os.getenv("RERANKER_TOP_N", "5")),
    )
    # multi_query 需要 LLM；generation_agent 内部有 llm 属性
    _mq_llm = getattr(generation_agent, "llm", None)
    multi_query_fn = functools.partial(generate_query_variants, llm=_mq_llm, n_variants=2) if _mq_llm else None

    retrieval_agent = RetrievalAgent(
        vectorstore, evaluation_agent, top_k=retriever_top_k,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        multi_query_fn=multi_query_fn,
    )

    # CSV 表头
    csv_fields = [
        "qid","question","reference","answer",
        "faithfulness","response_relevancy","noise_sensitivity","semantic_f1",
        "context_precision","context_recall",
        "em","official_f1",
        "doc_count","generation_time_ms","retrieval_time_ms","total_time_ms"
    ]
    # 如果不存在则写表头
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_fields)

    rows_accum: List[Dict[str, Any]] = []

    for ex in dataset:
        q   = ex.get("question") or ex.get("query") or ""
        ref = ex.get("reference") or ex.get("answer") or None
        qid = ex.get("qid") or str(uuid.uuid4())

        traj_logger = TrajectoryLogger(out_dir=out_dir)
        # logger.start(qid=qid, query_raw=q)
        try:
            policy_path = os.getenv("ROUTER_POLICY_PATH", "agents/router_policy.pt")
            res = run_rag_pipeline(
                question=q,
                retrieval_agent=retrieval_agent,
                reasoning_agent=reasoning_agent,
                generation_agent=generation_agent,
                evaluation_agent=evaluation_agent,
                reference=ref,
                qid=qid,
                use_router=use_router,
                use_par2=use_par2,
                use_adaptive_retrieval=use_adaptive_retrieval,
                visualize=False,
                gen_max_attempts=gen_max_attempts,
                logger=traj_logger,
                router_device="cpu",
                router_policy_path=policy_path,

            )
        except Exception as e:
            print(f"[driver.error] {type(e).__name__}: {e}")
            traceback.print_exc()
            if debug:
                raise
            res = {
                "question": q, "answer": "ERROR",
                "faithfulness_score": 0.0, "response_relevancy": 0.0, "noise_sensitivity": 1.0, "semantic_f1_score": 0.0,
                "context_precision": 0.0, "context_recall": 0.0,
                "metrics": {"generation_time": 0.0, "retrieval_time": 0.0, "total_time": 0.0, "doc_count": 0.0,}
            }


        _answer = res.get("answer", "")
        _ref_str = ref if isinstance(ref, str) else (ref[0] if isinstance(ref, list) and ref else "")
        row = {
            "qid": qid,
            "question": q,
            "reference": ref,
            "answer": _answer,
            "faithfulness": _num(res.get("faithfulness_score", 0.0)),
            "response_relevancy": _num(res.get("response_relevancy", 0.0)),
            "noise_sensitivity": _num(res.get("noise_sensitivity", 1.0)),
            "semantic_f1": _num(res.get("semantic_f1_score", res.get("semantic_f1", 0.0))),
            "context_precision": _num(res.get("context_precision", 0.0)),
            "context_recall": _num(res.get("context_recall", 0.0)),
            "em": _exact_match(_answer, _ref_str) if _ref_str else 0.0,
            "official_f1": _token_f1(_answer, _ref_str) if _ref_str else 0.0,
            "generation_time_ms": _num(res.get("metrics",{}).get("generation_time", 0.0)),
            "retrieval_time_ms": _num(res.get("metrics",{}).get("retrieval_time", 0.0)),
            "total_time_ms": _num(res.get("metrics",{}).get("total_time", 0.0)),
            "doc_count": _num(res.get("metrics", {}).get("doc_count", 0.0)),
            "routing_decision": res.get("routing_decision", "n/a"),
        }
        rows_accum.append(row)

        # 逐条落盘 JSONL
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 逐条追加到 CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                row["qid"], row["question"], row["reference"], row["answer"],
                row["faithfulness"], row["response_relevancy"], row["noise_sensitivity"], row["semantic_f1"],
                row["context_precision"], row["context_recall"],
                row["em"], row["official_f1"],
                row["doc_count"], row["generation_time_ms"], row["retrieval_time_ms"], row["total_time_ms"]
            ])
        print(f"📝 saved trajectory: {qid}")

    # === 运行结束后：计算统计并写入 summary_stats.csv ===
    compute_and_write_stats(rows_accum, out_dir)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to JSONL with question/reference pairs")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--gen_max_attempts", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default="runs/trajectories")
    ap.add_argument("--debug", action="store_true")
    # ✅ 新增：是否启用 LangGraph + Router
    ap.add_argument(
        "--use_router",
        type=int,
        default=0,
        help="1 = use LangGraph + BC router; 0 = linear baseline"
    )
    ap.add_argument(
        "--use_par2",
        type=int,
        default=0,
        help="1 = enable PAR2-RAG two-stage retrieval (requires --use_router 1)"
    )
    ap.add_argument(
        "--use_adaptive_retrieval",
        type=int,
        default=0,
        help="1 = IRCoT first, fall back to PAR2 when retrieval quality is poor (requires --use_router 1)"
    )
    return ap.parse_args()

def main():
    args = parse_args()
    dataset = load_jsonl(args.dataset)
    vectorstore = get_vectorstore()

    # 维度自检（建议保留）
    try:
        vs = vectorstore
        index_dim = getattr(getattr(vs, "index", None), "d", None)
        if hasattr(vs, "embedding_function") and hasattr(vs.embedding_function, "embed_query"):
            qvec = vs.embedding_function.embed_query("hello")
            query_dim = len(qvec) if qvec is not None else None
        else:
            query_dim = None

        print(f"🔧 FAISS index dim={index_dim}, query dim={query_dim}")
        if index_dim and query_dim and index_dim != query_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: index={index_dim}, query={query_dim}. "
                "Set EMB_MODEL to the one used to build the FAISS index."
            )
    except Exception as e:
        print(f"[vectorstore.dimcheck] {e}")
        if args.debug:
            raise

    run_dataset_and_collect(
        dataset=dataset,
        vectorstore=vectorstore,
        out_dir=args.out_dir,
        retriever_top_k=args.top_k,
        gen_max_attempts=args.gen_max_attempts,
        use_router=bool(args.use_router),
        use_par2=bool(args.use_par2),
        use_adaptive_retrieval=bool(args.use_adaptive_retrieval),
        debug=args.debug,
    )

    print("\n✅ Done. Check:")
    print(f"  - {args.out_dir}/*.jsonl  (单条轨迹)")
    print(f"  - {args.out_dir}/traj.jsonl  (汇总)")
    print(f"  - {args.out_dir}/summary.csv (明细)")
    print(f"  - {args.out_dir}/summary_stats.csv (统计)")

if __name__ == "__main__":
    main()
# ===== END PASTE =====
