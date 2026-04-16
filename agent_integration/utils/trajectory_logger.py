# utils/trajectory_logger.py
# 通用轨迹记录器：把一次问答的 plan/retrieval/generation/eval/router 动作记录到 JSONL

from __future__ import annotations
import json, os, time, hashlib, re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.1"

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def safe_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# ---- 安全工具 ----
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
def _clean_text(s: str, max_len: int = 2000) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _CONTROL_RE.sub(" ", s).strip()
    if max_len and len(s) > max_len:
        s = s[:max_len]
    return s

def _to_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        # list → 取均值（更稳）
        if isinstance(x, list):
            nums = []
            for v in x:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            return float(sum(nums)/len(nums)) if nums else default
        return float(x)
    except Exception:
        # 有些对象有 .value
        try:
            return float(getattr(x, "value"))
        except Exception:
            return default


# ---- 评估键名规范化映射 ----
_EVAL_KEY_MAP = {
    # faithfulness
    "faith": "faith",
    "faithfulness": "faith",
    "faithfulness_score": "faith",
    "faithfulness_status": "faith_status",
    # relevancy
    "response_relevancy": "response_relevancy",
    "answer_relevancy": "response_relevancy",
    "response_relevancy_status": "relevancy_status",
    "answer_relevancy_status": "relevancy_status",
    # noise
    "noise_sensitivity": "noise_sensitivity",
    "noise": "noise_sensitivity",
    "noise_sensitivity_status": "noise_status",
    # context precision / recall
    "context_precision": "context_precision",
    "ctxp": "context_precision",
    #"ctxp_status": "ctxp_status",
    "context_recall": "context_recall",
    "ctxr": "context_recall",
    #"ctxr_status": "ctxr_status",
    # semantic f1
    "semantic_f1": "semantic_f1",
    "semantic_f1_score": "semantic_f1",
    "doc_count": "doc_count",     # ← 新增：允许记录 doc_count
}

# ---- 评估状态键规范化 ----
# 允许上层用不同别名传入 *_status 字段，这里统一收敛为固定 6 个指标的 status
_EVAL_STATUS_KEYS = {
    # faith
    "faith_status", "faithfulness_status",
    # relevancy
    "response_relevancy_status", "answer_relevancy_status",
    # noise
    "noise_status", "noise_sensitivity_status",
    # ctx precision / recall
    "context_precision_status", "ctxp_status",
    "context_recall_status", "ctxr_status",
    # semantic f1
    "semantic_f1_status",
}

# 将各种 *_status 键映射到标准指标名
_STATUS_TO_METRIC = {
    # faith
    "faith_status": "faith", "faithfulness_status": "faith",
    # relevancy
    "response_relevancy_status": "response_relevancy",
    "answer_relevancy_status": "response_relevancy",
    # noise
    "noise_status": "noise_sensitivity",
    "noise_sensitivity_status": "noise_sensitivity",
    # ctx precision / recall
    "context_precision_status": "context_precision",
    "ctxp_status": "context_precision",
    "context_recall_status": "context_recall",
    "ctxr_status": "context_recall",
    # semantic f1
    "semantic_f1_status": "semantic_f1",
}


def _normalize_eval_dict(d: Dict[str, Any]) -> Dict[str, float]:
    """把各种别名键合到标准键；数值化；不在映射表内的键会原样返回（可选保留）。"""
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        k_norm = _EVAL_KEY_MAP.get(str(k).lower(), None)
        if k_norm:
            out[k_norm] = _to_float(v, 0.0)
        # 其他键丢弃（避免污染 eval），如需保留可在此扩展
    return out

def _normalize_eval_with_status(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    同时规范化数值与 status：
      - 数值走 _normalize_eval_dict（已有函数），只保留我们关注的 6 个标准键；
      - status 键（*_status 或在 _EVAL_STATUS_KEYS 内）统一映射为 {<metric>_status: "ok"/"missing"/...}。
    """
    base = _normalize_eval_dict(d)  # 已将别名数值键映射为: faith / response_relevancy / noise_sensitivity / context_precision / context_recall / semantic_f1
    out: Dict[str, Any] = dict(base)

    # 额外处理 *_status
    for k, v in (d or {}).items():
        lk = str(k).lower().strip()
        if (lk in _EVAL_STATUS_KEYS) or lk.endswith("_status"):
            # 明确映射：若在映射表，直接用；否则尝试去掉后缀做别名匹配
            metric = _STATUS_TO_METRIC.get(lk)
            if metric is None and lk.endswith("_status"):
                # 去掉 _status 后再走 _EVAL_KEY_MAP
                raw_metric_key = lk[:-7]  # 去掉 "_status"
                metric = _EVAL_KEY_MAP.get(raw_metric_key, None)
                # 特判：噪声也可能叫 "noise"
                if metric is None and raw_metric_key == "noise":
                    metric = "noise_sensitivity"
            if metric:
                out[f"{metric}_status"] = str(v)
            else:
                # 没法识别的 status，丢弃或按需保留为原键（这里选择保留，方便排查）
                out[lk] = v

    return out


@dataclass
class TrajectoryRecord:
    qid: str
    query_raw: str
    started_at: str = field(default_factory=_now_iso)
    schema_version: str = SCHEMA_VERSION
    reason_steps: List[str] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)   # 建议只放摘要/哈希
    generations: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    eval: Dict[str, Any] = field(default_factory=dict)      # 规范化后的关键指标
    router_action: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)      # 预留：模型名、版本、超参等
    finished_at: Optional[str] = None

class TrajectoryLogger:
    def __init__(self, out_dir: str = "runs/trajectories"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.rec: Optional[TrajectoryRecord] = None
        self.started: bool = False

    # -------- 生命周期 --------
    def start(self, qid: str, query_raw: str, **meta):
        try:
            self.rec = TrajectoryRecord(qid=qid, query_raw=_clean_text(query_raw, 4000))
            if meta:
                self.rec.meta.update(meta)
            self.started = True
        except Exception as e:
            print(f"[logger.start.error] {type(e).__name__}: {e}")
            self.rec = None
            self.started = False


    def add_model_ident(self, model=None, base_url=None, ctx_tokens=None, gen_tokens=None):
        if not self.started or self.rec is None:
            return
        if model:
            self.rec.meta["model"] = str(model)
        if base_url:
            self.rec.meta["base_url"] = str(base_url)
        if ctx_tokens is not None:
            try:
                self.rec.meta["max_ctx_tokens"] = int(float(ctx_tokens))
            except Exception:
                pass
        if gen_tokens is not None:
            try:
                self.rec.meta["max_gen_tokens"] = int(float(gen_tokens))
            except Exception:
                pass


    def commit(self, out_path: Optional[str] = None):
        if not self.started or self.rec is None:
            print("[logger.commit.error] commit() called before start()")
            self.started = False
            return
        try:
            self.rec.finished_at = _now_iso()
            data = asdict(self.rec)
            if out_path is None:
                out_path = os.path.join(self.out_dir, f"{self.rec.qid}.jsonl")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[logger.commit.error] {e}")
        finally:
            self.rec = None
            self.started = False

    def to_dict(self) -> Dict[str, Any]:
        """返回当前轨迹快照（不落盘），便于外层统一写 JSONL/CSV。"""
        assert self.rec is not None, "to_dict() called before start()"
        return asdict(self.rec)

    # -------- 记录内容 --------
    def add_reason(self, text: str):
        if not self.started or self.rec is None:
            print("[logger.error] add_reason() ignored: logger not started")
            return
        self.rec.reason_steps.append(_clean_text(text, 2000))

    def add_tool_call(self, **kwargs):
        if not self.started or self.rec is None:
            print("[logger.error] add_tool_call() ignored: logger not started")
            return
        safe_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, str):
                safe_kwargs[k] = _clean_text(v, 2000)
            else:
                safe_kwargs[k] = v
        self.rec.tools.append(safe_kwargs)

    def add_observation(self, text_or_summary: str, do_hash: bool = True, max_len: int = 500):
        if not self.started or self.rec is None:
            print("[logger.error] add_observation() ignored: logger not started")
            return
        if do_hash:
            self.rec.observations.append(f"[OBS:{safe_hash(text_or_summary)}]")
        else:
            self.rec.observations.append(_clean_text(text_or_summary, max_len))

    def add_generation(self, attempt: int, prompt_id: str, answer: str):
        if not self.started or self.rec is None:
            print("[logger.error] add_generation() ignored: logger not started")
            return
        self.rec.generations.append({
            "attempt": int(attempt),
            "prompt_id": _clean_text(prompt_id, 128),
            "answer": _clean_text(answer, 2000)
        })

    def set_final_answer(self, answer: str):
        if not self.started or self.rec is None:
            print("[logger.error] set_final_answer() ignored: logger not started")
            return
        self.rec.final_answer = _clean_text(answer, 4000)

    def add_eval(self, **metrics):
        if not self.started or self.rec is None:
            print("[logger.error] add_eval() ignored: logger not started")
            return
        try:
            norm = _normalize_eval_with_status(metrics)
            if not self.rec.eval:
                self.rec.eval = {}
            self.rec.eval.update(norm)
        except Exception as e:
            print(f"[logger.add_eval.error] {type(e).__name__}: {e}")


    # 在 TrajectoryLogger 内新增：
    def add_generation_attempt(self, attempt: int, prompt_id: str, answer: str,
                           latency_ms: float = None,
                           eval_scores: Optional[Dict[str, Any]] = None):
        """
        记录单次生成 attempt 的详情：答案、耗时、该次评测分。
        eval_scores 只需传你有的键，add_generation_attempt 内部会做键名规范化。
        """
        if not self.started or self.rec is None:
            print("[logger.error] add_generation_attempt() ignored: logger not started")
            return
        entry = {
            "attempt": int(attempt),
            "prompt_id": _clean_text(prompt_id, 128),
            "answer": _clean_text(answer, 2000)
        }
        if latency_ms is not None:
            try:
                entry["latency_ms"] = float(latency_ms)
            except Exception:
                pass
        if eval_scores:
            entry["eval"] = _normalize_eval_dict(eval_scores)
        self.rec.generations.append(entry)


    def set_router_action(self, action: str):
        if not self.started or self.rec is None:
            print("[logger.error] set_router_action() ignored: logger not started")
            return
        self.rec.router_action = _clean_text(str(action).lower().strip(), 64)

    # -------- 便捷元信息（供主流程调用）--------
    def set_reference(self, reference: Optional[str]):
        if not self.started or self.rec is None:
            return
        if reference:
            self.rec.meta["reference"] = _clean_text(reference, 4000)

    def set_refined_query(self, rq: Optional[str]):
        if not self.started or self.rec is None:
            return
        if rq:
            self.rec.meta["refined_query"] = _clean_text(rq, 4000)

    def to_summary_row(self) -> Optional[List[Any]]:
        """导出一行 CSV 摘要（若还未结束，可返回 None）。"""
        if self.rec is None:
            return None
        ev = self.rec.eval or {}
        meta = self.rec.meta or {}
        # doc_count 是通过 add_eval(...) 写进 self.rec.eval 的；若不存在则为 0.0
        doc_count = float(ev.get("doc_count", 0.0))
        return [
            self.rec.qid,
            self.rec.query_raw,
            (self.rec.meta or {}).get("reference", None),
            self.rec.final_answer,
            ev.get("faith", 0.0),
            ev.get("response_relevancy", 0.0),
            ev.get("noise_sensitivity", 1.0),
            ev.get("semantic_f1", 0.0),
            ev.get("context_precision", 0.0),
            ev.get("context_recall", 0.0),
            doc_count,                      # ✅ 新增 doc_count，放在最后一列
        ]
