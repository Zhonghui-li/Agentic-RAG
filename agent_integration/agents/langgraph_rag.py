# langgraph_rag.py  （改造版）
from typing import Dict, List, Any, TypedDict, Optional
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
import time
import os, re
import uuid
import numpy as np
from decimal import Decimal

from agents.reasoning_agent import ReasoningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent
from agents.RLRouterAgent import RLRouterAgent, POLICY_SAVE_PATH
from agents.multi_query import decompose_query
from agents.esc import EvidenceSufficiencyController
from agents.retrieval_router_bc import RetrievalRouterBC


# 统一轨迹日志（兜底为 None）
try:
    from utils.trajectory_logger import TrajectoryLogger
except Exception:
    TrajectoryLogger = None  # 兜底


# ---- util: 将各种结果转成 float ----
def extract_scalar(val):
    if val is None:
        return 0.0
    if isinstance(val, list) and val:
        try:
            return float(val[0])
        except Exception:
            # 尝试均值更稳妥
            try:
                return float(np.mean([float(v) for v in val]))
            except Exception:
                return 0.0
    if isinstance(val, (int, float, np.floating, np.generic, Decimal)):
        return float(val)
    try:
        return float(val)
    except Exception:
        if hasattr(val, "value"):
            try:
                return float(getattr(val, "value"))
            except Exception:
                return 0.0
        return 0.0


# ---- State ----
class AgentState(TypedDict):
    question: str
    conversation_context: List[Dict[str, str]]  # [{q: ..., a: ...}] from prior turns
    refined_query: str
    docs: List[Document]
    answer: str
    faithfulness_score: float
    response_relevancy: float
    noise_sensitivity: float
    semantic_f1_score: float
    context_recall: float
    context_precision: float
    attempts: int
    next_step: str
    messages: List[Dict[str, Any]]
    error: Optional[str]
    start_time: float
    metrics: Dict[str, Any]
    requery_count: int
    regenerate_count: int
    max_attempts: int
    max_regenerates: int
    max_requeries: int
    reference: Optional[str]
    qid: str
    logger: Optional[TrajectoryLogger]
    # PAR2-RAG fields
    refine_hop: int        # current Stage 2 hop counter
    par2_max_hops: int     # Stage 2 hop budget
    used_follow_ups: List[str]  # de-dup ESC follow-up queries
    # Adaptive retrieval router
    retrieval_quality: str  # "ok" | "poor" — set by retrieval_router_node


def create_rag_graph(
    retrieval_agent: RetrievalAgent,
    reasoning_agent: ReasoningAgent,
    generation_agent: GenerationAgent,
    evaluation_agent: EvaluationAgent,
    rl_router: Optional["RLRouterAgent"] = None,
    use_par2: bool = False,
    use_adaptive_retrieval: bool = False,
    par2_llm=None,
    par2_n_subqueries: int = 5,
    par2_max_hops: int = 4,
):
    # 初始化 RL 路由策略（优先使用传入的 router 实例，回退到默认路径）
    if rl_router is not None:
        _rl_router = rl_router
    else:
        _rl_router = RLRouterAgent(policy_path=os.path.join(os.path.dirname(__file__), 'router_policy.pt'))

    # 注入统一 logger（若外部未注入）
    def _ensure_logger_on_state(state: AgentState) -> Optional[TrajectoryLogger]:
        """
        确保 state 里有一个可用的 logger：
        - 如果 state["logger"] 是 None：创建并 start()
        - 如果有 logger 但还没 start：帮它 start()
        返回最终可用的 logger（如果失败就返回 None）
        """
        logger = state.get("logger")

        # 1) 没有 logger → 新建并启动
        if logger is None:
            try:
                out_dir = state.get("traj_out_dir", "runs/trajectories")
                logger = TrajectoryLogger(out_dir=out_dir)
                qid = state.get("qid") or state.get("question_id") or "unknown"
                logger.start(qid=str(qid), query_raw=state.get("question", ""))
                state["logger"] = logger
            except Exception as e:
                print(f"[logger.ensure_state.error] {type(e).__name__}: {e}")
                return None
            return logger

        # 2) 已有 logger 但尚未 started → 帮它 start 一下
        if hasattr(logger, "started") and not logger.started:
            try:
                qid = state.get("qid") or state.get("question_id") or "unknown"
                logger.start(qid=str(qid), query_raw=state.get("question", ""))
            except Exception as e:
                print(f"[logger.ensure_state.start.error] {type(e).__name__}: {e}")
                return None

        return logger

    def _ensure_logger(agent, logger: Optional[TrajectoryLogger]):
        """
        把 logger 注入到各个 agent 上（如果 agent 有 .logger 属性）。
        """
        if logger is None:
            return
        if hasattr(agent, "logger"):
            try:
                agent.logger = logger
            except Exception:
                pass

    # —— 检索缓存（简单 LRU）——
    retrieve_cache: Dict[Any, Dict[str, Any]] = {}
    max_cache_size = 20

    def _empty_retrieve_result(latency_ms: float = 0.0):
        return {
            "docs": [],
            "context_precision": 0.0,
            "context_recall": 0.0,
            "latency_ms": latency_ms,
            "hits_meta": []
        }

    def cached_retrieve(query: str, reference: Optional[str] = None) -> Dict[str, Any]:
        try:
            result = retrieval_agent.retrieve(query, reference=reference)
            # 兜底补全字段
            if not isinstance(result, dict):
                return _empty_retrieve_result()
            result.setdefault("docs", [])
            result.setdefault("context_precision", 0.0)
            result.setdefault("context_recall", 0.0)
            result.setdefault("latency_ms", 0.0)
            result.setdefault("hits_meta", [])
            return result
        except Exception as e:
            print(f"检索错误: {e}")
            return _empty_retrieve_result()

    def cached_retrieve_with_resource_mgmt(query: str, reference: Optional[str] = None) -> Dict[str, Any]:
        cache_key = (query, reference)
        if cache_key in retrieve_cache:
            return retrieve_cache[cache_key]
        result = cached_retrieve(query, reference=reference)
        if len(retrieve_cache) >= max_cache_size:
            oldest_key = next(iter(retrieve_cache))
            del retrieve_cache[oldest_key]
        retrieve_cache[cache_key] = result
        return result

    # -----------------------------
    # (1) Query optimization node
    # -----------------------------
    def query_optimizer(state: AgentState) -> AgentState:
        # ✅ 统一拿 logger（创建 + start）
        logger = _ensure_logger_on_state(state)

        # ✅ 把同一个 logger 注入到各个 agent（方便它们内部用）
        if logger is not None:
            if hasattr(reasoning_agent, "logger"):
                reasoning_agent.logger = logger
            if hasattr(retrieval_agent, "logger"):
                retrieval_agent.logger = logger
            if hasattr(generation_agent, "logger"):
                generation_agent.logger = logger
            if hasattr(evaluation_agent, "logger"):
                evaluation_agent.logger = logger

        try:
            print(f"\n🧠 优化查询: {state['question']}")
            start = time.time()

            if logger:
                logger.add_reason(f"[query_optimizer.start] q={state['question']}")

            conv_ctx = state.get("conversation_context") or []
            reasoning_result = reasoning_agent.plan(
                user_question=state["question"],
                retrieved_docs=None,
                conversation_context=conv_ctx
            )
            refined_query = reasoning_result.get("refined_query") or state["question"]
            duration = time.time() - start

            if logger:
                logger.add_reason(f"[query_optimizer.refined] {refined_query}")

            return {
                **state,
                "refined_query": refined_query,
                "metrics": {**state["metrics"], "query_optimization_time": duration},
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Optimized query: {refined_query}"
                }]
            }
        except Exception as e:
            import traceback
            print(f"⚠️ Query optimization error: {e}")
            print(traceback.format_exc())
            if logger:
                logger.add_reason(f"[query_optimizer.error] {e}")
            return {
                **state,
                "refined_query": state["question"],
                "error": f"Query optimization failed: {str(e)}",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Query optimization failed: {str(e)}"
                }]
            }

    # -----------------------------
    # (2) Retrieval node
    # -----------------------------
    def retriever(state: AgentState) -> AgentState:
        logger = _ensure_logger_on_state(state)

        try:
            query = state["refined_query"]
            reference = state.get("reference")
            print(f"\n📚 Retrieving based on optimized query: {query}")

            start = time.time()
            ret_result = cached_retrieve_with_resource_mgmt(query, reference=reference)

            docs_raw = ret_result.get("docs", [])
            ctxP = extract_scalar(ret_result.get("context_precision", 0.0))
            ctxR = extract_scalar(ret_result.get("context_recall", 0.0))
            duration = time.time() - start

            if not docs_raw:
                print("⚠️ No relevant documents found")
                if logger:
                    logger.add_reason("[retriever.warning] no relevant documents found")
                return {
                    **state,
                    "docs": [],
                    "answer": "Sorry, I couldn't find any information related to your question.",
                    "faithfulness_score": 0.0,
                    "next_step": "end",
                    "metrics": {
                        **state["metrics"],
                        "retrieval_time": duration,
                        "doc_count": 0,
                        "context_precision": ctxP,
                        "context_recall": ctxR
                    },
                    "messages": state["messages"] + [{
                        "role": "system",
                        "content": "No relevant documents found"
                    }]
                }

            # 防御性包装为 LC Document（裁剪）
            def _to_lc_doc(d) -> Document:
                if isinstance(d, Document):
                    txt = d.page_content or ""
                    if len(txt) > 3000:
                        txt = txt[:3000]
                    return Document(page_content=txt, metadata=d.metadata or {})
                txt = ""
                meta = {}
                if isinstance(d, dict):
                    txt = d.get("page_content") or d.get("text") or d.get("content") or ""
                    meta = d.get("metadata") or {}
                else:
                    txt = str(getattr(d, "page_content", "") or d)
                if len(txt) > 3000:
                    txt = txt[:3000]
                return Document(page_content=txt, metadata=meta)

            docs = [_to_lc_doc(d) for d in docs_raw]

            print(f"🎯 Retrieval Metrics: Precision={ctxP:.2f}, Recall={ctxR:.2f}")
            if logger:
                logger.add_reason(f"[retriever.done] ctxP={ctxP:.3f}, ctxR={ctxR:.3f}, docs={len(docs)}")

            return {
                **state,
                "docs": docs,
                "context_precision": ctxP,
                "context_recall": ctxR,
                "metrics": {
                    **state["metrics"],
                    "retrieval_time": duration,
                    "doc_count": len(docs),
                    "context_precision": ctxP,
                    "context_recall": ctxR
                },
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Retrieved {len(docs)} documents"
                }]
            }
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            if logger:
                logger.add_reason(f"[retriever.error] {e}")
            return {
                **state,
                "docs": [],
                "error": f"Retrieval failed: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Retrieval failed: {str(e)}"
                }]
            }

    # -----------------------------
    # (3) Generate answer node
    # -----------------------------
    def generator(state: AgentState) -> AgentState:
        logger = _ensure_logger_on_state(state)

        try:
            # ⚠️ 使用"用户问题"来生成答案；refined_query 只用于检索
            question = state["question"]
            docs = state["docs"]
            reference = state.get("reference")

            # --- Router-level regenerate feedback ---
            regenerate_count = state.get("regenerate_count", 0)
            previous_answer = state.get("answer") if regenerate_count > 0 else None
            failure_hint = None
            if previous_answer:
                faith = float(state.get("faithfulness_score") or 0.0)
                rel   = float(state.get("response_relevancy") or 0.0)
                noise = float(state.get("noise_sensitivity") or 1.0)
                semf1 = float(state.get("semantic_f1_score") or 0.0)
                if faith < 0.5:
                    failure_hint = ("Your previous answer contains claims not directly supported "
                                    "by the retrieved documents. Stick strictly to what the documents say.")
                elif noise > 0.7:
                    failure_hint = ("Your previous answer was influenced by irrelevant documents. "
                                    "Identify which documents are actually relevant to the question.")
                elif rel < 0.3:
                    failure_hint = ("Your previous answer does not directly address the question. "
                                    "Focus on answering exactly what was asked.")
                else:
                    failure_hint = ("Re-examine the context carefully. "
                                    "Make sure your answer is concise and factually grounded.")
                print(f"\n✍️ Generate answer... (regenerate #{regenerate_count}, feedback injected)")
            else:
                print(f"\n✍️ Generate answer... (use original question)")
            start = time.time()
            answer_result = generation_agent.answer(
                question=question,
                docs=docs,
                evaluation_agent=evaluation_agent,
                ground_truth=reference,
                previous_answer=previous_answer,
                failure_hint=failure_hint,
            )
            duration = time.time() - start

            if logger:
                logger.add_reason("[generator.done] answer generated and evaluated")

            relevancy = (
                answer_result.get("response_relevancy")
                or answer_result.get("answer_relevancy")
                or 0.0
            )

            metrics_update = {
                "generation_time": duration,
                "cached_eval_result": answer_result.get("cached_eval_result", None)
            }

            return {
                **state,
                "answer": answer_result.get("answer", ""),
                "faithfulness_score": answer_result.get("faithfulness_score", 0.0),
                "response_relevancy": extract_scalar(relevancy),
                "noise_sensitivity": answer_result.get("noise_sensitivity", 1.0),
                "semantic_f1_score": (
                    answer_result.get("semantic_f1_score", 0.0)
                    if answer_result.get("semantic_f1_score") is not None
                    else answer_result.get("semantic_f1", 0.0)
                ),
                "eval_result": answer_result.get("cached_eval_result", None),
                "metrics": {**state["metrics"], **metrics_update},
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": answer_result.get("answer", "")
                }]
            }
        except Exception as e:
            print(f"⚠️ Generating answers incorrectly: {e}")
            if logger:
                logger.add_reason(f"[generator.error] {e}")
            return {
                **state,
                "answer": "Sorry, I encountered an issue while generating an answer.",
                "error": f"Failed to generate an answer: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Failed to generate an answer: {str(e)}"
                }]
            }

    # -----------------------------
    # (4) Evaluator node（保留接口，当前不使用）
    # -----------------------------
    def evaluator(state: AgentState) -> AgentState:
        logger = _ensure_logger_on_state(state)
        print(f"⚡ Evaluator skipped (由 Generator 已评估)")
        if logger:
            logger.add_reason("[evaluator.skip] generator already evaluated")
        return state

    # -----------------------------
    # (5) Router node
    # -----------------------------
    def router(state: AgentState) -> AgentState:
        # 统一确保 logger 存在并且已 start
        logger = _ensure_logger_on_state(state)

        print(
            f"[Router.node] ctxP={state.get('context_precision')}, "
            f"ctxR={state.get('context_recall')}, "
            f"faith={state.get('faithfulness_score')}, "
            f"rel={state.get('response_relevancy')}"
        )

        decision_state = {
            "context_precision": state.get("context_precision", 0.0),
            "context_recall": state.get("context_recall", 0.0),
            "faithfulness_score": state.get("faithfulness_score", 0.0),
            "response_relevancy": state.get("response_relevancy", 0.0),
            "noise_sensitivity": state.get("noise_sensitivity", 1.0),
            "semantic_f1_score": state.get("semantic_f1_score", 0.0),
        }

        action = _rl_router.decide(decision_state)
        print(f"🔄 RLRouterAgent decided action: {action}")

        # 记录 router 决策到轨迹里
        if logger:
            logger.set_router_action(action)

        attempts = state.get("attempts", 0)
        requery_count = state.get("requery_count", 0)
        regenerate_count = state.get("regenerate_count", 0)

        if action == "requery" and requery_count < state["max_requeries"]:
            return {
                **state,
                "next_step": "requery",
                "requery_count": requery_count + 1,
                "attempts": attempts + 1,
            }

        if action == "regenerate" and regenerate_count < state["max_regenerates"]:
            return {
                **state,
                "next_step": "regenerate",
                "regenerate_count": regenerate_count + 1,
                "attempts": attempts + 1,
            }

        # 默认 / 超出上限：结束
        return {
            **state,
            "next_step": "end",
            "attempts": attempts + 1,
        }

    # -----------------------------
    # (6) requery_optimizer
    # -----------------------------
    def requery_optimizer(state: AgentState) -> AgentState:
        # 统一确保 logger 存在 & started
        logger = _ensure_logger_on_state(state)

        try:
            print(f"\n🔄 Re-optimizing query...")
            start = time.time()

            # 确保 ReasoningAgent 也拿到同一个 logger（可选，但推荐）
            _ensure_logger(reasoning_agent, logger)
            _ensure_logger(retrieval_agent, logger)
            _ensure_logger(generation_agent, logger)
            _ensure_logger(evaluation_agent, logger)

            if logger:
                logger.add_reason(f"[query_optimizer.start] q={state['question']}")

            reasoning_result = reasoning_agent.plan(
                user_question=state["question"],
                retrieved_docs=state["docs"]
            )
            refined_query = reasoning_result.get("refined_query") or state["question"]
            duration = time.time() - start

            if logger:
                logger.add_reason(f"[requery.refined] {refined_query}")

            return {
                **state,
                "refined_query": refined_query,
                "metrics": {
                    **state["metrics"],
                    "requery_optimization_time": duration,
                },
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Re-optimized query: {refined_query}",
                }],
            }
        except Exception as e:
            print(f"⚠️ Re-optimizing query error: {e}")
            if logger:
                logger.add_reason(f"[requery.error] {e}")
            return {
                **state,
                "error": f"Re-optimizing query failed: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Re-optimizing query failed: {str(e)}",
                }],
            }

    # -----------------------------
    # (7) finalizer
    # -----------------------------
    def finalizer(state: AgentState) -> AgentState:
        # 统一确保 logger 存在 & started
        logger = _ensure_logger_on_state(state)

        total_time = time.time() - state["start_time"]
        print(f"\n⏱️ Total processing time: {total_time:.2f} seconds")

        if logger:
            # 记录最终答案
            if state.get("answer"):
                logger.set_final_answer(state["answer"])
            # 落盘 JSONL
            logger.commit()

        # 清理检索缓存
        retrieve_cache.clear()

        return {
            **state,
            "metrics": {
                **state["metrics"],
                "total_time": total_time,
            },
        }

    # ---- PAR2-RAG nodes (only built when use_par2=True) ----
    _esc = EvidenceSufficiencyController(llm=par2_llm, max_hops=par2_max_hops) if (use_par2 or use_adaptive_retrieval) else None

    def anchor_node(state: AgentState) -> AgentState:
        """PAR2-RAG Stage 1: decompose question into sub-queries, retrieve all, merge into C_anchor."""
        logger = _ensure_logger_on_state(state)
        question = state["refined_query"] or state["question"]
        reference = state.get("reference")

        print(f"\n🔍 [PAR2 Stage 1] Decomposing: {question}")
        try:
            sub_queries = decompose_query(question, llm=par2_llm, n_subqueries=par2_n_subqueries)
        except Exception as e:
            print(f"⚠️ [PAR2 anchor] decompose failed ({e}), falling back to original query")
            sub_queries = [question]

        all_docs: List[Document] = []
        seen_ids: set = set()

        def _to_lc_doc(d) -> Document:
            if isinstance(d, Document):
                txt = (d.page_content or "")[:3000]
                return Document(page_content=txt, metadata=d.metadata or {})
            txt = ""
            meta = {}
            if isinstance(d, dict):
                txt = d.get("page_content") or d.get("text") or d.get("content") or ""
                meta = d.get("metadata") or {}
            else:
                txt = str(getattr(d, "page_content", "") or d)
            return Document(page_content=txt[:3000], metadata=meta)

        for sq in sub_queries:
            result = cached_retrieve(sq, reference=reference)
            for d in result.get("docs", []):
                doc = _to_lc_doc(d)
                doc_id = hash(doc.page_content[:128])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        print(f"✅ [PAR2 Stage 1] C_anchor: {len(all_docs)} unique docs from {len(sub_queries)} sub-queries")

        # Evaluate merged C_anchor docs to get real ctxP/ctxR
        ctxP, ctxR = 0.0, 0.0
        if evaluation_agent is not None and reference and all_docs:
            try:
                eval_res = evaluation_agent.evaluate_retrieval(
                    user_query=question, retrieved_docs=all_docs, reference=reference
                )
                ctxP = extract_scalar(eval_res.get("context_precision", 0.0) or 0.0)
                ctxR = extract_scalar(eval_res.get("context_recall", 0.0) or 0.0)
                print(f"📊 [PAR2 Stage 1] ctxP={ctxP:.3f}, ctxR={ctxR:.3f}")
            except Exception as e:
                print(f"⚠️ [PAR2 anchor eval] {e}")

        if logger:
            logger.add_reason(f"[anchor_node] {len(sub_queries)} sub-queries → {len(all_docs)} docs, ctxP={ctxP:.3f}, ctxR={ctxR:.3f}")

        return {
            **state,
            "docs": all_docs,
            "refine_hop": 0,
            "par2_max_hops": par2_max_hops,
            "used_follow_ups": [],
            "context_precision": ctxP,
            "context_recall": ctxR,
            "messages": state["messages"] + [{
                "role": "system",
                "content": f"[PAR2 Stage 1] {len(all_docs)} docs anchored from {len(sub_queries)} sub-queries"
            }]
        }

    def refine_node(state: AgentState) -> AgentState:
        """PAR2-RAG Stage 2: ESC-gated iterative refinement. Loops back via conditional edge."""
        logger = _ensure_logger_on_state(state)
        question = state["question"]
        docs = state["docs"]
        hop = state.get("refine_hop", 0)
        reference = state.get("reference")

        action, follow_up = _esc.check(question=question, docs=docs, current_hop=hop)

        if action == "STOP":
            if logger:
                logger.add_reason(f"[refine_node] hop={hop} ESC=STOP → generator")
            return {**state, "next_step": "generate"}

        # De-dup: if ESC repeats a follow-up we already tried, stop early
        used_follow_ups = list(state.get("used_follow_ups") or [])
        if follow_up in used_follow_ups:
            print(f"[ESC] hop={hop} follow-up already tried → STOP (dedup)")
            if logger:
                logger.add_reason(f"[refine_node] hop={hop} ESC repeat query → STOP (dedup)")
            return {**state, "next_step": "generate"}

        # CONTINUE: retrieve follow-up query and merge
        used_follow_ups.append(follow_up)
        print(f"\n🔄 [PAR2 Stage 2] hop={hop+1} follow-up: {follow_up}")
        result = cached_retrieve(follow_up, reference=reference)
        new_docs = result.get("docs", [])

        seen_ids = {hash((getattr(d, "page_content", "") or "")[:128]) for d in docs}
        merged = list(docs)
        for d in new_docs:
            did = hash((getattr(d, "page_content", "") or "")[:128])
            if did not in seen_ids:
                seen_ids.add(did)
                if isinstance(d, Document):
                    merged.append(Document(page_content=(d.page_content or "")[:3000], metadata=d.metadata or {}))
                else:
                    txt = (d.get("page_content", "") if isinstance(d, dict) else str(d))[:3000]
                    merged.append(Document(page_content=txt, metadata={}))

        print(f"📚 [PAR2 Stage 2] hop={hop+1}: {len(merged)} total docs after merge")
        if logger:
            logger.add_reason(f"[refine_node] hop={hop+1} ESC=CONTINUE → {len(merged)} docs")

        return {
            **state,
            "docs": merged,
            "refine_hop": hop + 1,
            "used_follow_ups": used_follow_ups,
            "next_step": "refine",
            "messages": state["messages"] + [{
                "role": "system",
                "content": f"[PAR2 Stage 2] hop={hop+1}, follow-up: {follow_up}, total docs: {len(merged)}"
            }]
        }

    # ---- Adaptive Retrieval Router ----
    # BC retrieval router (lazy-loaded; falls back to hard rule if no policy)
    _retrieval_router_bc = RetrievalRouterBC(
        policy_path=os.path.join(os.path.dirname(__file__), "retrieval_router_policy.pt")
    )

    def retrieval_router_node(state: AgentState) -> AgentState:
        """
        BC 检索路由：检索后、生成前判断检索质量。
        - 优先使用训练好的 BC 分类器（RetrievalRouterBC）
        - 无 policy 时退化为硬规则：ctxP < 0.2 → poor
        """
        docs = state.get("docs") or []
        ctx_prec = float(state.get("context_precision") or 0.0)
        ctx_rec  = float(state.get("context_recall") or 0.0)
        doc_count = len(docs)

        quality = _retrieval_router_bc.decide(ctxP=ctx_prec, ctxR=ctx_rec, doc_count=doc_count)

        logger = _ensure_logger_on_state(state)
        if logger:
            logger.add_reason(f"[retrieval_router] quality={quality} doc_count={doc_count} ctxP={ctx_prec:.2f}")

        return {**state, "retrieval_quality": quality}

    # ---- 构建图 ----
    workflow = StateGraph(AgentState)
    workflow.add_node("query_optimizer", query_optimizer)
    workflow.add_node("retriever", retriever)
    workflow.add_node("generator", generator)
    workflow.add_node("router", router)
    workflow.add_node("requery_optimizer", requery_optimizer)
    workflow.add_node("finalizer", finalizer)

    if use_adaptive_retrieval:
        # Adaptive graph:
        # IRCoT first → retrieval_router → ok: generator | poor: PAR2 fallback → generator
        # query_optimizer → retriever → retrieval_router → {ok→generator, poor→anchor_node}
        # anchor_node → refine_node ⟲ → generator → router → {end→finalizer, ...}
        workflow.add_node("retrieval_router", retrieval_router_node)
        workflow.add_node("anchor_node", anchor_node)
        workflow.add_node("refine_node", refine_node)

        workflow.add_edge("query_optimizer", "retriever")
        workflow.add_edge("retriever", "retrieval_router")
        workflow.add_conditional_edges(
            "retrieval_router",
            lambda st: st.get("retrieval_quality", "ok"),
            {"ok": "generator", "poor": "anchor_node"}
        )
        workflow.add_edge("anchor_node", "refine_node")
        workflow.add_conditional_edges(
            "refine_node",
            lambda st: st.get("next_step", "generate"),
            {"refine": "refine_node", "generate": "generator"}
        )
        workflow.add_edge("generator", "router")
        workflow.add_conditional_edges(
            "router",
            lambda st: st["next_step"],
            {"end": "finalizer", "regenerate": "generator", "requery": "requery_optimizer"}
        )
        workflow.add_edge("requery_optimizer", "retriever")

    elif use_par2:
        # Pure PAR2-RAG graph (kept for backward compat):
        # query_optimizer → anchor_node → refine_node ⟲ → generator → router → finalizer
        workflow.add_node("anchor_node", anchor_node)
        workflow.add_node("refine_node", refine_node)

        workflow.add_edge("query_optimizer", "anchor_node")
        workflow.add_edge("anchor_node", "refine_node")
        workflow.add_conditional_edges(
            "refine_node",
            lambda st: st.get("next_step", "generate"),
            {"refine": "refine_node", "generate": "generator"}
        )
        workflow.add_edge("generator", "router")
        workflow.add_conditional_edges(
            "router",
            lambda st: st["next_step"],
            {"end": "finalizer", "regenerate": "generator", "requery": "finalizer"}
        )
    else:
        # Original IRCoT-only graph
        workflow.add_edge("query_optimizer", "retriever")
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("generator", "router")
        workflow.add_conditional_edges(
            "router",
            lambda st: st["next_step"],
            {"end": "finalizer", "regenerate": "generator", "requery": "requery_optimizer"}
        )
        workflow.add_edge("requery_optimizer", "retriever")

    workflow.add_edge("finalizer", END)
    workflow.set_entry_point("query_optimizer")
    return workflow.compile()


# -----------------------------
# IRCoT iterative multi-hop retrieval
# -----------------------------
def _ircot_retrieve(question, retrieval_agent, llm, max_hops=3, top_k_per_hop=5, logger=None):
    """IRCoT 迭代检索：每一跳从已检索文档中提取中间推理，构造下一跳 query。"""
    all_docs = []
    seen_contents = set()
    reasoning_trace = ""

    for hop in range(max_hops):
        if hop == 0:
            hop_query = question
        else:
            # 根据已有信息生成下一跳 query
            cot_prompt = (
                "Based on the question and information gathered so far, "
                "determine what to search for next.\n\n"
                f"Question: {question}\n\n"
                f"Information gathered so far:\n{reasoning_trace}\n\n"
                "What specific entity, fact, or relationship do we still need to find? "
                "Write a SHORT search query (under 20 words) to find this missing information. "
                "If we already have enough information, respond with exactly: DONE\n\n"
                "Search query:"
            )
            resp = llm.invoke(cot_prompt)
            hop_query = (getattr(resp, "content", None) or str(resp)).strip()
            if "DONE" in hop_query.upper():
                break

        # 检索（不传 reference，中间跳无需评估）
        retrieval_agent.set_top_k(top_k_per_hop)
        ret = retrieval_agent.retrieve(hop_query) or {}
        hop_docs = ret.get("docs", [])

        # 去重累积
        for d in hop_docs:
            content = getattr(d, "page_content", None) or str(d)
            if content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(d)

        if not hop_docs:
            break

        # 从本跳结果中提取中间推理
        snippets = "\n".join(
            (getattr(d, "page_content", "") or "")[:300] for d in hop_docs[:3]
        )
        extract_prompt = (
            f"Question: {question}\n\n"
            f"New retrieved information:\n{snippets}\n\n"
            f"Previously gathered:\n{reasoning_trace or '(none)'}\n\n"
            "Write a brief note (1-2 sentences): what do we now know, "
            "and what's still missing to answer the question?"
        )
        resp = llm.invoke(extract_prompt)
        hop_reasoning = (getattr(resp, "content", None) or str(resp)).strip()
        reasoning_trace += f"\n[Hop {hop+1}] {hop_reasoning}"

        if logger and getattr(logger, "started", False):
            logger.add_reason(f"[ircot.hop{hop+1}] query='{hop_query}' → {len(hop_docs)} docs")

    return all_docs


# -----------------------------
# Run RAG process
# -----------------------------
def run_rag_pipeline(
    question: str,
    retrieval_agent,
    reasoning_agent,
    generation_agent,
    evaluation_agent,
    **kwargs
) -> Dict[str, Any]:

    reference = kwargs.get("reference", None)
    use_router: bool = bool(kwargs.get("use_router", False))
    visualize: bool = bool(kwargs.get("visualize", False))



    # 轨迹 & 初始状态
    import uuid, time
    qid = kwargs.get("qid") or str(uuid.uuid4())

    # === 统一 logger 生命周期 ===
    logger = kwargs.get("logger", None)

    # 如果外面没传，自己创建一个
    if logger is None and TrajectoryLogger is not None:
        try:
            logger = TrajectoryLogger(out_dir=kwargs.get("traj_out_dir", "runs/trajectories"))
        except Exception as e:
            print(f"[logger.init.error] {type(e).__name__}: {e}")
            logger = None

    # 如果有 logger，但还没 start，这里统一 start 一次
    if logger is not None and getattr(logger, "started", False) is False:
        try:
            logger.start(qid=qid, query_raw=question)
        except Exception as e:
            print(f"[logger.start.error] {type(e).__name__}: {e}")
            logger = None

    # 下面保持你原来的 _log_safe 和 ref/model 记录逻辑……
    def _log_safe(msg: str):
        try:
            if logger and getattr(logger, "started", False):
                logger.add_reason(msg)
        except Exception:
            pass

    if logger and getattr(logger, "started", False):
        try:
            logger.set_reference(kwargs.get("reference"))
        except Exception:
            pass
        try:
            llm = getattr(generation_agent, "llm", None)
            model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
            base_url   = getattr(llm, "base_url", None) or getattr(getattr(llm, "client", None), "base_url", None)
            ctx_tokens = getattr(generation_agent, "max_ctx_tokens", None)
            gen_tokens = getattr(generation_agent, "max_gen_tokens", None)
            logger.add_model_ident(model=model_name, base_url=base_url,
                                   ctx_tokens=ctx_tokens, gen_tokens=gen_tokens)
        except Exception:
            pass



    # === Policy 路由（可选，先初始化，不要遮蔽 logger）===
    router = None
    if use_router:
        try:
            from agents.RLRouterAgent import RLRouterAgent as RouterCls  # 用别名，避免局部重绑全局名
            router_device = kwargs.get("router_device", "cpu")
            policy_path = (
                kwargs.get("router_policy_path")
                or os.getenv("ROUTER_POLICY_PATH")
                or kwargs.get("policy_path")  # 兼容你之前的参数名
                or POLICY_SAVE_PATH           # 默认 agents/router_policy.pt
            )
            router = RouterCls(policy_path=policy_path, device=router_device, logger=logger)
            _log_safe(f"[router.init] policy_path={policy_path} ready={router is not None}")
        except Exception as e:
            print(f"[router.init.warning] {type(e).__name__}: {e}")
            _log_safe(f"[router.init.warning] {type(e).__name__}: {e}")
            router = None



    # 如果用 Router，尝试构图；失败则回落直通
    graph = None
    if use_router:
        print("🚦 Using LangGraph StateGraph with router")
        try:
            _use_par2 = bool(kwargs.get("use_par2", False))
            _use_adaptive = bool(kwargs.get("use_adaptive_retrieval", False))
            _par2_llm = kwargs.get("par2_llm") or getattr(generation_agent, "llm", None)
            graph = create_rag_graph(
                retrieval_agent, reasoning_agent, generation_agent, evaluation_agent,
                rl_router=router,
                use_par2=_use_par2,
                use_adaptive_retrieval=_use_adaptive,
                par2_llm=_par2_llm,
                par2_n_subqueries=int(kwargs.get("par2_n_subqueries", 5)),
                par2_max_hops=int(kwargs.get("par2_max_hops", 4)),
            )
            if visualize and graph is not None:
                try:
                    from IPython.display import display
                    display(graph.get_graph().draw_mermaid_png())
                except Exception as e:
                    print(f"无法生成可视化: {e}")
        except Exception as e:
            print(f"⚠️ 无法创建路由图（将回退直通模式）: {e}")
            graph = None
            use_router = False

        # === NEW: 如果成功构建了 LangGraph，就走多步 StateGraph + Router 模式 ===
    if use_router and graph is not None:
        print("[run_rag_pipeline] 🚀 Using LangGraph StateGraph + router")

        # 构造初始 AgentState
        init_state: AgentState = {
            "question": question,
            "conversation_context": kwargs.get("conversation_context") or [],
            "refined_query": "",
            "docs": [],
            "answer": "",
            "faithfulness_score": 0.0,
            "response_relevancy": 0.0,
            "noise_sensitivity": 1.0,
            "semantic_f1_score": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "attempts": 0,
            "next_step": "end",
            "messages": [],
            "error": None,
            "start_time": time.time(),
            "metrics": {},
            "requery_count": 0,
            "regenerate_count": 0,
            "max_attempts": int(kwargs.get("max_attempts", 6)),
            "max_regenerates": int(kwargs.get("max_regenerates", 2)),
            "max_requeries": int(kwargs.get("max_requeries", 2)),
            "reference": reference,
            "qid": qid,
            "logger": logger,
            "refine_hop": 0,
            "par2_max_hops": int(kwargs.get("par2_max_hops", 4)),
            "used_follow_ups": [],
            "retrieval_quality": "ok",
        }


        # 运行 LangGraph 多步流程
        final_state: AgentState = graph.invoke(init_state)

        # 为了兼容 evaluate_dataset_real，只需把关键字段打包返回
        result: Dict[str, Any] = {
            "question": final_state.get("question", question),
            "refined_query": final_state.get("refined_query", question),
            "docs": final_state.get("docs", []),
            "answer": final_state.get("answer", ""),
            "faithfulness_score": final_state.get("faithfulness_score", 0.0),
            "response_relevancy": final_state.get("response_relevancy", 0.0),
            "noise_sensitivity": final_state.get("noise_sensitivity", 1.0),
            "semantic_f1_score": final_state.get("semantic_f1_score", 0.0),
            "context_recall": final_state.get("context_recall", 0.0),
            "context_precision": final_state.get("context_precision", 0.0),
            "metrics": final_state.get("metrics", {}),
            "routing_decision": final_state.get("retrieval_quality", "n/a"),
        }

        # LangGraph 的 finalizer 里已经会 commit logger，这里不用再管
        return result


    # ---------------- 直通模式（完整替换块） ----------------
    if not use_router or graph is None:
        t0 = time.time()
        metrics: Dict[str, Any] = {}

    # 1) Reasoning → refined_query
    try:
        rstart = time.time()
        plan_out = reasoning_agent.plan(
            user_question=question,
            retrieved_docs=None,
            conversation_context=kwargs.get("conversation_context") or []
        )
        refined_query = plan_out.get("refined_query") or question
        fallback = bool(plan_out.get("fallback", False))
        metrics["query_optimization_time"] = round((time.time() - rstart) * 1000.0, 2)
        if logger and getattr(logger, "started", False):
            logger.add_reason(f"[pipeline] refined_query={refined_query} (fallback={fallback})")
            logger.set_refined_query(refined_query)  # CHANGE
    except Exception as e:
        if logger and getattr(logger, "started", False):
            logger.add_reason(f"[pipeline.error] reasoning: {e}")
        refined_query, fallback = question, True

    # 2) Retrieval
    try:
        q_for_ret = refined_query if refined_query else question  # always use refined (pronoun-resolved) query
        ret = retrieval_agent.retrieve(q_for_ret, reference=reference) or {}
        docs = ret.get("docs", [])
        metrics["retrieval_time"]    = round(ret.get("latency_ms", 0.0), 2)
        metrics["doc_count"]         = len(docs)
        metrics["context_precision"] = extract_scalar(ret.get("context_precision"))
        metrics["context_recall"]    = extract_scalar(ret.get("context_recall"))
        if logger and getattr(logger, "started", False):
            logger.add_eval(
                context_precision=metrics["context_precision"],
                context_recall=metrics["context_recall"],
                doc_count=metrics["doc_count"],
            )
    except Exception as e:
        if logger and getattr(logger, "started", False):
            logger.add_reason(f"[pipeline.error] retrieval: {e}")
        docs = []
        metrics.update({
            "retrieval_time": 0.0,
            "doc_count": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        })
        if logger and getattr(logger, "started", False):
            logger.add_eval(context_precision=0.0, context_recall=0.0, doc_count=0.0)

    # 2b) 多策略检索重试
    ctx_recall = extract_scalar(metrics.get("context_recall", 0.0) or 0.0)
    need_retry = (not docs) or (ctx_recall < 0.5)
    metrics["retrieval_retry_triggered"] = bool(need_retry)

    if need_retry:
        r2_t0 = time.time()
        base_q = (refined_query or question or "").strip()
        if logger and getattr(logger, "started", False):
            logger.add_reason(f"[retrieval.retry] prev_ctxR={ctx_recall:.2f}")

        def _score_tuple(r):
            try:
                r_ctxR = extract_scalar(r.get("context_recall", 0.0) or 0.0)
                r_ctxP = extract_scalar(r.get("context_precision", 0.0) or 0.0)
            except Exception:
                r_ctxR, r_ctxP = 0.0, 0.0
            r_docs = len(r.get("docs", []) or [])
            return (r_ctxR, r_ctxP, r_docs)

        best_ret = ret
        best_score = _score_tuple(ret)
        took_retry = False
        prev_k = getattr(retrieval_agent, "top_k", 3)
        llm = getattr(generation_agent, "llm", None)

        # --- 策略 1: IRCoT 迭代多跳检索 ---
        ircot_docs = []
        if llm is not None:
            try:
                ircot_docs = _ircot_retrieve(
                    question=base_q,
                    retrieval_agent=retrieval_agent,
                    llm=llm,
                    max_hops=3,
                    top_k_per_hop=min(6, max(1, int(prev_k))),
                    logger=logger,
                )
            except Exception as e:
                if logger and getattr(logger, "started", False):
                    logger.add_reason(f"[retrieval.retry.ircot.error] {e}")

        if ircot_docs:
            # 与初始检索文档合并去重
            merged = list(docs) + ircot_docs
            seen = set()
            deduped = []
            for d in merged:
                content = getattr(d, "page_content", None) or str(d)
                if content not in seen:
                    seen.add(content)
                    deduped.append(d)
            # 评估合并后的文档
            ret_ircot = {"docs": deduped, "context_precision": 0.0, "context_recall": 0.0}
            if evaluation_agent is not None and reference:
                try:
                    eval_res = evaluation_agent.evaluate_retrieval(
                        user_query=base_q, retrieved_docs=deduped, reference=reference
                    )
                    ret_ircot["context_precision"] = extract_scalar(eval_res.get("context_precision", 0.0) or 0.0)
                    ret_ircot["context_recall"] = extract_scalar(eval_res.get("context_recall", 0.0) or 0.0)
                except Exception:
                    pass
            s_ircot = _score_tuple(ret_ircot)
            if s_ircot > best_score:
                best_ret = ret_ircot
                best_score = s_ircot
                took_retry = True
                if logger and getattr(logger, "started", False):
                    logger.add_reason(f"[retrieval.retry.ircot] improved ctxR={s_ircot[0]:.2f}")

        # --- 策略 2: 关键词回退 ---
        if best_score[0] < 0.5:
            kw_query = ""
            if llm is not None:
                try:
                    kw_prompt = (
                        "Extract the 2-4 most important named entities or keywords "
                        "from this question. Return them as a single search query.\n\n"
                        f"Question: {base_q}"
                    )
                    resp = llm.invoke(kw_prompt)
                    kw_query = (getattr(resp, "content", None) or str(resp)).strip()
                except Exception:
                    pass
            if not kw_query:
                kw_query = " ".join(base_q.split()[:8]) if base_q else (question or "")

            if logger and getattr(logger, "started", False):
                logger.add_reason(f"[retrieval.retry.keyword] kw_query='{kw_query}'")

            try:
                retrieval_agent.set_top_k(min(12, max(1, int(prev_k) + 4)))
                ret_kw = retrieval_agent.retrieve(kw_query, reference=reference) or {}
                s_kw = _score_tuple(ret_kw)
                if s_kw > best_score:
                    best_ret = ret_kw
                    best_score = s_kw
                    took_retry = True
                    if logger and getattr(logger, "started", False):
                        logger.add_reason(f"[retrieval.retry.keyword] improved ctxR={s_kw[0]:.2f}")
            except Exception as e:
                if logger and getattr(logger, "started", False):
                    logger.add_reason(f"[retrieval.retry.keyword.error] {e}")

        try:
            retrieval_agent.set_top_k(prev_k)
        except Exception:
            pass

        ret = best_ret
        metrics["retrieval_retry_taken"] = bool(took_retry)
        metrics["retrieval_retry_time"] = round((time.time() - r2_t0) * 1000.0, 2)

        docs = ret.get("docs", [])
        metrics["doc_count"]         = len(docs)
        metrics["context_precision"] = extract_scalar(ret.get("context_precision", 0.0) or 0.0)
        metrics["context_recall"]    = extract_scalar(ret.get("context_recall", 0.0) or 0.0)
        if logger and getattr(logger, "started", False):
            logger.add_eval(
                context_precision=metrics["context_precision"],
                context_recall=metrics["context_recall"],
                doc_count=float(metrics["doc_count"]),
            )

    # 3) Generation（内部会评估）
    try:
        gstart = time.time()
        gen = generation_agent.answer(
            question=question,
            docs=docs,
            evaluation_agent=evaluation_agent,
            ground_truth=reference,
            max_attempts=int(kwargs.get("gen_max_attempts", 3)),
            prompt_id="gen_v1"
        ) or {}

        def _num(x, default=0.0):
            try:
                return float(extract_scalar(x))
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return float(default)

        answer = gen.get("answer", "")

        faith = _num(gen.get("faithfulness_score", 0.0), 0.0)

        rel_val = gen.get("response_relevancy", None)
        if rel_val is None:
            rel_val = gen.get("answer_relevancy", 0.0)
        rel   = _num(rel_val, 0.0)

        noise = _num(gen.get("noise_sensitivity", 1.0), 1.0)

        semf1_val = gen.get("semantic_f1_score", gen.get("semantic_f1", 0.0))
        semf1 = _num(semf1_val, 0.0)

        metrics["generation_time"] = round((time.time() - gstart) * 1000.0, 2)

        # 把生成阶段的分数先写入轨迹
        if logger and getattr(logger, "started", False):
            logger.add_eval(
                faith=faith,
                response_relevancy=rel,
                noise_sensitivity=noise,
                semantic_f1=semf1,
                doc_count=float(metrics.get("doc_count", 0.0))
            )

    except Exception as e:
        if logger and getattr(logger, "started", False):
            logger.add_reason(f"[pipeline.error] generation: {e}")
        print(f"[pipeline.error] generation: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        answer, faith, rel, noise, semf1 = (
            "Sorry, I encountered an issue while generating an answer.", 0.0, 0.0, 1.0, 0.0
        )
        metrics["generation_time"] = 0.0
        if logger and getattr(logger, "started", False):
            logger.add_eval(
                faith=0.0,
                response_relevancy=0.0,
                noise_sensitivity=1.0,
                semantic_f1=0.0,
                doc_count=0.0
            )

    # 4) === Policy 路由（一次性动作；安全兜底） ===
    try:
        if router is not None:
            decision_state = {
                "context_precision": metrics.get("context_precision", 0.0),
                "context_recall":   metrics.get("context_recall", 0.0),
                "faithfulness_score": faith,
                "response_relevancy": rel,
                "noise_sensitivity":  noise,
                "semantic_f1_score":  semf1,
            }
            action = router.decide(decision_state, greedy=True)
            print(f"[router] action={action}")

            # 记录路由动作到轨迹
            if logger and getattr(logger, "started", False):
                try:
                    logger.set_router_action(action)
                    logger.add_eval(
                        context_precision=decision_state["context_precision"],
                        context_recall=decision_state["context_recall"],
                        faith=decision_state["faithfulness_score"],
                        response_relevancy=decision_state["response_relevancy"],
                        noise_sensitivity=decision_state["noise_sensitivity"],
                        semantic_f1=decision_state["semantic_f1_score"],
                        doc_count=float(metrics.get("doc_count", 0.0))
                    )
                except Exception as e:
                    print(f"[router.decide.warning] {type(e).__name__}: {e}")

            # —— 执行动作（只执行一次，避免复杂循环）——
            if action == "requery":
                rq_t0 = time.time()
                base_q = (refined_query or question or "").strip()
                short_q = " ".join(base_q.split()[:8]) if base_q else (question or "")
                prev_k = getattr(retrieval_agent, "top_k", 3)
                try:
                    retrieval_agent.set_top_k(min(8, max(1, int(prev_k) + 2)))
                except Exception:
                    pass
                try:
                    ret3 = retrieval_agent.retrieve(short_q, reference=reference) or {}
                except Exception as e:
                    print(f"[router.requery.error] {type(e).__name__}: {e}")
                    ret3 = {"docs": [], "context_precision": 0.0, "context_recall": 0.0, "latency_ms": 0.0}
                try:
                    retrieval_agent.set_top_k(prev_k)
                except Exception:
                    pass

                docs2 = ret3.get("docs", [])

                def _score_tuple2(r):
                    try:
                        r_ctxR = extract_scalar(r.get("context_recall", 0.0) or 0.0)
                        r_ctxP = extract_scalar(r.get("context_precision", 0.0) or 0.0)
                    except Exception:
                        r_ctxR, r_ctxP = 0.0, 0.0
                    r_docs = len(r.get("docs", []) or [])
                    return (r_ctxR, r_ctxP, r_docs)

                if _score_tuple2(ret3) > _score_tuple2({"context_precision": metrics.get("context_precision",0.0),
                                                        "context_recall": metrics.get("context_recall",0.0),
                                                        "docs": docs}):
                    docs = docs2
                    metrics["context_precision"] = extract_scalar(ret3.get("context_precision", 0.0) or 0.0)
                    metrics["context_recall"]    = extract_scalar(ret3.get("context_recall", 0.0) or 0.0)
                    metrics["retrieval_time"]    = round(ret3.get("latency_ms", 0.0), 2)
                    metrics["doc_count"]         = len(docs)
                    if logger and getattr(logger, "started", False):
                        logger.add_eval(
                            context_precision=metrics["context_precision"],
                            context_recall=metrics["context_recall"],
                            doc_count=float(metrics.get("doc_count", 0.0))
                        )

                    # 重新生成一次（温和重试）
                    g2_t0 = time.time()
                    gen2 = generation_agent.answer(
                        question=question,
                        docs=docs,
                        evaluation_agent=evaluation_agent,
                        ground_truth=reference,
                        max_attempts=int(kwargs.get("gen_max_attempts", 2)),
                        prompt_id="gen_v1_requery"
                    ) or {}

                    def _num2(x, default=0.0):
                        try:
                            return float(extract_scalar(x))
                        except Exception:
                            try:
                                return float(x)
                            except Exception:
                                return float(default)

                    new_answer = gen2.get("answer", "")
                    if new_answer:
                        answer = new_answer
                    faith = _num2(gen2.get("faithfulness_score", faith), faith)
                    r_tmp  = gen2.get("response_relevancy", gen2.get("answer_relevancy", rel))
                    rel   = _num2(r_tmp, rel)
                    noise = _num2(gen2.get("noise_sensitivity", noise), noise)
                    s_tmp = gen2.get("semantic_f1_score", gen2.get("semantic_f1", semf1))
                    semf1 = _num2(s_tmp, semf1)
                    metrics["generation_time"] += round((time.time() - g2_t0) * 1000.0, 2)

            elif action == "regenerate":
                g2_t0 = time.time()
                gen2 = generation_agent.answer(
                    question=question,
                    docs=docs,
                    evaluation_agent=evaluation_agent,
                    ground_truth=reference,
                    max_attempts=int(kwargs.get("gen_max_attempts", 2)),
                    prompt_id="gen_v1_retry"
                ) or {}

                def _num3(x, default=0.0):
                    try:
                        return float(extract_scalar(x))
                    except Exception:
                        try:
                            return float(x)
                        except Exception:
                            return float(default)

                new_answer = gen2.get("answer", "")
                faith2 = _num3(gen2.get("faithfulness_score", 0.0), 0.0)
                rel2_v = gen2.get("response_relevancy", gen2.get("answer_relevancy", 0.0))
                rel2   = _num3(rel2_v, 0.0)
                noise2 = _num3(gen2.get("noise_sensitivity", 1.0), 1.0)
                semf12_v = gen2.get("semantic_f1_score", gen2.get("semantic_f1", 0.0))
                semf12 = _num3(semf12_v, 0.0)
                better = (faith2 + rel2 + semf12 - noise2) > (faith + rel + semf1 - noise)
                if better and new_answer:
                    answer, faith, rel, noise, semf1 = new_answer, faith2, rel2, noise2, semf12
                    metrics["generation_time"] += round((time.time() - g2_t0) * 1000.0, 2)

    except Exception as e:
        print(f"[router.block.warning] {type(e).__name__}: {e}")

    # 5) 汇总 & 落盘
    metrics["total_time"] = round((time.time() - t0) * 1000.0, 2)

    result = {
        "question": question,
        "refined_query": refined_query,
        "docs": docs,
        "answer": answer,
        "faithfulness_score": faith,
        "response_relevancy": rel,
        "noise_sensitivity": noise,
        "semantic_f1_score": semf1,
        "context_recall": metrics.get("context_recall", 0.0),
        "context_precision": metrics.get("context_precision", 0.0),
        "metrics": metrics
    }

    print(f"\n✅ 最终答案: {result['answer']}")
    if reference:
        print(f"📖 Ground Truth: {reference}")
    print(f"📊 Faithfulness: {result['faithfulness_score']:.2f}, "
            f"Relevancy: {result['response_relevancy']:.2f}, "
            f"Noise: {result['noise_sensitivity']:.2f}")
    if "semantic_f1_score" in result:
        print(f"🎯 Semantic F1: {result['semantic_f1_score']:.2f}")

    print("\n📈 Performance metrics:")
    for metric, value in result["metrics"].items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.2f}")
        else:
            print(f"  - {metric}: {value}")

    if logger and getattr(logger, "started", False):
        try:
            logger.set_final_answer(result.get("answer", ""))
            logger.add_eval(
                faith=result.get("faithfulness_score", 0.0),
                response_relevancy=result.get("response_relevancy", 0.0),
                noise_sensitivity=result.get("noise_sensitivity", 1.0),
                semantic_f1=result.get("semantic_f1_score", 0.0),
                context_precision=result.get("context_precision", 0.0),
                context_recall=result.get("context_recall", 0.0),
                doc_count=float(metrics.get("doc_count", 0.0)),
            )
        except Exception:
            pass
        try:
            logger.commit()
        except Exception as e:
            print(f"[logger.commit.error] {e}")

    return result
