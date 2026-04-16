# agents/retrieval_agent.py
import time
from typing import Optional, Dict, Any, List, Callable, Tuple, Union

try:
    from langchain.schema import Document as LC_Document
except Exception:
    LC_Document = None

from utils.trajectory_logger import TrajectoryLogger


Number = Union[int, float]


def _to_float_safe(x: Any) -> Optional[float]:
    try:
        # 兼容 "0.42" / numpy 标量
        return float(x)
    except Exception:
        return None


class RetrievalAgent:
    def __init__(
        self,
        vectorstore,
        evaluation_agent,
        top_k: int = 5,
        logger: Optional[TrajectoryLogger] = None,
        reranker: Optional[Callable[[List], List]] = None,
        dedupe: bool = True,
        min_score: Optional[float] = None,   # 🔹可选：过滤低分命中
        obs_snippet_len: int = 200,          # 🔹记录到 logger 的摘要长度
        hybrid_retriever=None,               # 🔹HybridRetriever 实例
        multi_query_fn: Optional[Callable] = None,  # 🔹query -> [query_variants]
    ):
        """
        Args:
            vectorstore: LangChain VectorStore（已建好索引）
            evaluation_agent: 你的评估器，需提供 evaluate_retrieval()
            top_k: 默认 Top-K
            logger: 轨迹记录器（可为空）
            reranker: 可选重排函数：docs -> docs（同类型列表）
            dedupe: 是否按文档 id 去重
            min_score: 若提供，则过滤 score < min_score 的文档（基于 metadata/attr）
            obs_snippet_len: 写入日志的片段长度上限
        """
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.evaluation_agent = evaluation_agent
        self.top_k = top_k
        self.logger = logger
        self.reranker = reranker
        self.dedupe = dedupe
        self.min_score = min_score
        self.obs_snippet_len = max(0, int(obs_snippet_len))
        self.hybrid_retriever = hybrid_retriever
        self.multi_query_fn = multi_query_fn

    # ---- 文本 / 元信息抽取 ----
    def _doc_text(self, d) -> str:
        if d is None:
            return ""
        txt = getattr(d, "page_content", None)
        if isinstance(txt, str):
            return txt
        if isinstance(d, dict):
            v = d.get("page_content") or d.get("text") or d.get("content")
            if isinstance(v, str):
                return v
        if isinstance(d, str):
            return d
        return ""

    def _doc_meta(self, d) -> Dict[str, Any]:
        if d is None:
            return {}
        m = getattr(d, "metadata", None)
        if isinstance(m, dict):
            return m
        if isinstance(d, dict):
            v = d.get("metadata") or {}
            if isinstance(v, dict):
                return v
        return {}

    # ---- 标准化输入结构 ----
    def _normalize_docs(self, raw) -> List:
        """
        接受：
          - List[Document] / List[dict] / List[str]
          - dict: {"documents": [... or list[list]], "metadatas": [...]}
          - tuple: (dict_like, extra) 或 (list_like, extra)
        返回：
          - List[Document 风格对象]
        """
        # tuple: 取第一个
        if isinstance(raw, tuple) and raw:
            raw = raw[0]

        docs: List = []
        if raw is None:
            return docs

        # dict 结构
        if isinstance(raw, dict):
            if "documents" in raw:
                maybe_docs = raw.get("documents") or []
                maybe_metas = raw.get("metadatas") or []
                # 展平二维：部分 retriever 批量返回 list[list[str]]
                if maybe_docs and isinstance(maybe_docs[0], list):
                    flat_docs: List = []
                    flat_metas: List = []
                    for i, row in enumerate(maybe_docs):
                        # 对应的 metas 行
                        metas_row = maybe_metas[i] if (isinstance(maybe_metas, list) and i < len(maybe_metas)) else []
                        # 拉平
                        for j, content in enumerate(row):
                            m = metas_row[j] if (isinstance(metas_row, list) and j < len(metas_row) and isinstance(metas_row[j], dict)) else {}
                            flat_docs.append(content)
                            flat_metas.append(m)
                    maybe_docs, maybe_metas = flat_docs, flat_metas

                # 一维对齐
                for i, content in enumerate(maybe_docs):
                    meta = {}
                    if isinstance(maybe_metas, list) and i < len(maybe_metas) and isinstance(maybe_metas[i], dict):
                        meta = maybe_metas[i]
                    text = self._doc_text(content)
                    if LC_Document is not None:
                        docs.append(LC_Document(page_content=text, metadata=meta))
                    else:
                        docs.append({"page_content": text, "metadata": meta})
                return docs

            # 其他常见键：{"docs": [...]} / {"results": [...]}
            maybe = raw.get("docs") or raw.get("results")
            if isinstance(maybe, list):
                raw = maybe  # 继续 list 分支

        # list 结构
        if isinstance(raw, list):
            for item in raw:
                if LC_Document is not None and hasattr(item, "page_content"):
                    docs.append(item)  # 已是 LC_Document
                elif isinstance(item, dict) and ("page_content" in item or "text" in item or "content" in item):
                    docs.append({"page_content": self._doc_text(item), "metadata": self._doc_meta(item)})
                elif isinstance(item, str):
                    if LC_Document is not None:
                        docs.append(LC_Document(page_content=item, metadata={}))
                    else:
                        docs.append({"page_content": item, "metadata": {}})
                else:
                    # 兜底抽文本
                    txt = self._doc_text(item)
                    if txt:
                        if LC_Document is not None:
                            docs.append(LC_Document(page_content=txt, metadata={}))
                        else:
                            docs.append({"page_content": txt, "metadata": {}})
            return docs

        # 其他类型（单个对象）
        txt = self._doc_text(raw)
        if txt:
            if LC_Document is not None:
                docs.append(LC_Document(page_content=txt, metadata={}))
            else:
                docs.append({"page_content": txt, "metadata": {}})
        return docs

    # ---- 统一 doc id ----
    def _doc_id_of(self, d) -> str:
        meta = getattr(d, "metadata", {}) or {}
        # 常见 id 字段
        for key in ("id", "_id", "doc_id"):
            if key in meta and meta[key]:
                return str(meta[key])
        from hashlib import sha256
        pc = (getattr(d, "page_content", "") or "")
        base = pc[:256] if pc else repr(meta)[:256]
        return sha256(base.encode("utf-8")).hexdigest()[:16]

    # ---- 命中摘要 ----
    def _hits_meta(self, docs: List) -> List[Dict[str, Any]]:
        hits = []
        for d in docs:
            doc_id = self._doc_id_of(d)
            score = None
            meta = getattr(d, "metadata", {}) or {}
            if "score" in meta:
                score = _to_float_safe(meta["score"])
            elif hasattr(d, "score"):
                score = _to_float_safe(getattr(d, "score"))
            hits.append({"doc_id": doc_id, "score": score})
        return hits

    # ---- 去重（可选：保留高分） ----
    def _dedupe_docs(self, docs: List) -> List:
        if not self.dedupe:
            return docs
        seen: Dict[str, Any] = {}
        for d in docs:
            did = self._doc_id_of(d)
            cur_score = None
            meta = getattr(d, "metadata", {}) or {}
            if "score" in meta:
                cur_score = _to_float_safe(meta["score"])
            elif hasattr(d, "score"):
                cur_score = _to_float_safe(getattr(d, "score"))
            if did not in seen:
                seen[did] = (d, cur_score)
            else:
                # 若有分数则保留分数更高的
                _, old_score = seen[did]
                if (cur_score is not None) and (old_score is None or cur_score > old_score):
                    seen[did] = (d, cur_score)
        return [pair[0] for pair in seen.values()]

    # ---- 过滤低分 ----
    def _filter_by_min_score(self, docs: List) -> List:
        if self.min_score is None:
            return docs
        out = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            score = meta.get("score", getattr(d, "score", None))
            score_f = _to_float_safe(score)
            if score_f is None or score_f >= self.min_score:
                out.append(d)
        return out

    # ---- 动态修改 k ----
    def set_top_k(self, k: int):
        self.top_k = max(1, int(k))
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    # ---- 主流程 ----
    def retrieve(self, query: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Single-round retrieval logic, controlled by the upper layer router whether to retry.
        Returns:
          {
            "docs": [...],
            "context_precision": float,
            "context_recall": float,
            "latency_ms": float,
            "hits_meta": [{"doc_id":..., "score":...}, ...],
          }
        """
        t0 = time.time()

        # 1) Build list of queries (multi-query expansion if configured)
        queries = [query]
        if self.multi_query_fn:
            try:
                queries = self.multi_query_fn(query)
            except Exception:
                queries = [query]

        # 2) Retrieve documents for each query variant, merge results
        try:
            all_docs = []
            for q in queries:
                if self.hybrid_retriever is not None:
                    # Hybrid BM25 + FAISS path
                    q_docs = self.hybrid_retriever.retrieve(q, k=self.top_k)
                    all_docs.extend(q_docs)
                else:
                    # Original FAISS-only path
                    try:
                        raw = self.retriever.invoke(q)
                    except Exception:
                        raw = self.retriever.get_relevant_documents(q)
                    all_docs.extend(self._normalize_docs(raw))
            docs = all_docs
        except Exception as e:
            if self.logger:
                self.logger.add_tool_call(type="retrieval_error", query=query, topk=self.top_k, error=str(e))
            return {
                "docs": [],
                "context_precision": 0.0,
                "context_recall": 0.0,
                "latency_ms": (time.time() - t0) * 1000.0,
                "hits_meta": []
            }

        # 3) 可选重排
        if self.reranker and docs:
            try:
                docs = self.reranker(query, docs)
            except TypeError:
                # Fallback: old-style reranker that takes only docs
                docs = self.reranker(docs)
            except Exception:
                pass

        # 4) 去重 + 低分过滤
        docs = self._dedupe_docs(docs or [])
        docs = self._filter_by_min_score(docs)

        latency_ms = (time.time() - t0) * 1000.0
        hits_meta = self._hits_meta(docs)

        # 5) 记录调用
        if self.logger:
            self.logger.add_tool_call(
                type="retrieval",
                query=query,
                topk=self.top_k,
                latency_ms=round(latency_ms, 2),
                hits=hits_meta
            )
            for d in docs[: self.top_k]:
                snippet = (self._doc_text(d) or "")[: self.obs_snippet_len]
                if snippet:
                    self.logger.add_observation(snippet, do_hash=True)

        if not docs:
            print("⚠️ No documents retrieved")
            return {
                "docs": [],
                "context_precision": 0.0,
                "context_recall": None,   # 改为 None
                "weak_recall": None,      # 新增
                "latency_ms": latency_ms,
                "hits_meta": hits_meta
            }

        # 6) 评估检索效果（ctx-P / ctx-R）
        eval_result = self.evaluation_agent.evaluate_retrieval(
            user_query=query,
            retrieved_docs=docs,
            reference=reference
        ) or {}

        def _get(ev: Dict[str, Any], *keys, default=None):
            for k in keys:
                if k in ev:
                    v = ev[k]
                    try:
                        return float(v[0] if isinstance(v, list) else v)
                    except Exception:
                        continue
            return default

        context_precision = _get(eval_result, "context_precision", "ctxP", default=0.0)
        context_recall    = _get(eval_result, "context_recall", "ctxR", default=None)  # 可能为 None
        weak_recall       = _get(eval_result, "weak_recall", default=None)             # 新增：弱召回

        # 友好打印：有真 recall 打真 recall；否则打印 weak_recall；都没有就显示 "n/a"
        if context_recall is not None:
            print(f"🔎 [Retrieval] k={self.top_k}  latency={latency_ms:.1f}ms  "
                  f"Precision={context_precision:.2f}, Recall={context_recall:.2f}")
        elif weak_recall is not None:
            print(f"🔎 [Retrieval] k={self.top_k}  latency={latency_ms:.1f}ms  "
                  f"Precision={context_precision:.2f}, WeakRecall={weak_recall:.2f}")
        else:
            print(f"🔎 [Retrieval] k={self.top_k}  latency={latency_ms:.1f}ms  "
                  f"Precision={context_precision:.2f}, Recall=n/a")

        return {
            "docs": docs,
            "context_precision": float(context_precision or 0.0),
            "context_recall": context_recall,      # 可能是 None
            "weak_recall": weak_recall,            # 可能是 None
            "latency_ms": latency_ms,
            "hits_meta": hits_meta
        }
