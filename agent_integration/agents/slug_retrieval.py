"""search_catalog: the retrieval tool for the Slug Advisor agent.

This wraps the project's retrieval stack (hybrid BM25+FAISS retrieval with RRF
fusion over the UCSC catalog + CrossEncoder reranker) and returns the top course
passages as text — it does NOT generate a final answer (the agent does that).

Hybrid (sparse BM25 + dense FAISS) matters here: course questions hinge on exact
entities/codes like "CSE 101", which BM25 nails and dense vectors can blur.

Multi-hop questions (e.g. prerequisite chains) are handled by the *agent*
calling this tool repeatedly, not by IRCoT inside the tool. IRCoT remains
available in langgraph_rag.py if single-shot retrieval proves insufficient.

Reranker attaches `rerank_score` to each passage — a reference-free retrieval
quality signal (the production-safe replacement for the gold-dependent ctxP/ctxR
discussed in the roadmap; usable later for adaptive routing / web_search fallback).
"""
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

_VS_PATH = os.environ.get(
    "UCSC_VECTORSTORE",
    os.path.join(os.path.dirname(__file__), "..", "vectorstore-ucsc", "ucsc_cse_faiss"),
)
_EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
_RERANK = os.environ.get("RERANK", "1") == "1"

_vectorstore = None
_hybrid = None
_reranker = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        emb = OpenAIEmbeddings(model=_EMB_MODEL)
        if os.environ.get("DATABASE_URL"):
            # pgvector (Postgres) — catalog decoupled from the image
            from agents.pg_vectorstore import PgVectorStore
            _vectorstore = PgVectorStore(emb)
        else:
            # FAISS file (local dev / backward compatible)
            _vectorstore = FAISS.load_local(
                _VS_PATH, emb, allow_dangerous_deserialization=True
            )
    return _vectorstore


def _get_hybrid():
    """Lazy-build the hybrid (BM25+FAISS+RRF) retriever over the catalog."""
    global _hybrid
    if _hybrid is None:
        from agents.hybrid_retriever import HybridRetriever
        _hybrid = HybridRetriever(_get_vectorstore())
    return _hybrid


def _get_reranker():
    """Lazy-load the CrossEncoder reranker; degrade gracefully if unavailable."""
    global _reranker
    if _reranker is None and _RERANK:
        try:
            from agents.reranker import create_cross_encoder_reranker
            _reranker = create_cross_encoder_reranker()
        except Exception as e:
            print(f"[search_catalog] reranker disabled ({type(e).__name__}: {e})")
            _reranker = False  # sentinel: tried and failed
    return _reranker or None


def search_catalog(query: str, k: int = 4) -> str:
    """Search the UCSC CSE course catalog for course descriptions, prerequisites,
    credits, and requirements. Returns the most relevant course entries. Call it
    again with a different query to follow prerequisite chains (multi-hop)."""
    # hybrid (BM25 + FAISS + RRF) over-retrieve, then rerank down to k
    docs = _get_hybrid().retrieve(query, k=max(k * 3, k))

    reranker = _get_reranker()
    if reranker and docs:
        docs = reranker(query, docs)[:k]
    else:
        docs = docs[:k]

    if not docs:
        return f"No catalog entries found for '{query}'."

    out: List[str] = []
    for d in docs:
        score = d.metadata.get("rerank_score")
        tag = f" (relevance {score:.2f})" if isinstance(score, float) else ""
        url = d.metadata.get("url", "")
        src = f"\n[catalog: {url}]" if url else ""
        out.append(f"[{d.metadata.get('course_code', '?')}]{tag}\n{d.page_content}{src}")
    return "\n\n".join(out)
