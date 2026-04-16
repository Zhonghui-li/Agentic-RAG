# agents/hybrid_retriever.py
"""Hybrid retriever: BM25 + FAISS with Reciprocal Rank Fusion (RRF)."""

import re
from typing import List, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser, lowercased."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """
    Combines dense (FAISS) and sparse (BM25) retrieval using RRF fusion.

    Usage::

        hr = HybridRetriever(vectorstore)
        docs = hr.retrieve("some query", k=5)
    """

    def __init__(self, vectorstore, rrf_k: int = 60):
        """
        Args:
            vectorstore: A LangChain FAISS vectorstore (already loaded).
            rrf_k: RRF constant (default 60, per the original RRF paper).
        """
        self.vectorstore = vectorstore
        self.rrf_k = rrf_k

        # Build BM25 index from all documents in the FAISS docstore.
        self._docs: List[Document] = []
        self._bm25 = None
        self._build_bm25_index()

    # ------------------------------------------------------------------
    def _build_bm25_index(self):
        from rank_bm25 import BM25Okapi

        docstore = self.vectorstore.docstore
        # FAISS docstore stores docs in _dict (id -> Document)
        id_map = getattr(docstore, "_dict", None)
        if id_map is None:
            raise RuntimeError(
                "Cannot extract documents from vectorstore.docstore._dict; "
                "is this a LangChain FAISS vectorstore?"
            )

        self._docs = list(id_map.values())
        corpus_tokens = [_tokenize(d.page_content) for d in self._docs]
        self._bm25 = BM25Okapi(corpus_tokens)
        print(f"[HybridRetriever] BM25 index built over {len(self._docs)} documents.")

    # ------------------------------------------------------------------
    def _bm25_search(self, query: str, k: int) -> List[tuple]:
        """Return list of (Document, bm25_score) sorted descending."""
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        # argsort descending, take top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Run both FAISS and BM25, fuse with RRF, return top-k Documents.
        Fusion score is stored in ``metadata["score"]``.
        """
        # --- Dense retrieval (FAISS) ---
        faiss_results = self.vectorstore.similarity_search_with_score(query, k=k)
        # faiss_results: list of (Document, distance) — lower distance = better

        # --- Sparse retrieval (BM25) ---
        bm25_results = self._bm25_search(query, k=k)

        # --- RRF fusion ---
        # Build rank maps  (doc_content_hash -> rank, starting from 1)
        def _doc_key(doc: Document) -> str:
            return doc.page_content[:256]

        rrf_scores: dict = {}  # key -> cumulative RRF score
        doc_map: dict = {}     # key -> Document

        # FAISS ranks (sorted by ascending distance = best first)
        for rank, (doc, _dist) in enumerate(faiss_results, start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[key] = doc

        # BM25 ranks (already sorted descending by score)
        for rank, (doc, _score) in enumerate(bm25_results, start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[key] = doc

        # Sort by RRF score descending, take top-k
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        results: List[Document] = []
        for key in sorted_keys:
            doc = doc_map[key]
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["score"] = rrf_scores[key]
            results.append(Document(page_content=doc.page_content, metadata=meta))

        return results
