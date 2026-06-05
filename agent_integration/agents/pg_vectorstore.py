"""pgvector-backed store exposing the minimal FAISS-compatible interface that
HybridRetriever needs: `similarity_search_with_score(query, k)` and `docstore._dict`.

This lets the existing hybrid (BM25 + dense + RRF) + CrossEncoder reranker stack run
unchanged on Postgres instead of a FAISS file baked into the image. Dense search is a
`embedding <=> query` (cosine) nearest-neighbour query; BM25 is still built in-app from
the corpus (loaded once from Postgres here, mirroring FAISS's docstore._dict).

Active when DATABASE_URL is set (see slug_retrieval._get_vectorstore). Connections are
per-query (low QPS); use the Neon "-pooler" connection string in serverless deploys.
"""
import os
from types import SimpleNamespace

import psycopg
from langchain_core.documents import Document

_COLS = "course_code,title,division,credits,ge_code,url,page_content"


def _vec_literal(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _row_to_doc(code, title, division, credits, ge_code, url, page_content):
    return Document(page_content=page_content, metadata={
        "type": "course", "course_code": code, "title": title, "division": division,
        "credits": credits, "ge_code": ge_code or "", "url": url or "",
    })


class PgVectorStore:
    """Minimal FAISS-shaped adapter over a pgvector `courses` table."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self._url = os.environ["DATABASE_URL"]
        # load the full corpus once for BM25 (mirrors FAISS docstore._dict)
        docs = {}
        with psycopg.connect(self._url) as conn, conn.cursor() as cur:
            cur.execute(f"select {_COLS} from courses")
            for i, row in enumerate(cur.fetchall()):
                docs[i] = _row_to_doc(*row)
        self.docstore = SimpleNamespace(_dict=docs)
        print(f"[PgVectorStore] loaded {len(docs)} courses from Postgres.")

    def similarity_search_with_score(self, query: str, k: int = 5):
        """Cosine nearest-neighbour over pgvector. Returns [(Document, distance)]
        (lower distance = better), matching FAISS's signature."""
        qvec = _vec_literal(self.embeddings.embed_query(query))
        with psycopg.connect(self._url) as conn, conn.cursor() as cur:
            cur.execute(
                f"select {_COLS}, embedding <=> %s::vector as distance "
                "from courses order by distance limit %s",
                (qvec, k),
            )
            rows = cur.fetchall()
        return [(_row_to_doc(*r[:7]), float(r[7])) for r in rows]
