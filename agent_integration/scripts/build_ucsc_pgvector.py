"""Build the pgvector-backed catalog store in Postgres (the FAISS -> pgvector migration).

Same page_content / metadata as the FAISS build (imports `course_to_document`), so
retrieval quality is comparable. Decouples the catalog from the image: update the
data in Postgres without rebuilding/redeploying the container.

Run once to (re)load the catalog (idempotent — drops and recreates the table):

    DATABASE_URL=postgresql://... OPENAI_API_KEY=... EMB_MODEL=text-embedding-3-small \
        python scripts/build_ucsc_pgvector.py
"""
import os
import json
import argparse

import psycopg
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from build_ucsc_vectorstore import course_to_document  # identical formatting

load_dotenv()

DDL = """
create extension if not exists vector;
drop table if exists courses;
create table courses (
    id serial primary key,
    course_code text,
    title text,
    division text,
    credits int,
    ge_code text,
    url text,
    page_content text,
    embedding vector(1536)
);
"""


def _vec_literal(v):
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", default=os.path.join(
        os.path.dirname(__file__), "..", "data-ucsc", "cse_courses.json"))
    args = ap.parse_args()

    with open(args.courses) as f:
        courses = json.load(f)
    docs = [course_to_document(c) for c in courses]
    print(f"Prepared {len(docs)} course documents.")

    emb_model = os.getenv("EMB_MODEL", "text-embedding-3-small")
    print(f"Embedding {len(docs)} courses with {emb_model} ...")
    vectors = OpenAIEmbeddings(model=emb_model).embed_documents(
        [d.page_content for d in docs])

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            for d, v in zip(docs, vectors):
                m = d.metadata
                cur.execute(
                    "insert into courses (course_code,title,division,credits,"
                    "ge_code,url,page_content,embedding) "
                    "values (%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                    (m.get("course_code"), m.get("title"), m.get("division"),
                     m.get("credits"), m.get("ge_code"), m.get("url"),
                     d.page_content, _vec_literal(v)),
                )
            # HNSW index for cosine distance (production touch; instant at this size)
            cur.execute(
                "create index on courses using hnsw (embedding vector_cosine_ops);")
            conn.commit()
            cur.execute("select count(*) from courses;")
            print(f"Inserted {cur.fetchone()[0]} courses into pgvector (Postgres).")


if __name__ == "__main__":
    main()
