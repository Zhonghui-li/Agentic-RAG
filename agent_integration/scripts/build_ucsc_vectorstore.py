"""Build a FAISS vectorstore from the scraped UCSC CSE catalog.

Each course becomes one retrievable document (courses are short, so no chunking).
page_content is formatted for retrieval (code + title + description + prereqs);
metadata carries the structured fields used for citations and tool hand-off.

Embedding model defaults to text-embedding-3-small (1536-dim). The RAG service
must use the SAME EMB_MODEL or FAISS will raise a dimension mismatch.
"""
import os
import json
import argparse

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()


def course_to_document(c: dict) -> Document:
    div = c["division"].replace("-", " ")
    ge = f", General Education: {c['ge_code']}" if c.get("ge_code") else ""
    parts = [
        f"{c['course_code']} {c['title']}",
        f"{div}, {c['credits']} credits{ge}",
        c["description"],
    ]
    if c.get("requirements"):
        parts.append(f"Prerequisites: {c['requirements']}")
    page_content = "\n".join(p for p in parts if p)

    return Document(
        page_content=page_content,
        metadata={
            "type": "course",
            "course_code": c["course_code"],
            "title": f"{c['course_code']} {c['title']}",
            "division": c["division"],
            "credits": c["credits"],
            "ge_code": c.get("ge_code", ""),
            "url": c.get("url", ""),
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", default="../data-ucsc/cse_courses.json")
    ap.add_argument("--out", default="../vectorstore-ucsc/ucsc_cse_faiss")
    args = ap.parse_args()

    with open(args.courses) as f:
        courses = json.load(f)
    docs = [course_to_document(c) for c in courses]
    print(f"Prepared {len(docs)} course documents.")
    print("Sample doc:\n", docs[0].page_content[:200], "\n  meta:", docs[0].metadata)

    emb_model = os.getenv("EMB_MODEL", "text-embedding-3-small")
    print(f"Embedding with {emb_model} ...")
    embeddings = OpenAIEmbeddings(model=emb_model)
    vs = FAISS.from_documents(docs, embeddings)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    vs.save_local(args.out)
    print(f"Saved FAISS index ({len(docs)} vectors) to {args.out}")


if __name__ == "__main__":
    main()
