"""Generate a synthetic RAG test set from the UCSC catalog with Ragas
TestsetGenerator (knowledge-graph + single/multi-hop query synthesizers).

This broadens the search_catalog dimension beyond our hand-written templates.
Output is RAW and MUST be human-verified before use (LLM-generated questions
can be ambiguous / unanswerable / wrong). Tool/schedule/calendar/out-of-scope
dimensions are NOT covered here — those stay hand-curated.

Usage (from agent_integration/, with OPENAI_API_KEY + EMB_MODEL set):
    python -m eval.gen_ragas_testset --n 10 --docs 60
"""
import json
import argparse

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator


def course_docs(path, limit):
    courses = json.load(open(path))[:limit]
    docs = []
    for c in courses:
        ge = f", General Education: {c['ge_code']}" if c.get("ge_code") else ""
        body = "\n".join(filter(None, [
            f"{c['course_code']} {c['title']}",
            f"{c['division'].replace('-', ' ')}, {c['credits']} credits{ge}",
            c["description"],
            f"Prerequisites: {c['requirements']}" if c.get("requirements") else "",
        ]))
        docs.append(Document(page_content=body,
                             metadata={"course_code": c["course_code"], "title": c["title"]}))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", default="data-ucsc/cse_courses.json")
    ap.add_argument("--out", default="eval/ragas_testset.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--docs", type=int, default=60)
    args = ap.parse_args()

    docs = course_docs(args.courses, args.docs)
    print(f"Generating {args.n} questions from {len(docs)} course docs...")

    gen_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    gen_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    generator = TestsetGenerator(llm=gen_llm, embedding_model=gen_emb)

    dataset = generator.generate_with_langchain_docs(docs, testset_size=args.n)
    df = dataset.to_pandas()

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "question": r.get("user_input", ""),
            "reference": r.get("reference", ""),
            "synthesizer": r.get("synthesizer_name", ""),
        })
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} items to {args.out}")
    by_syn = {}
    for r in rows:
        by_syn[r["synthesizer"]] = by_syn.get(r["synthesizer"], 0) + 1
    print("By synthesizer:", by_syn)
    print("\n=== SAMPLE (review these for quality) ===")
    for r in rows[:6]:
        print(f"\n[{r['synthesizer']}]")
        print("Q:", r["question"])
        print("ref:", (r["reference"] or "")[:200])


if __name__ == "__main__":
    main()
