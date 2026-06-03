"""Reference-free LLM-judged quality metrics (Ragas): faithfulness and
answer relevancy. These catch what deterministic substring matching can't —
e.g. a confident answer that ISN'T grounded in what the tools returned
(hallucination).

The judge model is PINNED (version + temperature=0) so scores don't drift
run-to-run from the judge side. LLM-judge metrics still vary somewhat; treat
them as directional and track against a baseline, not as exact numbers.
"""
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Pin the judge (override in CI to a dated snapshot for full reproducibility).
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-4o-mini")


def score_quality(items):
    """items: list of {question, answer, contexts: list[str]}.
    Returns list of {faithfulness, answer_relevancy} aligned to items.
    """
    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=os.environ.get("EMB_MODEL", "text-embedding-3-small")))

    samples = [
        SingleTurnSample(
            user_input=it["question"],
            response=it["answer"] or "",
            retrieved_contexts=it["contexts"] or ["(no tool context)"],
        )
        for it in items
    ]
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=emb)],
        show_progress=False,
    )
    df = result.to_pandas()

    faith_col = next((c for c in df.columns if "faith" in c.lower()), None)
    relev_col = next((c for c in df.columns if "relevan" in c.lower()), None)

    out = []
    for _, r in df.iterrows():
        def num(col):
            v = r.get(col) if col else None
            try:
                return round(float(v), 3)
            except (TypeError, ValueError):
                return None
        out.append({"faithfulness": num(faith_col), "answer_relevancy": num(relev_col)})
    return out


if __name__ == "__main__":
    # Quick validation: hallucinated answers should score LOW faithfulness,
    # grounded answers HIGH.
    demo = [
        {"question": "Can I take CSE 3 pass/no-pass?",
         "answer": "Yes, you can take CSE 3 on a pass/no-pass basis.",
         "contexts": ["CSE 3 Computing Technology in a Changing Society. 5 credits. "
                      "Offered Fall 2025, TuTh 10:40-11:45."]},  # context has NO P/NP info
        {"question": "How many credits is CSE 3?",
         "answer": "CSE 3 is a 5-credit course.",
         "contexts": ["CSE 3 Computing Technology in a Changing Society. 5 credits."]},
    ]
    for it, sc in zip(demo, score_quality(demo)):
        print(f"faith={sc['faithfulness']} relev={sc['answer_relevancy']}  | {it['answer'][:50]}")
