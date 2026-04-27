# RAG Pipeline — Onboarding Guide

> **Goal:** Get you running the RAG pipeline locally in under 30 minutes.
> For running the full application (frontend + backend), see [`fullstack-local-dev.md`](./fullstack-local-dev.md).
> For hosting and infrastructure, see [`rag-service-schema.md`](./rag-service-schema.md).

---

## What This Pipeline Does

Given a question, the pipeline retrieves relevant documents from a knowledge base and generates a grounded answer using chain-of-thought reasoning. It's designed for **multi-hop questions** that require connecting information across multiple documents.

```
User Question
    ↓
ReasoningAgent  — breaks question into sub-queries, runs up to 4 retrieval hops (IRCoT)
    ↓
RetrievalAgent  — hybrid BM25 + FAISS search, CrossEncoder reranking, top-5 docs
    ↓
GenerationAgent — CoT prompt + few-shot examples → answer
    ↓
EvaluationAgent — scores faithfulness, relevancy, Semantic F1 (RAGAS)
    ↓
RL Router       — (optional) decides if IRCoT result is good enough, or triggers PAR2 fallback
```

The whole flow is orchestrated by a **LangGraph state machine** (`agents/langgraph_rag.py`).

### Knowledge Base Scope

> **Important:** The current vector store is built exclusively from the **HotpotQA Wikipedia corpus** (~24,000 paragraph-level chunks). The pipeline works well for HotpotQA-style multi-hop questions but **will not reliably answer questions outside this corpus**. For example, questions about recent events, specialized domains, or topics not covered in HotpotQA Wikipedia articles will likely return poor or no answers.
>
> To use the pipeline on a different knowledge base, build a new FAISS index from your own documents (see Section 3).

---

## 1. Prerequisites

- Python 3.11
- An OpenAI API key (`OPENAI_API_KEY`)
- The FAISS vector index (see [Vector Store](#3-vector-store) below)

---

## 2. Setup

```bash
# Clone and enter the project
git clone https://github.com/aiea-lab/LLM-logic.git
cd webapp
git checkout agentic-rag

# Create virtual environment at the repo root
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r agent_integration/requirements.txt
```

Create a `.env` file in the **repo root** (`webapp/`, same level as `agent_integration/`):

```bash
OPENAI_API_KEY=sk-...

# Point to the FAISS index directory (see Vector Store section)
FAISS_PATH_OPENAI=vectorstore-hotpot/hotpotqa_faiss

# Generation model — use gpt-4o-mini for best results (matches training data for RL router)
# gpt-3.5-turbo is the code default but produces lower quality answers
GEN_LLM_MODEL=gpt-4o-mini
EMB_MODEL=text-embedding-3-large
RETR_TOP_K=5
GEN_FORCE_ANSWER=1
```

> **Note:** The pipeline loads `.env` from the repo root (two levels up from `agents/`). Do **not** place `.env` inside `agent_integration/`.

---

## 3. Vector Store

This is the knowledge base the pipeline retrieves from.

### What's in it

| Property | Value |
|----------|-------|
| Source | HotpotQA Wikipedia corpus |
| Documents | ~24,000 paragraph-level chunks |
| Embedding model | `text-embedding-3-large` (OpenAI) |
| Index format | FAISS (local file, loaded into memory) |
| Size on disk | ~500 MB |
| Location | `vectorstore-hotpot/hotpotqa_faiss/` |

> **The FAISS index is not in the repo** (too large for git). Build it yourself using the script below (~20–30 min, one-time setup).

### How retrieval works

```
Query
  ├── FAISS dense search    (semantic similarity, text-embedding-3-large)
  ├── BM25 sparse search    (keyword match, built at startup from FAISS docstore)
  └── RRF fusion            (Reciprocal Rank Fusion, combines both ranked lists)
       ↓
  CrossEncoder reranker     (ms-marco-MiniLM-L-6-v2, picks top-5)
       ↓
  Top-5 documents → GenerationAgent
```

### Build the index from scratch

```bash
cd agent_integration
python scripts/build_vectorstore.py \
    --input data-hotpot/hotpot_mini_corpus.json \
    --output vectorstore-hotpot/hotpotqa_faiss \
    --model text-embedding-3-large
```

This will take ~20–30 minutes and cost a few dollars in OpenAI embedding API calls.

### Swap in a different knowledge base

To use the pipeline on your own documents:
1. Prepare your documents as a JSON list: `[{"title": "...", "text": "..."}]`
2. Run `build_vectorstore.py` with your file
3. Point `FAISS_PATH_OPENAI` to the new index directory

---

## 4. Verify the Pipeline (terminal smoke test)

This runs one question end-to-end in the terminal — useful to confirm the pipeline is set up correctly before starting the full stack.

```bash
# From the repo root (webapp/)
source venv/bin/activate
cd agent_integration

python - <<'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

from agents.evaluate_dataset_real import get_vectorstore
from agents.langgraph_rag import run_rag_pipeline
from agents.retrieval_agent import RetrievalAgent
from agents.generation_agent import GenerationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reasoning_agent import ReasoningAgent
from agents.hybrid_retriever import HybridRetriever
from agents.reranker import create_cross_encoder_reranker

vectorstore     = get_vectorstore()
eval_agent      = EvaluationAgent()
gen_agent       = GenerationAgent()
reasoning_agent = ReasoningAgent()
retrieval_agent = RetrievalAgent(
    vectorstore, eval_agent,
    hybrid_retriever=HybridRetriever(vectorstore),
    reranker=create_cross_encoder_reranker(),
)

result = run_rag_pipeline(
    question="Who is the dance partner of Yulia Zagoruychenko?",
    retrieval_agent=retrieval_agent,
    reasoning_agent=reasoning_agent,
    generation_agent=gen_agent,
    evaluation_agent=eval_agent,
    use_router=False,
    visualize=False,
)
print("Answer:", result["answer"])
EOF
```

Expected output:
```
Answer: Riccardo Cocchi
```

> **For the full application (frontend UI), skip this step and go directly to [`fullstack-local-dev.md`](./fullstack-local-dev.md).**

### Sample questions to try

The file `agent_integration/data-hotpot/dev_real.jsonl` contains 30 HotpotQA multi-hop questions you can use for testing. Pick any question from it:

```bash
# Print 5 random questions with their answers
python3 -c "
import json, random
with open('agent_integration/data-hotpot/dev_real.jsonl') as f:
    questions = [json.loads(l) for l in f]
for q in random.sample(questions, 5):
    print('Q:', q['question'])
    print('A:', q['answer'])
    print()
"
```

---

## 5. Key Files

```
agent_integration/
├── agents/
│   ├── langgraph_rag.py        # Main pipeline orchestration (LangGraph state machine)
│   ├── retrieval_agent.py      # Hybrid retrieval + reranking
│   ├── generation_agent.py     # CoT answer generation
│   ├── evaluation_agent.py     # RAGAS scoring + Semantic F1
│   ├── reasoning_agent.py      # DSPy sub-question decomposition
│   ├── hybrid_retriever.py     # BM25 + FAISS + RRF fusion
│   ├── reranker.py             # CrossEncoder reranker (ms-marco-MiniLM-L-6-v2)
│   ├── esc.py                  # Evidence Sufficiency Controller (PAR2 stage 2)
│   ├── multi_query.py          # Multi-query expansion
│   ├── offline_rl_router.py    # V2 Oracle RL router (routing logic + inference)
│   └── offline_rl_router_policy_v2.pt  # Required: trained router weights
├── scripts/
│   └── build_vectorstore.py    # Build FAISS index from corpus
├── data-hotpot/
│   └── dev_real.jsonl          # 30-question development set
├── vectorstore-hotpot/         # FAISS index (not in repo, build with script above)
└── requirements.txt
```

---

## 6. The RL Router (background)

The pipeline has two retrieval modes:

| Mode | Description |
|------|-------------|
| **IRCoT** (default) | Iterative retrieval, up to 4 hops — handles ~70% of questions |
| **PAR2** | Anchor-based two-stage retrieval — wider coverage for harder questions |

The **V2 Oracle RL Router** (`agents/offline_rl_router.py`) automatically picks the mode per question based on retrieval quality signals. The pre-trained weights (`offline_rl_router_policy_v2.pt`) are required at startup — the router runs automatically when the full stack is running.

---

## 7. Running the Full Stack (Frontend + Backend)

To run the complete application — Next.js frontend, Flask backend, and RAG service together — see **[`fullstack-local-dev.md`](./fullstack-local-dev.md)**.

For Docker-based deployment and cloud hosting, see **[`rag-service-schema.md`](./rag-service-schema.md)**.

---

## 8. Common Issues

**`FAISS index not found`**
→ Check that `FAISS_PATH_OPENAI` points to a directory containing `index.faiss` and `index.pkl`.

**`OpenAI API key not set`**
→ Make sure `.env` is in the **repo root** (`webapp/`) and contains `OPENAI_API_KEY=sk-...`.

**`ModuleNotFoundError: rank_bm25`**
→ Run `pip install rank-bm25 sentence-transformers` (these are at the bottom of `requirements.txt`).

**Slow first run**
→ The CrossEncoder model (`ms-marco-MiniLM-L-6-v2`) downloads from HuggingFace on first use (~90MB). Subsequent runs use the cache.

**Pipeline answers questions outside HotpotQA poorly**
→ Expected behavior. The vector store only covers the HotpotQA Wikipedia corpus. To support a different domain, build a new FAISS index from your own documents (see Section 3).

---

*Questions? Reach out to Zhonghui (the RAG subteam lead) or open an issue on the `agentic-rag` branch.*
