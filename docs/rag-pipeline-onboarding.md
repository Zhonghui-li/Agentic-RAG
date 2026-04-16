# RAG Pipeline — Onboarding Guide

> **Goal:** Get you running the RAG pipeline locally in under 30 minutes.
> For the full API contract and infrastructure requirements, see [`rag-service-schema.md`](./rag-service-schema.md).

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
cd LLM-logic
git checkout agentic-rag

# Create virtual environment at the repo root
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r agent_integration/requirements.txt
```

Create a `.env` file in the **repo root** (`LLM-logic/`, same level as `agent_integration/`):

```bash
OPENAI_API_KEY=sk-...

# Point to the FAISS index directory (see Vector Store section)
FAISS_PATH_OPENAI=vectorstore-hotpot/hotpotqa_faiss

# Optional overrides (defaults shown)
GEN_LLM_MODEL=gpt-3.5-turbo
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

If you have a different corpus or need to rebuild:

```bash
cd agent_integration
python scripts/build_vectorstore.py \
    --input data-hotpot/hotpot_mini_corpus.json \
    --output vectorstore-hotpot/hotpotqa_faiss \
    --model text-embedding-3-large
```

This will take ~20–30 minutes and cost a few dollars in OpenAI embedding API calls.

### Swap in a different vector store

To use a different knowledge base:
1. Prepare your documents as a JSON list: `[{"title": "...", "text": "..."}]`
2. Run `build_vectorstore.py` with your file
3. Point `FAISS_PATH_OPENAI` to the new index directory

---

## 4. Run a Sample Query

The quickest way to test the pipeline end-to-end:

```bash
# From the repo root (LLM-logic/)
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
    question="What government position did the publisher of Jane's Fighting Ships hold?",
    retrieval_agent=retrieval_agent,
    reasoning_agent=reasoning_agent,
    generation_agent=gen_agent,
    evaluation_agent=eval_agent,
    use_router=False,
    visualize=False,
)
print("Answer:", result["answer"])
print("Semantic F1:", result.get("semantic_f1_score"))
EOF
```

Expected output (roughly):
```
Answer: Fred T. Jane served as a naval officer...
Semantic F1: 0.82
```

---

## 5. Run the Evaluation Script (500 questions)

```bash
# From the repo root (LLM-logic/)
source venv/bin/activate
cd agent_integration
python main-hotpot.py
```

Key env vars for the eval script:

| Variable | Default | Effect |
|----------|---------|--------|
| `TESTSET_SIZE` | `5` | How many questions to run |
| `USE_ROUTER` | `0` | Enable RL routing (set to `1`) |
| `EVAL_MODE` | `hybrid` | Scoring mode: `strict` / `lenient` / `hybrid` |
| `GEN_FORCE_ANSWER` | `1` | Force model to answer even with thin context |

---

## 6. Key Files

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
│   ├── offline_rl_router.py    # V2 Oracle RL router (train + inference)
│   └── offline_rl_router_policy_v2.pt  # Pre-trained V2 router weights
├── scripts/
│   └── build_vectorstore.py    # Build FAISS index from corpus
├── data-hotpot/
│   ├── dev_500.jsonl           # 500-question evaluation set
│   └── dev_real.jsonl          # 30-question development set
├── vectorstore-hotpot/         # FAISS index (not in repo, build with script above)
├── main-hotpot.py              # Batch evaluation entry point
└── requirements.txt
```

---

## 7. The RL Router (optional, advanced)

The pipeline has two retrieval modes:

| Mode | Description | When to use |
|------|-------------|-------------|
| **IRCoT** (default) | Iterative retrieval, up to 4 hops | Most questions (~70%) |
| **PAR2** | Anchor-based two-stage retrieval, wider coverage | Hard questions where IRCoT fails |

The **V2 Oracle RL Router** (`agents/offline_rl_router.py`) automatically decides which mode to use, based on IRCoT retrieval quality signals (context precision, recall, doc count). It's a small MLP trained on 500 paired experiments.

To enable:
```bash
USE_ROUTER=1 python main-hotpot.py
```

---

## 8. Running the Full Stack Locally

This runs the complete application: Next.js frontend → Flask backend → RAG FastAPI service.

### Prerequisites

- Redis running locally (`brew install redis && brew services start redis`)
- MongoDB running locally (`brew install mongodb-community && brew services start mongodb-community`)
- Node.js + pnpm (`npm install -g pnpm`)

### Start all three services

Open three separate terminal tabs from the **repo root** (`LLM-logic/`):

**Terminal 1 — RAG service (port 8001)**
```bash
source venv/bin/activate
uvicorn rag_service.main:app --port 8001 --host 0.0.0.0
```
Startup takes ~20 seconds (BM25 index build + model load). Wait for `RAG Service initialized successfully!` before continuing.

**Terminal 2 — Flask backend (port 5001)**
```bash
source venv/bin/activate
cd LLM-logic/backend
python app.py
```

**Terminal 3 — Frontend (port 3000)**
```bash
cd LLM-logic/frontend
pnpm run dev
```

Then open `http://localhost:3000` in your browser. Select **RAG Agent** as the provider and type a question.

> **Note:** The frontend calls the Flask backend at `http://127.0.0.1:5001`, which forwards to the RAG service at `http://localhost:8001`. All URLs are hardcoded for local development.

### Verify via curl (no frontend needed)

```bash
# 1. Check RAG service health
curl http://localhost:8001/health

# 2. Query directly
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which country does the River Thames flow through?", "use_router": false}'
```

### Hosting

For Docker-based deployment and cloud hosting, see [`rag-service-schema.md`](./rag-service-schema.md). It covers Docker Compose setup, all environment variables, infrastructure requirements, and open questions for the hosting team.

---

## 9. Common Issues

**`FAISS index not found`**
→ Check that `FAISS_PATH_OPENAI` points to a directory containing `index.faiss` and `index.pkl`.

**`OpenAI API key not set`**
→ Make sure `.env` is in the **repo root** (`LLM-logic/`) and contains `OPENAI_API_KEY=sk-...`.

**`ModuleNotFoundError: rank_bm25`**
→ Run `pip install rank-bm25 sentence-transformers` (these are at the bottom of `requirements.txt`).

**Slow first run**
→ The CrossEncoder model (`ms-marco-MiniLM-L-6-v2`) downloads from HuggingFace on first use (~90MB). Subsequent runs use the cache.

---

## 10. API Reference

The pipeline also runs as a FastAPI service. See [`rag-service-schema.md`](./rag-service-schema.md) for:
- Endpoint definitions (`POST /query`, `POST /query/stream`, `GET /health`)
- All environment variables and their defaults
- Infrastructure requirements (CPU, memory, Redis)
- How to run with Docker

---

*Questions? Reach out to Zhonghui (the RAG subteam lead) or open an issue on the `agentic-rag` branch.*
