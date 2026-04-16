# LLM Logic + Agentic RAG Integration

A multi-model LLM chat application with an integrated agentic RAG (Retrieval-Augmented Generation) pipeline, evaluated on [HotpotQA](https://hotpotqa.github.io/) multi-hop question answering. semF1 improved from **0.416 → 0.755 (+81.5%)** through systematic retrieval, generation, and routing optimization.

## Full-Stack Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend                            │
│              (Next.js - Port 3000)                       │
│   Model Selection: [OpenAI] [Claude] [Gemini] [RAG Agent]│
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Main Backend                           │
│               (Flask - Port 5000)                        │
│  /get_response                                           │
│    ├─ provider=openai  → OpenAI API                     │
│    ├─ provider=claude  → Anthropic API                  │
│    ├─ provider=gemini  → Google API                     │
│    └─ method=rag-agent → RAG Service                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Service                           │
│               (FastAPI - Port 8001)                      │
│  Agentic RAG Pipeline:                                   │
│  - ReasoningAgent (Query Optimization)                   │
│  - RetrievalAgent (FAISS Vector Search)                 │
│  - GenerationAgent (Answer Generation)                   │
│  - EvaluationAgent (Quality Assessment)                  │
│  - LangGraph Router (Multi-step Reasoning)              │
└─────────────────────────────────────────────────────────┘
```

## RAG Pipeline Architecture

```
                         User Query
                             │
                             ▼
                   ┌───────────────────┐
                   │  ReasoningAgent   │  Query optimization + Multi-query expansion
                   └────────┬──────────┘
                            │  3 query variants
                            ▼
                   ┌───────────────────┐
                   │  RetrievalAgent   │  Hybrid retrieval (BM25 + FAISS + RRF)
                   │                   │  → CrossEncoder reranking
                   └────────┬──────────┘
                            │  top-k documents
                            ▼
                   ┌───────────────────┐
                   │  IRCoT Loop       │  Iterative Retrieval Chain-of-Thought
                   │  (up to 4 hops)   │  reason → retrieve more → reason → ...
                   └────────┬──────────┘
                            │  accumulated context
                            ▼
                   ┌───────────────────┐
                   │  GenerationAgent  │  CoT Reasoning → Answer extraction
                   └────────┬──────────┘
                            │
                            ▼
                   ┌───────────────────┐
                   │  EvaluationAgent  │  Faithfulness / Relevancy / Semantic F1
                   └────────┬──────────┘
                            │  ctxP, ctxR, doc_count
                            ▼
                   ┌───────────────────┐
                   │  Retrieval Router │  BC MLP: IRCoT OK or anchor-based fallback?
                   └────────┬──────────┘
                    ircot_ok │           │ par2_needed
                             ▼           ▼
                        Generator   Anchor-Based Two-Stage Retrieval
                                    (5 sub-queries → ESC-gated refinement)
                                         │
                                         ▼
                                     Generator
                            │
                            ▼
                      Final Answer
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| ReasoningAgent | `reasoning_agent.py` | Sub-question decomposition, multi-query expansion, IRCoT iterative reasoning |
| RetrievalAgent | `retrieval_agent.py` | FAISS dense retrieval + BM25 sparse retrieval, RRF fusion |
| HybridRetriever | `hybrid_retriever.py` | BM25 + FAISS reciprocal rank fusion |
| CrossEncoder Reranker | `reranker.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking |
| Multi-Query | `multi_query.py` | LLM-based query variant generation + sub-query decomposition |
| GenerationAgent | `generation_agent.py` | CoT prompt + few-shot examples + answer parsing |
| EvaluationAgent | `evaluation_agent.py` | Ragas-based faithfulness, relevancy, noise sensitivity |
| ESC | `esc.py` | Evidence Sufficiency Controller for anchor-based two-stage retrieval |
| Offline RL Router | `offline_rl_router.py` | Reward-weighted imitation learning router (reward = semF1_adaptive − semF1_ircot) |
| LangGraph Orchestrator | `langgraph_rag.py` | State-machine orchestration of all agents |

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for full-stack deployment)
- OpenAI API Key (required for embeddings + generation)
- Anthropic API Key (optional, for Claude)
- Google API Key (optional, for Gemini)

### Run Evaluation

```bash
cd agent_integration

# Build vectorstore (one-time)
OPENAI_API_KEY_REAL=sk-... python scripts/build_vectorstore.py

# Run evaluation on HotpotQA dev set
FAISS_PATH_OPENAI=vectorstore-hotpot/hotpotqa_faiss_v3 \
python -m agents.evaluate_dataset_real \
  --dataset data-hotpot/dev_real.jsonl \
  --top_k 8 \
  --out_dir runs/trajectories_latest \
  --use_router 0
```

### Run Full-Stack Application

```bash
# Copy environment file and add your API keys
cp .env.example .env

# Build and start all services
docker-compose up --build
```

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **RAG Service**: http://localhost:8001

### Local Development

```bash
# Backend (Flask)
cd LLM-logic/backend
pip install -r requirements.txt
python app.py

# Frontend (Next.js)
cd LLM-logic/frontend
pnpm install
pnpm dev

# RAG Service (FastAPI)
cd rag_service
pip install -r requirements.txt
python main.py
```

## Available Models

### LLM Providers
- **OpenAI**: GPT-4o-mini, GPT-4o, GPT-3.5-turbo
- **Claude**: Claude 3 Opus, Sonnet, Haiku
- **Gemini**: Gemini 1.5 Pro, Flash

### Methods
- **RAG Agent**: Multi-step agentic RAG pipeline with reasoning and evaluation
- **Pro-SLM**: Prolog-based symbolic reasoning
- **RAG**: Simple retrieval-augmented generation
- **Chain of Thought**: Step-by-step reasoning
- **Standard**: Direct LLM query

## Project Structure

```
agent_rl/
├── agent_integration/              # Core RAG pipeline
│   ├── agents/
│   │   ├── reasoning_agent.py      # Query optimization + IRCoT loop
│   │   ├── retrieval_agent.py      # FAISS/BM25 retrieval + retry logic
│   │   ├── hybrid_retriever.py     # BM25 + FAISS + RRF fusion
│   │   ├── reranker.py             # CrossEncoder reranking
│   │   ├── multi_query.py          # LLM query expansion + sub-query decomposition
│   │   ├── esc.py                  # Evidence Sufficiency Controller (anchor-based Stage 2)
│   │   ├── generation_agent.py     # CoT generation + answer extraction
│   │   ├── evaluation_agent.py     # Ragas-based quality metrics
│   │   ├── langgraph_rag.py        # LangGraph state-machine orchestrator
│   │   ├── retrieval_router_bc.py  # BC retrieval router (IRCoT vs anchor-based fallback)
│   │   ├── offline_rl_router.py    # Offline RL router (reward-weighted imitation learning)
│   │   ├── RLRouterAgent.py        # Legacy generation-level router (BC + PPO)
│   │   └── ppo_router_trainer.py   # PPO training for generation-level router
│   ├── data-hotpot/                # HotpotQA evaluation dataset
│   ├── runs/                       # Experiment trajectories & stats
│   ├── scripts/                    # Vectorstore build scripts
│   ├── utils/                      # Text processing, trajectory logging
│   └── vectorstore-hotpot/         # FAISS indices
│
├── rag_service/                    # FastAPI RAG Service (production)
├── LLM-logic/
│   ├── backend/                    # Flask Backend (multi-provider LLM)
│   └── frontend/                   # Next.js Frontend
│
├── docker-compose.yml
└── .env.example
```

## API Endpoints

### Backend API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get_response` | POST | Send message and get LLM response |
| `/new_conversation` | POST | Create new conversation |
| `/conversation/<id>` | GET | Get conversation by ID |
| `/user` | POST | Create new user |
| `/login` | POST | User login |

### RAG Service API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query the RAG pipeline (synchronous) |
| `/query/stream` | POST | Stream response via Server-Sent Events |
| `/health` | GET | Health check (agents loaded, Redis connected) |
| `/metrics` | GET | Prometheus metrics |
| `/cache/stats` | GET | Redis cache hit/miss statistics |
| `/cache/clear` | DELETE | Clear all cached queries |

## Customizing the Vector Database

The RAG Service uses FAISS for vector search. To use a different dataset:

1. Build your FAISS index using OpenAI embeddings
2. Update `VECTORSTORE_PATH` in your environment
3. Restart the RAG Service

## License

MIT License
