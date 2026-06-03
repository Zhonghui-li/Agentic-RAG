# Agentic RAG

A full-stack chat application with an integrated agentic RAG (Retrieval-Augmented Generation) pipeline, evaluated on [HotpotQA](https://hotpotqa.github.io/) multi-hop question answering. semF1 improved from **0.416 → 0.755 (+81.5%)** through systematic retrieval, generation, and routing optimization.

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
│               (Flask - Port 5001)                        │
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
│  - RetrievalAgent (FAISS + BM25 hybrid retrieval)       │
│  - GenerationAgent (CoT Answer Generation)               │
│  - EvaluationAgent (Quality Assessment)                  │
│  - V2 Oracle RL Router (IRCoT vs PAR2 routing)          │
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
                   │  V2 Oracle RL     │  MLP trained on 500-question counterfactual
                   │  Retrieval Router │  experiments: IRCoT OK or PAR2 fallback?
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
| ReasoningAgent | `reasoning_agent.py` | Query optimization, pronoun resolution via conversation context, multi-query expansion |
| RetrievalAgent | `retrieval_agent.py` | FAISS dense retrieval + BM25 sparse retrieval, RRF fusion |
| HybridRetriever | `hybrid_retriever.py` | BM25 + FAISS reciprocal rank fusion |
| CrossEncoder Reranker | `reranker.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking |
| Multi-Query | `multi_query.py` | LLM-based query variant generation + sub-query decomposition |
| GenerationAgent | `generation_agent.py` | CoT prompt + few-shot examples + answer parsing |
| EvaluationAgent | `evaluation_agent.py` | Ragas-based faithfulness, relevancy, noise sensitivity |
| ESC | `esc.py` | Evidence Sufficiency Controller for anchor-based two-stage retrieval |
| Offline RL Router | `offline_rl_router.py` | V2 Oracle RL router trained on counterfactual experiments |
| LangGraph Orchestrator | `langgraph_rag.py` | State-machine orchestration of all agents |
| Semantic Cache | `rag_service/main.py` | Redis embedding-based cache; bypassed for pronoun/context-dependent queries |
| Guardrails | `rag_service/main.py` | Out-of-scope question filtering before pipeline execution |

## Quick Start

### Prerequisites

- Python 3.11
- OpenAI API key (required for embeddings + generation)
- FAISS vector index (see [RAG Pipeline Onboarding](./docs/rag-pipeline-onboarding.md))

### Run the Full Application Locally

See **[`docs/fullstack-local-dev.md`](./docs/fullstack-local-dev.md)** for step-by-step instructions to start all three services (RAG service, Flask backend, Next.js frontend).

### RAG Pipeline Only (no frontend)

See **[`docs/rag-pipeline-onboarding.md`](./docs/rag-pipeline-onboarding.md)** for pipeline setup, vector store build, and terminal smoke test.

### Docker (all services)

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
docker-compose up
```

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **RAG Service**: http://localhost:8001

### Local Development

```bash
# Backend (Flask)
cd webapp/backend
pip install -r requirements.txt
python app.py

# Frontend (Next.js)
cd webapp/frontend
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

### RAG Agent
Multi-step agentic pipeline with:
- **Hybrid retrieval**: BM25 + FAISS + RRF fusion, CrossEncoder reranking
- **IRCoT reasoning**: iterative retrieval up to 4 hops
- **V2 Oracle RL routing**: MLP router switching between IRCoT and PAR2 fallback
- **Short-term memory**: conversation context passed across turns for pronoun resolution
- **Semantic cache**: Redis embedding-based cache for repeated queries
- **Guardrails**: out-of-scope filtering before pipeline runs
- **Streaming**: SSE token-by-token response with source citations

**Works best on HotpotQA-style multi-hop questions** — the vector store is built from the HotpotQA Wikipedia corpus.

## Project Structure

```
agent_rl/
├── agent_integration/              # Core RAG pipeline
│   ├── agents/
│   │   ├── langgraph_rag.py        # LangGraph state-machine orchestrator
│   │   ├── reasoning_agent.py      # Query optimization + IRCoT loop
│   │   ├── retrieval_agent.py      # FAISS/BM25 retrieval
│   │   ├── hybrid_retriever.py     # BM25 + FAISS + RRF fusion
│   │   ├── reranker.py             # CrossEncoder reranking
│   │   ├── multi_query.py          # LLM query expansion
│   │   ├── esc.py                  # Evidence Sufficiency Controller (PAR2 Stage 2)
│   │   ├── generation_agent.py     # CoT generation + answer extraction
│   │   ├── evaluation_agent.py     # Ragas-based quality metrics
│   │   ├── offline_rl_router.py    # V2 Oracle RL router
│   │   └── offline_rl_router_policy_v2.pt  # Pre-trained router weights
│   ├── data-hotpot/                # HotpotQA evaluation dataset
│   ├── scripts/                    # Vectorstore build scripts
│   ├── utils/                      # Text processing, trajectory logging
│   └── vectorstore-hotpot/         # FAISS indices (not in repo)
│
├── rag_service/                    # FastAPI RAG Service (production)
├── webapp/
│   ├── backend/                    # Flask Backend (multi-provider LLM)
│   └── frontend/                   # Next.js Frontend
│
├── docs/
│   ├── rag-pipeline-onboarding.md  # RAG pipeline setup guide
│   ├── fullstack-local-dev.md      # Full-stack local development guide
│   └── rag-service-schema.md       # API contract + hosting requirements
├── docker-compose.yml
└── .env.example
```

## API Endpoints

### RAG Service (`localhost:8001`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query the RAG pipeline (synchronous) |
| `/query/stream` | POST | Stream response via Server-Sent Events |
| `/health` | GET | Health check (agents loaded, Redis connected) |
| `/metrics` | GET | Prometheus metrics |

For full request/response schema and all environment variables, see [`docs/rag-service-schema.md`](./docs/rag-service-schema.md).

### Backend API (`localhost:5001`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get_response` | POST | Send message, get LLM response |
| `/new_conversation` | POST | Create new conversation |
| `/conversation/<id>` | GET | Get conversation by ID |
| `/login` | POST | User login |

## License

MIT License
