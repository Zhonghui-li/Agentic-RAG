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
| BC Retrieval Router | `retrieval_router_bc.py` | MLP classifier: routes between IRCoT and anchor-based fallback |
| Offline RL Router | `offline_rl_router.py` | Reward-weighted imitation learning router (reward = semF1_adaptive − semF1_ircot) |
| LangGraph Orchestrator | `langgraph_rag.py` | State-machine orchestration of all agents |

## Optimization Journey

We iteratively optimized the pipeline across **retrieval**, **chunking**, and **generation** stages. All experiments evaluated on 30 HotpotQA multi-hop questions.

### Performance Progression

| Version | Key Change | semF1 | semF1 >=0.8 | ctxR | ctxR >=0.8 | faith |
|---------|-----------|-------|-------------|------|-----------|-------|
| Baseline | FAISS-only, direct generation | 0.416 | 33.3% | 0.697 | 63.3% | 0.639 |
| + Hybrid Retrieval | BM25+FAISS+RRF, CrossEncoder, multi-query | 0.470 | 40.0% | 0.667 | 60.0% | 0.614 |
| + Paragraph Chunking | Document-aware splitting for HotpotQA | 0.561 | 53.3% | 0.667 | 60.0% | 0.503 |
| + Concise Generation | Shortest-answer prompt + extraction LLM call | 0.493 | 46.7% | 0.700 | 63.3% | 0.497 |
| + Embedding v3 | `text-embedding-3-small` upgrade | 0.595 | 56.7% | 0.717 | 66.7% | 0.547 |
| + IRCoT | Iterative multi-hop retrieval (up to 4 hops) | 0.586 | 60.0% | 0.783 | 76.7% | 0.589 |
| + CoT Prompt | Reasoning/Answer format + answer parsing | 0.672 | 70.0% | 0.750 | 73.3% | 0.519 |
| **+ Few-Shot Examples** | **2 in-context examples for multi-hop reasoning** | **0.705** | **73.3%** | **0.750** | **73.3%** | **0.592** |

### GPT-4o-mini + Retrieval Router Stages (30-question eval set)

| Stage | Key Change | semF1 | semF1 ≥0.8 |
|-------|-----------|-------|------------|
| GPT-4o-mini baseline | Switch from GPT-3.5 | 0.571 | 56.7% |
| + force_answer | Remove "insufficient context" fallback | 0.640 | — |
| + few-shot v2 | Redesigned examples for GPT-4o-mini behavior | 0.702 | 73.3% |
| + Adaptive retrieval router | Hard rule: ctxP<0.2 → anchor-based fallback | 0.727 | — |
| **+ BC retrieval router** | **MLP replaces hard rule; learns non-linear boundary** | **0.755** | **76.7%** |
| + Offline RL router* | Reward-weighted IL on 500-question trajectories | 0.824* | — |

*Offline RL result is simulation-based (held-out test set, not a full re-run).

### Cumulative Improvement

```
semF1:       0.416  →  0.755   (+81.5%)
semF1 >=0.8: 33.3%  →  76.7%  (+43.4pp)

500-question evaluation:
  IRCoT baseline:    semF1 = 0.669
  Adaptive BC router: semF1 = 0.710
```

### What Each Optimization Did

**1. Hybrid Retrieval** (`a86fa1b`)
- Combined BM25 sparse + FAISS dense retrieval with Reciprocal Rank Fusion
- Added CrossEncoder (`ms-marco-MiniLM-L-6-v2`) for reranking
- Multi-query expansion: LLM generates 3 query variants to improve recall

**2. Paragraph-based Chunking** (`4a26ec5`)
- Replaced fixed-size token chunking with document-aware paragraph splitting
- Preserves natural document boundaries in HotpotQA's paragraph structure
- Better context coherence for multi-hop reasoning

**3. Concise Generation** (`950ecd5`)
- Prompt engineering for shortest possible answers (name, date, number, place)
- Added `_extract_concise_answer()`: secondary LLM call to compress verbose answers
- Improved semantic F1 by reducing noise in predictions

**4. IRCoT - Iterative Retrieval Chain-of-Thought** (`553b7cc`)
- Replaced one-shot sub-question decomposition with iterative retrieve-reason loop
- Up to 4 hops: generate reasoning → identify knowledge gaps → retrieve more → continue
- Major retrieval quality jump: ctxR >=0.8 from 60% to 76.7%

**5. CoT Prompt + Answer Parsing** (`81cca43`)
- Structured `Reasoning: ... / Answer: ...` prompt format
- Forces model to explicitly connect facts across documents before answering
- `_parse_cot_answer()` extracts the Answer line; `_extract_concise_answer()` as fallback
- Lowered early-stop relevancy threshold (0.6→0.4) to avoid false-positive retries on short answers
- semF1 jump: 0.586 → 0.672 (+14.7%)

**6. Few-Shot Examples** (current)
- Added 2 in-context examples to the generation prompt demonstrating multi-hop reasoning
- Example 1: entity linking chain (A→B→attribute) — teaches bridging across documents
- Example 2: comparison reasoning — teaches extracting and comparing facts
- Carefully budgeted at ~150 tokens each to fit within the 2048-token prompt limit
- semF1: 0.672 → 0.705 (+4.9%), semF1≥0.8: 70.0% → 73.3%

---

## RL Router Experiments

After reaching semF1=0.705 via pipeline optimization, we added a learned routing layer to dynamically decide whether to accept or regenerate each answer. The router is trained in two stages: **Behavior Cloning (BC)** followed by **PPO reinforcement learning**.

### Router Design

- **State**: 6 RAGAS metrics (faithfulness, response relevancy, noise sensitivity, context precision, context recall, semantic F1)
- **Action space**: `end` (accept answer) / `regenerate` (retry with failure diagnosis)
- **Regenerate mechanism**: Diagnoses failure mode from RAGAS metrics (hallucination / off-topic / noise-misled), injects the diagnosis and previous wrong answer into the prompt as explicit correction signal
- **Requery removed**: IRCoT already handles iterative retrieval internally; router-level requery was found to hurt context recall

### BC Router (SFT Stage)

Teacher rule labels router decisions on collected trajectories; a small MLP imitates these decisions.

| Configuration | semF1 | Notes |
|---------------|-------|-------|
| No router (0.705 pipeline) | 0.705 | Baseline |
| + Teacher Rule v1 | 0.700 | rel threshold (0.40) miscalibrated — almost all questions trigger requery |
| + Teacher Rule v2 (recalibrated) | 0.643 | Lowered threshold (0.22); still over-intervenes |
| + BC Router v2 (retrained on new trajectories) | 0.615 | BC inherits teacher's systematic bias |
| + BC Router v3 (2-action, no requery) | 0.561 | Policy over-regenerates (89/90×) |

**Root cause**: Teacher rule thresholds calibrated on the weak pipeline (0.416 era) fire too aggressively on the optimized 0.705 pipeline. BC faithfully inherits this bias. BC's ceiling = teacher's judgment quality.

### PPO Router (RL Stage)

PPO learns directly from real semF1 rewards, bypassing teacher label bias. BC policy serves as warm-start to avoid destructive random exploration.

```bash
PYTHONPATH=agent_integration LIGHT_MODE=1 \
FAISS_PATH_OPENAI=vectorstore-hotpot/hotpotqa_faiss_v3 \
EMB_MODEL=text-embedding-3-large \
python3 agent_integration/agents/ppo_router_trainer.py \
  --dataset agent_integration/data-hotpot/dev_real.jsonl \
  --init_policy agent_integration/agents/router_policy_v3.pt \
  --out_dir agent_integration/runs/ppo_router \
  --n_iter 20 --max_regen 2
```

| Configuration | semF1 | vs BC |
|---------------|-------|-------|
| BC Router v3 | 0.561 | — |
| **PPO Router** | **0.623** | **+6.2%** |
| No router | 0.705 | — |

**PPO corrects BC's over-regeneration** (+6.2% vs BC) but remains below the no-router baseline. Root causes: ~25% of questions fail due to retrieval (wrong documents → regenerate can't help); sparse reward signal (approx_kl ≈ 0, policy barely updated); 30-question training set limits convergence.

**Key finding**: The true bottleneck is retrieval quality (~25% hard retrieval failures), not routing policy. Future directions: smarter requery with failure-conditioned query reformulation; larger training set; better teacher rule calibration for cleaner BC initialization.

---

## PAR2-RAG: Two-Stage Anchoring + Evidence Sufficiency Control

Inspired by the PAR2-RAG paper (Mar 2026), we implemented a two-stage multi-hop retrieval strategy on top of our existing FAISS + BM25 + CrossEncoder stack, replacing IRCoT's single-query iterative loop.

### Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 1: Coverage Anchoring         │
│  Decompose → 5 complementary         │
│  sub-queries → retrieve all →        │
│  merge into C_anchor (15-25 docs)    │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  Stage 2: ESC-Gated Refinement       │
│  EvidenceSufficiencyController       │
│  → STOP (sufficient) or              │
│    CONTINUE + follow-up query →      │
│  retrieve + merge → repeat           │
│  (max 4 hops, dedup follow-ups)      │
└────────────────┬─────────────────────┘
                 │
                 ▼
           GenerationAgent
```

**Component mapping** (our implementation → PAR2 paper):
- Sub-query decomposition: `decompose_query()` in `multi_query.py`
- Coverage anchoring: `anchor_node` in `langgraph_rag.py`
- ESC: `EvidenceSufficiencyController` in `esc.py`
- Refinement loop: `refine_node` in `langgraph_rag.py`

### Ablation Results (n=30, HotpotQA dev set)

All PAR2 runs use the same generation stack as the best pipeline (CoT prompt + 2 few-shot examples + hybrid BM25+FAISS+CrossEncoder retrieval). PAR2 **replaces only the IRCoT iterative loop** with its two-stage anchoring + ESC refinement.

**GPT-3.5 ablation** (best pipeline = IRCoT + hybrid + CoT + few-shot):

| Version | Retrieval | Generation | semF1 | semF1 ≥0.8 | ctxP | ctxR | noise |
|---------|-----------|------------|-------|------------|------|------|-------|
| Baseline | FAISS only | direct | 0.416 | 43.3% | 0.613 | 0.697 | 0.244 |
| IRCoT (standalone) | IRCoT | direct | 0.586 | 60.0% | 0.749 | 0.783 | 0.400 |
| **Best pipeline** | **IRCoT + hybrid** | **CoT + few-shot** | **0.705** | **73.3%** | — | — | — |
| PAR2 v1 (no dedup) | PAR2 + hybrid | CoT + few-shot | 0.560 | 56.7% | — | — | 0.144 |
| PAR2 v2 (dedup + ctx eval) | PAR2 + hybrid | CoT + few-shot | 0.622 | 63.3% | 0.422 | 0.483 | 0.133 |

**GPT-4o-mini ablation** (same pipeline config, generation model swapped):

| Version | Model | semF1 | semF1 ≥0.8 | ctxP | ctxR | noise | insufficient |
|---------|-------|-------|------------|------|------|-------|-------------|
| IRCoT (best pipeline) | GPT-4o-mini | 0.571 | 56.7% | 0.696 | 0.672 | 0.394 | 8/30 |
| PAR2 v2 | GPT-4o-mini | 0.572 | 60.0% | 0.453 | 0.511 | **0.139** | 9/30 |

**Key findings:**
- PAR2 v2 (0.622) vs best IRCoT pipeline (0.705) on GPT-3.5 — GPT-3.5's 4096-token limit truncates C_anchor (15-25 docs), leaving only 2-3 visible ("Lost in the Middle")
- GPT-4o-mini PAR2 ≈ IRCoT (0.572 vs 0.571) — the bottleneck shifted: both have 8-9/30 questions returning "insufficient context" regardless of retrieval method
- **Noise sensitivity improvement is consistent across models**: IRCoT 0.394 → PAR2 0.139 (-65%) on GPT-4o-mini, same as GPT-3.5
- Next step: recalibrate generation prompt for GPT-4o-mini (remove "insufficient context" fallback)

> **Note on ctxP/ctxR**: PAR2 evaluates context on the merged C_anchor set (15-25 docs), while IRCoT evaluates on a focused top-5 set. Lower ctxP is expected with more docs.

### Key Design Decisions

- **ESC deduplication**: `refine_node` tracks used follow-up queries; if ESC repeats a query, STOP immediately instead of re-fetching cached results
- **`requery → finalizer`**: In PAR2 mode the BC router's `requery` action routes to `finalizer` (accept), since Stage 2 already handles retrieval quality
- **DSPy MIPRO disabled**: `ReasoningAgent(compile_on_init=False)` skips prompt optimization at eval time

```bash
# Run PAR2-RAG evaluation
EMB_MODEL=text-embedding-3-small make par2
```

---

### Metric Definitions

| Metric | Description |
|--------|-------------|
| **semF1** | Semantic F1 — token-level F1 between predicted and gold answer (primary metric) |
| **ctxR** | Context Recall — fraction of gold supporting facts retrieved |
| **ctxP** | Context Precision — fraction of retrieved docs that are relevant |
| **faith** | Faithfulness — are claims in the answer supported by retrieved context (Ragas) |
| **rel** | Answer Relevancy — embedding similarity between answer and question (Ragas) |

> **Note on faithfulness/relevancy**: These Ragas metrics score low on very short answers (e.g., single entity names) due to embedding cosine similarity limitations. The semF1 metric is a more reliable indicator of actual answer quality.

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
