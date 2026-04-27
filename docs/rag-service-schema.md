# Agentic RAG Service — Hosting Schema

> This document defines the interface, configuration, and infrastructure requirements
> for the Agentic RAG pipeline service. It serves as the contract between the RAG
> pipeline and the hosting infrastructure / platform team.

---

## What This Service Does

A multi-agent RAG pipeline for complex, multi-hop question answering. Given a user
question (and optionally prior conversation history), it retrieves relevant documents
from a knowledge base and generates a grounded answer using chain-of-thought reasoning.

**Core components:**
- **Hybrid retrieval** — BM25 + FAISS vector search + CrossEncoder reranker
- **IRCoT** — Iterative retrieval loop for multi-hop questions (up to 4 hops)
- **BC retrieval router** — Learned MLP that routes between IRCoT and anchor-based
  two-stage retrieval depending on retrieval quality
- **CoT generation** — Chain-of-thought prompting with few-shot examples
- **RAGAS evaluation** — Faithfulness, relevancy, and semantic F1 scoring per query

---

## API Endpoints

### `POST /query` — Synchronous

```json
Request:
{
  "question":   "string",          // required — the user's question
  "history":    [                  // optional — recent conversation turns
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "use_router": true               // optional — enable BC retrieval router (default: true)
}

Response:
{
  "answer":   "string",            // generated answer
  "question": "string",            // echoed input question
  "success":  true,                // whether the pipeline completed successfully
  "error":    "string | null"      // error message if success=false
}
```

### `POST /query/stream` — Server-Sent Events

Same request body as `/query`. Streams the response as SSE tokens:

```
data: {"type": "status", "content": "Analyzing question..."}
data: {"type": "status", "content": "Retrieving relevant documents..."}
data: {"type": "token",  "content": "The "}
data: {"type": "token",  "content": "answer "}
...
data: {"type": "done",   "content": ""}
```

### `GET /health`

```json
{
  "status":          "healthy",
  "agents_loaded":   true,
  "redis_connected": true
}
```

### `GET /metrics`

Prometheus metrics. Exposes per-endpoint request count, latency histograms,
cache hit/miss counters, and active request gauge.

---

## Configuration (Environment Variables)

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Used for LLM calls, embeddings, and RAGAS evaluation |

### Optional (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `GEN_LLM_MODEL` | `gpt-3.5-turbo` | Model for answer generation |
| `EMB_MODEL` | `text-embedding-ada-002` | Embedding model for vector search |
| `DSPY_MODEL` | `gpt-3.5-turbo` | Model for query reformulation (DSPy) |
| `EVAL_LLM_MODEL` | `gpt-3.5-turbo` | Model for RAGAS evaluation |
| `RETR_TOP_K` | `5` | Number of documents to retrieve per hop |
| `GEN_FORCE_ANSWER` | `1` | Force the model to answer even with thin context |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL for response caching |
| `CACHE_TTL` | `3600` | Cache time-to-live in seconds |
| `ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed CORS origins. Set to your hosted frontend domain in production (e.g. `https://your-domain.com`) |
| `VECTORSTORE_PATH` | `agent_integration/vectorstore-hotpot/hotpotqa_faiss` | Path to the FAISS index directory |
| `RAG_SERVICE_PORT` | `8001` | Port the service listens on |

---

## Data Dependencies

The following files must be present at startup. They are not included in the Docker
image and must be mounted or pre-loaded onto the host.

| File / Directory | Size | Purpose |
|-----------------|------|---------|
| `agent_integration/vectorstore-hotpot/hotpotqa_faiss/` | ~500 MB | FAISS vector index over ~10K Wikipedia passages (HotpotQA corpus) |
| `agent_integration/agents/offline_rl_router_policy_v2.pt` | ~50 KB | Trained V2 Oracle RL retrieval router weights |

> **Note on the knowledge base:** The current vectorstore is built from the HotpotQA
> Wikipedia corpus. To use a different knowledge base, build a new FAISS index with
> `agent_integration/scripts/build_vectorstore.py` and point `VECTORSTORE_PATH` to it.

---

## Infrastructure Requirements

| Resource | Minimum | Notes |
|----------|---------|-------|
| CPU | 4 cores | IRCoT multi-hop + CrossEncoder reranker are CPU-intensive |
| Memory | 4 GB RAM | FAISS index (~500 MB) + PyTorch models (~500 MB) stay in memory |
| Storage | 2 GB | Vectorstore + Python dependencies |
| GPU | Not required | All inference runs on CPU |
| Outbound network | Required | Calls OpenAI API for LLM, embeddings, and evaluation |
| Redis | Required | For response caching. Can be the same Redis instance used by other services |

---

## Current Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI + Uvicorn |
| Orchestration | LangGraph (StateGraph) |
| Query reformulation | DSPy |
| Vector search | FAISS (local file, in-memory) |
| Keyword search | BM25Okapi |
| Reranker | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM / Embeddings | OpenAI API |
| Evaluation | RAGAS (faithfulness, relevancy, noise sensitivity) |
| Caching | Redis |
| Monitoring | Prometheus metrics at `/metrics` |
| Containerization | Docker |

---

## How to Run Locally

```bash
# 1. Set environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Start all services (includes Redis, MongoDB, frontend, backend)
docker-compose up

# 3. RAG service is available at http://localhost:8001
# Test with:
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

---

## Open Questions for the Hosting Team

Before this service can be deployed, the following need to be decided:

1. **Hosting environment** — Where will the service run? (e.g., GCP, AWS, school
   server, Firebase + Cloud Run?) The RAG pipeline is a stateful Python process and
   cannot run directly on Firebase Hosting.

2. **Vector store strategy** — The current setup uses a local FAISS file. For cloud
   deployment this needs either: (a) a persistent volume mount, or (b) migration to
   a hosted vector database (Pinecone, Weaviate, etc.). What do other pipelines use?

3. **API key management** — The service requires an `OPENAI_API_KEY` at runtime.
   Is there a shared lab key for the hosted service, or does each pipeline manage
   its own key via a secrets manager?

4. **Service registry** — How does the Flask backend know which RAG service URLs
   are available? Currently hardcoded. Is there a plan for a config-driven registry
   so new pipeline variants can be added without changing backend code?

5. **CORS origin** — Set `ALLOWED_ORIGINS` to the frontend's deployed domain once
   the hosting URL is confirmed.
