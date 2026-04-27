# Full-Stack Local Development Guide

> How to run the complete application locally: Next.js frontend → Flask backend → RAG FastAPI service.
> For the RAG pipeline alone (no frontend), see [`rag-pipeline-onboarding.md`](./rag-pipeline-onboarding.md).

---

## Architecture

```
Browser (localhost:3000)
    ↓  HTTP
Flask Backend (localhost:5001)
    ↓  HTTP  POST /query
RAG Service   (localhost:8001)
    ↓
agent_integration/ pipeline (FAISS + BM25 + LLM)
    ↓
Redis (localhost:6379)   ← response cache
MongoDB (localhost:27017) ← conversation history
```

---

## Prerequisites

Install and start the local services:

```bash
# Redis
brew install redis
brew services start redis

# MongoDB
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Node.js + pnpm (for frontend)
# Install Node from https://nodejs.org, then:
npm install -g pnpm
```

Verify:
```bash
redis-cli ping          # should return PONG
mongosh --eval "db.runCommand({ping:1})"   # should return ok: 1
```

---

## Setup

Follow **Section 2** of [`rag-pipeline-onboarding.md`](./rag-pipeline-onboarding.md) to clone the repo, create the venv, install Python dependencies, and set up your `.env` file first.

Then install frontend dependencies:
```bash
cd webapp/frontend
pnpm install
```

---

## Start All Three Services

Open three terminal tabs from the **repo root** (`webapp/`).

### Terminal 1 — RAG Service (port 8001)

```bash
source venv/bin/activate
uvicorn rag_service.main:app --port 8001 --host 0.0.0.0
```

Wait for `RAG Service initialized successfully!` before starting the other services.
This takes ~20 seconds (BM25 index build over 24K docs + CrossEncoder model load).

### Terminal 2 — Flask Backend (port 5001)

```bash
source venv/bin/activate
cd webapp/backend
python app.py
```

### Terminal 3 — Frontend (port 3000)

```bash
cd webapp/frontend
pnpm run dev
```

Open `http://localhost:3000` in your browser.

---

## Using the Frontend

1. Sign up or log in (stored in MongoDB)
2. Select **RAG Agent** as the provider in the sidebar
3. Type a multi-hop question, e.g.:
   - *"What government position did the publisher of Jane's Fighting Ships hold?"*
   - *"Which city has a larger population, London or Paris?"*
4. The frontend streams the response back via Server-Sent Events

---

## Verify via curl (no frontend needed)

```bash
# Check RAG service health
curl http://localhost:8001/health

# Query the RAG service directly (fastest way to test)
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which country does the River Thames flow through?", "use_router": false}'

# Query through Flask backend (same path as the frontend)
CONV=$(curl -s -X POST http://127.0.0.1:5001/new_conversation \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "method": "rag-agent"}')
CONV_ID=$(echo $CONV | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -s -X POST http://127.0.0.1:5001/get_response \
  -H "Content-Type: application/json" \
  -d "{\"conv_id\":\"$CONV_ID\",\"method\":\"rag-agent\",\"messages\":[{\"role\":\"user\",\"content\":\"Which country does the River Thames flow through?\"}]}"
```

---

## Common Issues

**RAG service startup slow / timeout**
→ Normal on first run — CrossEncoder downloads ~90MB from HuggingFace. Subsequent starts use the cached model.

**Frontend can't reach backend**
→ The frontend hardcodes `http://127.0.0.1:5001`. Make sure the Flask backend is running on port 5001, not 5000.

**`CORS error` in browser console**
→ The RAG service allows all origins by default (`ALLOWED_ORIGINS=*`). If you've changed this, add `http://localhost:3000` to the list.

**Answers say "insufficient context"**
→ Make sure `GEN_FORCE_ANSWER=1` is set in your `.env` (repo root). Without it the model is allowed to refuse to answer.

**Redis cache returning stale answers**
→ Run `redis-cli FLUSHALL` to clear the cache, then retry.

---

## Hosting / Production

For Docker Compose setup and cloud deployment, see [`rag-service-schema.md`](./rag-service-schema.md). It covers:
- `docker-compose up` to run all services in containers
- Environment variables and their defaults
- Infrastructure requirements (CPU, memory, storage)
- Open questions for the hosting team (vector store strategy, API key management, CORS origin)
