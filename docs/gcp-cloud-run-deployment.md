# GCP Cloud Run Deployment Guide

> How to deploy and redeploy the Agentic RAG system to GCP Cloud Run.
> For running the system locally, see [`fullstack-local-dev.md`](./fullstack-local-dev.md).

---

## Architecture

Three independent Cloud Run services, each with its own public URL:

```
Browser
    ↓  HTTPS
Next.js Frontend   (Cloud Run, port 3000)
    ↓  HTTPS
Flask Backend      (Cloud Run, port 8080)
    ↓  HTTPS  POST /query
RAG Service        (Cloud Run, port 8080)
    ↓
FAISS vectorstore  (downloaded from Cloud Storage on startup)
OpenAI API         (key from Secret Manager)
MongoDB Atlas      (URI from Secret Manager)
```

---

## GCP Resources

| Resource | Name / ID |
|----------|-----------|
| Project | `proslm` (ID: `759005971862`) |
| Region | `us-central1` |
| Artifact Registry | `us-central1-docker.pkg.dev/proslm/proslm-repo` |
| Cloud Storage | `gs://proslm-vectorstore/hotpotqa_faiss_v3` |
| Secret Manager | `OPENAI_API_KEY`, `MONGO_URI` |

---

## Prerequisites

```bash
# Authenticate with GCP
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev

# Verify Docker is running
docker info
```

Make sure you have been added to the `proslm` GCP project with at least **Editor** access.

---

## Deployed Services

| Service | URL | Current Image |
|---------|-----|---------------|
| RAG Service | https://rag-service-759005971862.us-central1.run.app | `rag-service:v8` |
| Flask Backend | https://backend-service-759005971862.us-central1.run.app | `backend-service:v1` |
| Next.js Frontend | https://frontend-service-759005971862.us-central1.run.app | `frontend-service:v2` |

Verify all three are up:
```bash
curl https://rag-service-759005971862.us-central1.run.app/health
# Expected: {"status":"healthy","agents_loaded":true}
```

---

## Redeploying a Service

All three services follow the same three-step pattern: **build → push → deploy**.
Always increment the version tag (`vN`) when rebuilding.

### RAG Service

```bash
# 1. Build (from repo root)
docker build --platform linux/amd64 --provenance=false \
  -f rag_service/Dockerfile.cloudrun \
  -t us-central1-docker.pkg.dev/proslm/proslm-repo/rag-service:vN .

# 2. Push
docker push us-central1-docker.pkg.dev/proslm/proslm-repo/rag-service:vN

# 3. Deploy (update the image tag in deploy_rag.sh first, then run)
bash deploy_rag.sh
```

### Flask Backend

```bash
# 1. Build (from repo root)
docker build --platform linux/amd64 --provenance=false \
  -f webapp/backend/Dockerfile.cloudrun \
  -t us-central1-docker.pkg.dev/proslm/proslm-repo/backend-service:vN \
  webapp/backend/

# 2. Push
docker push us-central1-docker.pkg.dev/proslm/proslm-repo/backend-service:vN

# 3. Deploy (update image tag in deploy_backend.sh first, then run)
bash deploy_backend.sh
```

### Next.js Frontend

`NEXT_PUBLIC_BACKEND_URL` must be injected at **build time** — it gets baked into the static assets.

```bash
# 1. Build (from repo root)
docker build --platform linux/amd64 --provenance=false \
  -f webapp/frontend/Dockerfile.cloudrun \
  --build-arg NEXT_PUBLIC_BACKEND_URL=https://backend-service-759005971862.us-central1.run.app \
  -t us-central1-docker.pkg.dev/proslm/proslm-repo/frontend-service:vN \
  webapp/frontend/

# 2. Push
docker push us-central1-docker.pkg.dev/proslm/proslm-repo/frontend-service:vN

# 3. Deploy (update image tag in deploy_frontend.sh first, then run)
bash deploy_frontend.sh
```

---

## Secrets Management

Secrets are stored in GCP Secret Manager and injected as environment variables at runtime.

**Do not use `echo` to write secrets** — it appends a `\n` that corrupts the value.

```bash
# Correct: no trailing newline
printf '%s' 'sk-proj-...' | gcloud secrets versions add OPENAI_API_KEY \
  --data-file=- --project=proslm

# Verify the stored value is clean (should print the key with no extra characters)
gcloud secrets versions access latest --secret=OPENAI_API_KEY --project=proslm | cat -A
# The last character should be $ (end of line), NOT ^M$ or \n$
```

To update an existing secret:
```bash
gcloud secrets versions add OPENAI_API_KEY --data-file=- --project=proslm <<< "$(printf '%s' 'sk-proj-NEW-KEY')"
```

---

## Common Issues

**`Connection error` from OpenAI API (no traceback, just connection error)**
→ Almost always means the API key in Secret Manager has a trailing newline or whitespace. Re-store the key using `printf '%s'` (see Secrets Management above). The error is misleading because `httpx` wraps invalid header errors as connection errors.

**`RuntimeError: This event loop is already running`**
→ A LangChain or httpx call is being made directly inside a FastAPI `async` handler. Wrap the call in `await loop.run_in_executor(None, lambda: ...)` to move it off the main event loop.

**FAISS retrieval fails silently, returns empty results**
→ OpenAI embedding call is failing inside the async context (same root cause as above). The hybrid retriever falls back to BM25 automatically — check logs for `FAISS failed, falling back to BM25`.

**`--platform linux/amd64` is required**
→ Building on Apple Silicon (M1/M2/M3) without this flag produces an ARM64 image that Cloud Run cannot run.

**`PORT` env var conflict**
→ Cloud Run injects `PORT=8080` automatically. Do not set `--set-env-vars PORT=...` in deploy scripts — it's a reserved variable. Use `--port` in the deploy command instead.

**Cold start delay (~10–30s) on first request**
→ Expected behavior. Cloud Run scales to zero when idle. The RAG service takes longer because it downloads the FAISS vectorstore from Cloud Storage on startup.

**`ERR_UNKNOWN_BUILTIN_MODULE` during frontend Docker build**
→ `pnpm@latest` (v11+) is incompatible with Node 20. The Dockerfile pins `pnpm@10.28.2` — do not change this to `latest`.
