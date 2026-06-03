# Slug Advisor — live demo service

A minimal FastAPI service that exposes the UCSC Slug Advisor agent over HTTP
with a single-page chat UI. The UI shows the agent's **tool-call trace**
(🔧 search_catalog / lookup_schedule / get_academic_calendar / web_search),
which is the headline feature.

## Run locally

From `agent_integration/`, with the env set:

```bash
export OPENAI_API_KEY=sk-...
export EMB_MODEL=text-embedding-3-small
export GEN_LLM_MODEL=gpt-4o-mini
uvicorn slug_service.app:app --port 8100
```

Then open http://localhost:8100

> First start loads the FAISS vectorstore + CrossEncoder reranker, so it takes
> a few seconds. Build the vectorstore first if needed:
> `python scripts/build_ucsc_vectorstore.py --courses data-ucsc/cse_courses.json --out vectorstore-ucsc/ucsc_cse_faiss`

## API

- `GET /` — chat UI
- `GET /health` — `{"status":"ok","agent_loaded":true}`
- `POST /advisor` — body `{"question": "..."}` →
  `{"answer": "...", "trace": [{"tool","args"}], "tools_used": [...]}`

## Abuse / cost controls (for public deploy)

OpenAI + GCP here are shared lab accounts, so the app caps how many LLM calls the
public can trigger (don't rely on the OpenAI dashboard limit). All tunable via env:

| Env var | Default | What it does |
|---|---|---|
| `MAX_INPUT_CHARS` | 500 | reject over-long questions (token-inflation abuse) -> 400 |
| `RATE_LIMIT_PER_MIN` | 6 | per-IP requests/minute -> 429 |
| `DAILY_QUOTA` | 150 | hard ceiling on requests/day -> 429 ("try tomorrow") |

Worst-case cost (gpt-4o-mini ~$0.005/query): `DAILY_QUOTA=150` with
`--max-instances=1` ≈ $0.75/day ≈ **$22/month cap**, regardless of the lab key's
high limit. Deploy with **max-instances=1** so the in-memory quota is an exact
global cap (and it's cheaper). Realistic per-query cost is lower (~$0.001-0.002).

The agent also stays in scope (declines off-topic requests like "write me a poem")
so the demo can't be used as a free general-purpose LLM.

**Deploy checklist:** `gcloud run deploy ... --max-instances=1`; set a GCP billing
alert; optionally ask the lab for a project-scoped OpenAI key with its own low cap.

## Notes

- The endpoint is a sync `def`, so FastAPI runs it in a threadpool — the
  blocking LangGraph/LLM calls don't fight the event loop (the async bug the
  old rag_service hit).
- Decoupled from the legacy rag_service (no MongoDB / old pipeline) — this is a
  clean, independently deployable demo of the agent.
