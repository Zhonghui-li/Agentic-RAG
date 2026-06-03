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

## Notes

- The endpoint is a sync `def`, so FastAPI runs it in a threadpool — the
  blocking LangGraph/LLM calls don't fight the event loop (the async bug the
  old rag_service hit).
- Decoupled from the legacy rag_service (no MongoDB / old pipeline) — this is a
  clean, independently deployable demo of the agent.
