"""
RAG Service - FastAPI wrapper for the RAG pipeline
"""
import os
import sys
import json
import re
import functools
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import redis

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import time
import uuid

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add agent_integration to path
AGENT_INTEGRATION_PATH = os.path.join(os.path.dirname(__file__), "..", "agent_integration")
sys.path.insert(0, AGENT_INTEGRATION_PATH)

# Change working directory to agent_integration so relative paths work
os.chdir(AGENT_INTEGRATION_PATH)

# Imports from agent_integration
from agents.reasoning_agent import ReasoningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent
from agents.langgraph_rag import run_rag_pipeline
from agents.reranker import create_cross_encoder_reranker
from agents.hybrid_retriever import HybridRetriever
from agents.multi_query import generate_query_variants

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from dspy.evaluate import SemanticF1
import dspy
import requests as _requests


class _DirectOpenAILLM:
    """Thin LLM wrapper using requests (avoids httpx issues) exposing .invoke()."""

    def __init__(self, model: str, api_key: str, max_tokens: int = 512, temperature: float = 0.0, timeout: float = 60.0):
        self._model = model
        self.model_name = model  # compat with LangChain introspection
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    def invoke(self, prompt: str):
        resp = _requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        class _Msg:
            def __init__(self, c):
                self.content = c

        return _Msg(content)


# Global agents (initialized on startup)
_agents = {}

# Redis client (initialized on startup)
_redis_client: Optional[redis.Redis] = None
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Default 1 hour

# Semantic cache embedding client (initialized on startup)
_cache_embeddings: Optional[OpenAIEmbeddings] = None
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))

# Guardrails scope checker LLM (initialized on startup)
_scope_llm: Optional[_DirectOpenAILLM] = None
GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
CORPUS_DESCRIPTION = os.getenv(
    "CORPUS_DESCRIPTION",
    "Wikipedia-based factual knowledge, including encyclopedic topics such as people, places, events, science, history, and culture (HotpotQA benchmark)"
)

# ============================================
# Prometheus Metrics
# ============================================
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total number of RAG requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'rag_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

CACHE_HITS = Counter(
    'rag_cache_hits_total',
    'Total number of cache hits'
)

CACHE_MISSES = Counter(
    'rag_cache_misses_total',
    'Total number of cache misses'
)

ACTIVE_REQUESTS = Gauge(
    'rag_active_requests',
    'Number of currently active requests'
)

PIPELINE_STAGE_LATENCY = Histogram(
    'rag_pipeline_stage_latency_seconds',
    'Latency of each pipeline stage',
    ['stage'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

OUT_OF_SCOPE_COUNT = Counter(
    'rag_out_of_scope_total',
    'Total number of questions rejected by guardrails as out-of-scope'
)

TOKENS_USED = Counter(
    'rag_tokens_used_total',
    'Total LLM tokens consumed',
    ['component']  # 'guardrail' | 'pipeline'
)


class _TokenTrackingCallback(BaseCallbackHandler):
    """Callback that reports LLM token usage directly to Prometheus."""
    def __init__(self, component: str):
        super().__init__()
        self.component = component

    def on_llm_end(self, response: LLMResult, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                usage = {}
                if hasattr(gen, 'generation_info') and gen.generation_info:
                    usage = gen.generation_info.get('token_usage', {})
                total = usage.get('total_tokens', 0)
                if total:
                    TOKENS_USED.labels(component=self.component).inc(total)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


_CONTEXT_PRONOUNS = re.compile(
    r'\b(he|she|they|it|his|her|their|its|him|them|this|that|these|those)\b',
    re.IGNORECASE
)

def _is_context_dependent(question: str) -> bool:
    """Return True if the question contains pronouns that require conversation context."""
    return bool(_CONTEXT_PRONOUNS.search(question))


def semantic_cache_lookup(question: str):
    """Find a semantically similar cached answer. Returns (answer, sources) or (None, []).
    Skips cache for context-dependent questions (containing pronouns) to avoid
    returning answers resolved from a different conversation context.
    """
    if _is_context_dependent(question):
        return None, []
    if _redis_client is None or _cache_embeddings is None:
        return None, []
    try:
        query_emb = _cache_embeddings.embed_query(question)
        keys = _redis_client.keys("rag:*")
        best_score, best_entry = 0.0, None
        for key in keys:
            data = _redis_client.get(key)
            if not data:
                continue
            entry = json.loads(data)
            if "embedding" not in entry:
                continue
            score = _cosine_similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score, best_entry = score, entry
        if best_score >= SEMANTIC_CACHE_THRESHOLD and best_entry:
            return best_entry.get("answer"), best_entry.get("sources", [])
    except Exception as e:
        print(f"Semantic cache lookup error: {e}")
    return None, []


def set_cached_response_semantic(question: str, answer: str, sources: List[str] = []) -> None:
    """Cache answer with its query embedding and sources for semantic lookup.
    Skips caching for context-dependent questions to avoid polluting the cache
    with answers that are only valid in a specific conversation context.
    """
    if _is_context_dependent(question):
        return
    if _redis_client is None or _cache_embeddings is None:
        return
    try:
        query_emb = _cache_embeddings.embed_query(question)
        cache_key = f"rag:{abs(hash(question.strip().lower()))}"
        entry = {"embedding": query_emb, "answer": answer, "sources": sources}
        _redis_client.setex(cache_key, CACHE_TTL, json.dumps(entry))
    except Exception as e:
        print(f"Semantic cache set error: {e}")


def check_scope(question: str) -> bool:
    """
    Returns True if the question is within corpus scope, False if out-of-scope.
    Uses a cheap LLM call (gpt-3.5-turbo, max_tokens=5) to classify.
    Defaults to True (in-scope) on any error to avoid false rejections.
    """
    if not GUARDRAILS_ENABLED or _scope_llm is None:
        return True
    try:
        prompt = (
            f"You are a scope checker for a QA system whose knowledge base covers:\n"
            f"{CORPUS_DESCRIPTION}\n\n"
            f"Question: {question}\n\n"
            f"Reply 'no' ONLY if this question is clearly unrelated to the knowledge base "
            f"(e.g. coding help, recipes, personal advice, math problems). "
            f"If the question could plausibly involve a person, place, event, or fact "
            f"that might appear in Wikipedia, reply 'yes'. When in doubt, reply 'yes'.\n"
            f"Reply with only 'yes' or 'no'."
        )
        response = _scope_llm.invoke(prompt)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            TOKENS_USED.labels(component="guardrail").inc(
                response.usage_metadata.get('total_tokens', 0)
            )
        return response.content.strip().lower().startswith("yes")
    except Exception as e:
        print(f"Scope check error (defaulting to in-scope): {e}")
        return True


def extract_sources(docs: list) -> List[str]:
    """Extract unique document titles from retrieved docs.
    Prefers metadata['title']; falls back to parsing 'Title: content' page_content format.
    """
    seen, titles = set(), []
    for doc in docs:
        meta = getattr(doc, 'metadata', None) or (doc.get('metadata', {}) if isinstance(doc, dict) else {})
        title = (meta or {}).get('title', '').strip()

        # Fallback: parse from "Title: content text" page_content format
        # Only valid if colon appears before the first period (title-like prefix)
        if not title:
            content = getattr(doc, 'page_content', None) or (doc.get('page_content', '') if isinstance(doc, dict) else '')
            if content and ':' in content:
                colon_idx = content.index(':')
                first_period = content.index('.') if '.' in content else len(content)
                if colon_idx < first_period:
                    candidate = content[:colon_idx].strip()
                    if 0 < len(candidate) <= 80:
                        title = candidate

        _invalid = {'not provided', 'n/a', 'none', 'unknown', ''}
        if title and title.lower() not in _invalid and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def build_conversation_context(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert flat message history into [{q, a}] pairs for ReasoningAgent pronoun resolution."""
    if not history:
        return []
    context = []
    msgs = history[-6:]  # last 3 turns (6 messages)
    i = 0
    while i < len(msgs) - 1:
        if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
            context.append({
                "q": msgs[i].get("content", "").strip(),
                "a": msgs[i + 1].get("content", "").strip(),
            })
            i += 2
        else:
            i += 1
    return context




class QueryRequest(BaseModel):
    question: str
    use_router: bool = False  # Whether to use LangGraph router
    history: List[Dict[str, str]] = []  # Conversation history: [{"role": "user"/"assistant", "content": "..."}]


class QueryResponse(BaseModel):
    answer: str
    question: str
    success: bool
    sources: List[str] = []
    confidence: Optional[float] = None
    request_id: Optional[str] = None
    error: Optional[str] = None


def init_agents():
    """Initialize all agents and vectorstore"""
    global _agents, _scope_llm

    # Environment variables
    OPENAI_API_KEY = "".join((os.getenv("OPENAI_API_KEY") or "").split())
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    VECTORSTORE_PATH = os.getenv(
        "VECTORSTORE_PATH",
        os.path.join(AGENT_INTEGRATION_PATH, "vectorstore-hotpot", "hotpotqa_faiss")
    )

    print(f"Initializing RAG Service...")
    print(f"  - Vectorstore path: {VECTORSTORE_PATH}")

    # Configure DSPy
    dspy.configure(
        lm=dspy.LM(
            model=os.getenv("DSPY_MODEL", "gpt-3.5-turbo"),
            api_base="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(os.getenv("DSPY_MAX_TOKENS", "384")),
            timeout=30,
        )
    )

    # Scope checker LLM (cheap: gpt-3.5-turbo, only needs yes/no)
    if GUARDRAILS_ENABLED:
        _scope_llm = _DirectOpenAILLM(
            model="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            max_tokens=5,
            temperature=0.0,
            timeout=10.0,
        )
        print(f"  - Guardrails enabled (corpus: {CORPUS_DESCRIPTION[:60]}...)")
    else:
        print("  - Guardrails disabled")

    # Generation LLM (direct openai SDK — avoids LangChain httpx issues in Cloud Run)
    gen_llm = _DirectOpenAILLM(
        model=os.getenv("GEN_LLM_MODEL", "gpt-3.5-turbo"),
        api_key=OPENAI_API_KEY,
        max_tokens=int(os.getenv("GEN_MAX_TOKENS", "512")),
        temperature=0.0,
        timeout=60.0,
    )

    # Evaluation LLM (direct openai SDK)
    eval_llm = _DirectOpenAILLM(
        model=os.getenv("EVAL_LLM_MODEL", "gpt-3.5-turbo"),
        api_key=OPENAI_API_KEY,
        max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "1024")),
        temperature=0.0,
        timeout=60.0,
    )

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMB_MODEL", "text-embedding-ada-002"),
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
    )

    # Load vectorstore
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"  - Vectorstore loaded successfully")

    # Initialize agents
    semantic_f1_metric = SemanticF1(decompositional=True)

    reasoning_agent = ReasoningAgent()
    evaluation_agent = EvaluationAgent(llm=eval_llm)
    hybrid_retriever = HybridRetriever(vectorstore)
    reranker = create_cross_encoder_reranker(
        model_name=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_n=int(os.getenv("RERANKER_TOP_N", "5")),
    )
    multi_query_fn = functools.partial(generate_query_variants, llm=gen_llm, n_variants=2)
    retrieval_agent = RetrievalAgent(
        vectorstore=vectorstore,
        evaluation_agent=evaluation_agent,
        top_k=int(os.getenv("RETR_TOP_K", "5")),
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        multi_query_fn=multi_query_fn,
    )
    generation_agent = GenerationAgent(
        llm=gen_llm,
        semantic_f1_metric=semantic_f1_metric
    )

    _agents = {
        "reasoning": reasoning_agent,
        "retrieval": retrieval_agent,
        "generation": generation_agent,
        "evaluation": evaluation_agent,
    }

    print("RAG Service initialized successfully!")


def init_redis():
    """Initialize Redis connection and semantic cache embedding client"""
    global _redis_client, _cache_embeddings
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        OPENAI_API_KEY = "".join((os.getenv("OPENAI_API_KEY") or "").split())
        _cache_embeddings = OpenAIEmbeddings(
            model=os.getenv("EMB_MODEL", "text-embedding-ada-002"),
            api_key=OPENAI_API_KEY,
            base_url="https://api.openai.com/v1",
        )
        print(f"  - Redis connected: {redis_url} (semantic caching enabled)")
    except Exception as e:
        print(f"  - Redis connection failed: {e} (caching disabled)")
        _redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    init_redis()
    init_agents()
    yield
    # Shutdown
    if _redis_client:
        _redis_client.close()
    print("RAG Service shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RAG Service",
    description="RAG Pipeline API for agentic question answering",
    version="1.0.0",
    lifespan=lifespan
)

# CORS: set ALLOWED_ORIGINS env var once hosting domain is confirmed.
# Example: ALLOWED_ORIGINS=https://your-domain.com,http://localhost:3000
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    redis_connected = False
    if _redis_client:
        try:
            _redis_client.ping()
            redis_connected = True
        except Exception:
            pass
    return {
        "status": "healthy",
        "agents_loaded": len(_agents) > 0,
        "redis_connected": redis_connected
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if _redis_client is None:
        return {"enabled": False, "message": "Redis not connected"}
    try:
        info = _redis_client.info("stats")
        keys = _redis_client.keys("rag:*")
        return {
            "enabled": True,
            "cached_queries": len(keys),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "ttl_seconds": CACHE_TTL
        }
    except Exception as e:
        return {"enabled": False, "error": str(e)}


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached queries"""
    if _redis_client is None:
        return {"success": False, "message": "Redis not connected"}
    try:
        keys = _redis_client.keys("rag:*")
        if keys:
            _redis_client.delete(*keys)
        return {"success": True, "cleared": len(keys)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG pipeline with a question
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    ACTIVE_REQUESTS.inc()

    if not _agents:
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(endpoint="/query", status="error").inc()
        raise HTTPException(status_code=503, detail="Agents not initialized")

    try:
        # Check semantic cache first
        cached_answer, cached_sources = semantic_cache_lookup(request.question)
        if cached_answer:
            print(f"Cache HIT for: {request.question[:50]}...")
            CACHE_HITS.inc()
            REQUEST_COUNT.labels(endpoint="/query", status="success").inc()
            REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start_time)
            ACTIVE_REQUESTS.dec()
            return QueryResponse(
                answer=cached_answer,
                question=request.question,
                success=True,
                sources=cached_sources,
                request_id=request_id
            )

        print(f"Cache MISS for: {request.question[:50]}...")
        CACHE_MISSES.inc()

        # Guardrails: reject out-of-scope questions before running the pipeline
        if not check_scope(request.question):
            print(f"OUT-OF-SCOPE: {request.question[:80]}")
            OUT_OF_SCOPE_COUNT.inc()
            REQUEST_COUNT.labels(endpoint="/query", status="out_of_scope").inc()
            REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start_time)
            ACTIVE_REQUESTS.dec()
            return QueryResponse(
                answer=(
                    "This question appears to be outside the scope of my knowledge base. "
                    "I'm designed to answer questions grounded in the documents I have access to. "
                    "Please try asking about topics within that scope."
                ),
                question=request.question,
                success=True
            )

        # Build structured conversation context for pronoun resolution
        conv_context = build_conversation_context(request.history)
        if conv_context:
            print(f"  - Using {len(conv_context)} prior turn(s) for memory context")

        # Run pipeline with timing + token tracking via DSPy history
        pipeline_start = time.time()
        lm = dspy.settings.lm
        history_before = len(lm.history) if lm and hasattr(lm, 'history') else 0
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_rag_pipeline(
                question=request.question,
                retrieval_agent=_agents["retrieval"],
                reasoning_agent=_agents["reasoning"],
                generation_agent=_agents["generation"],
                evaluation_agent=_agents["evaluation"],
                use_router=request.use_router,
                visualize=False,
                conversation_context=conv_context,
            )
        )
        PIPELINE_STAGE_LATENCY.labels(stage="full_pipeline").observe(time.time() - pipeline_start)
        if lm and hasattr(lm, 'history'):
            new_calls = lm.history[history_before:]
            pipeline_tokens = sum(
                c.get('usage', {}).get('total_tokens', 0)
                for c in new_calls
                if isinstance(c, dict)
            )
            if pipeline_tokens:
                TOKENS_USED.labels(component="pipeline").inc(pipeline_tokens)

        answer = result.get("answer", "")
        sources = extract_sources(result.get("docs", []))
        raw_confidence = result.get("faithfulness_score")
        confidence = float(raw_confidence) if raw_confidence is not None else None

        # Cache the result with semantic embedding and sources
        set_cached_response_semantic(request.question, answer, sources)

        REQUEST_COUNT.labels(endpoint="/query", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()

        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        print(f"[{request_id}] sources={sources} confidence={conf_str}")
        return QueryResponse(
            answer=answer,
            question=request.question,
            success=True,
            sources=sources,
            confidence=confidence,
            request_id=request_id
        )

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        REQUEST_COUNT.labels(endpoint="/query", status="error").inc()
        REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()
        return QueryResponse(
            answer="",
            question=request.question,
            success=False,
            request_id=request_id,
            error=str(e)
        )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Stream the RAG pipeline response using Server-Sent Events (SSE)
    """
    if not _agents:
        raise HTTPException(status_code=503, detail="Agents not initialized")

    async def generate_stream():
        import concurrent.futures

        # Check semantic cache first
        cached_answer, cached_sources = semantic_cache_lookup(request.question)
        conv_context = build_conversation_context(request.history)

        if cached_answer:
            # Stream cached response word by word
            yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieved from cache...'})}\n\n"
            await asyncio.sleep(0.1)

            words = cached_answer.split()
            for i, word in enumerate(words):
                yield f"data: {json.dumps({'type': 'token', 'content': word + (' ' if i < len(words) - 1 else '')})}\n\n"
                await asyncio.sleep(0.03)

            yield f"data: {json.dumps({'type': 'done', 'content': '', 'sources': cached_sources})}\n\n"
            return

        # Guardrails: reject out-of-scope questions before running the pipeline
        if not check_scope(request.question):
            print(f"OUT-OF-SCOPE (stream): {request.question[:80]}")
            OUT_OF_SCOPE_COUNT.inc()
            out_of_scope_msg = (
                "This question appears to be outside the scope of my knowledge base. "
                "I'm designed to answer questions grounded in the documents I have access to. "
                "Please try asking about topics within that scope."
            )
            yield f"data: {json.dumps({'type': 'token', 'content': out_of_scope_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            return

        # Not cached - run the pipeline
        yield f"data: {json.dumps({'type': 'status', 'content': 'Analyzing question...'})}\n\n"
        await asyncio.sleep(0.2)

        yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieving relevant documents...'})}\n\n"

        # Run pipeline in thread pool to not block
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            try:
                lm = dspy.settings.lm
                history_before = len(lm.history) if lm and hasattr(lm, 'history') else 0

                result = await loop.run_in_executor(
                    pool,
                    lambda: run_rag_pipeline(
                        question=request.question,
                        retrieval_agent=_agents["retrieval"],
                        reasoning_agent=_agents["reasoning"],
                        generation_agent=_agents["generation"],
                        evaluation_agent=_agents["evaluation"],
                        use_router=request.use_router,
                        visualize=False,
                        conversation_context=conv_context,
                    )
                )
                if lm and hasattr(lm, 'history'):
                    new_calls = lm.history[history_before:]
                    pipeline_tokens = sum(
                        c.get('usage', {}).get('total_tokens', 0)
                        for c in new_calls
                        if isinstance(c, dict)
                    )
                    if pipeline_tokens:
                        TOKENS_USED.labels(component="pipeline").inc(pipeline_tokens)

                answer = result.get("answer", "")
                sources = extract_sources(result.get("docs", []))

                # Cache the result with semantic embedding and sources
                set_cached_response_semantic(request.question, answer, sources)

                yield f"data: {json.dumps({'type': 'status', 'content': 'Generating response...'})}\n\n"
                await asyncio.sleep(0.1)

                # Stream answer word by word
                words = answer.split()
                for i, word in enumerate(words):
                    yield f"data: {json.dumps({'type': 'token', 'content': word + (' ' if i < len(words) - 1 else '')})}\n\n"
                    await asyncio.sleep(0.05)  # Typing effect

                yield f"data: {json.dumps({'type': 'done', 'content': '', 'sources': sources})}\n\n"

            except Exception as e:
                print(f"Error in streaming RAG pipeline: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", os.getenv("RAG_SERVICE_PORT", "8001")))
    uvicorn.run(app, host="0.0.0.0", port=port)
