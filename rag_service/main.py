"""
RAG Service - FastAPI wrapper for the RAG pipeline
"""
import os
import sys
import json
import hashlib
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

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dspy.evaluate import SemanticF1
import dspy


# Global agents (initialized on startup)
_agents = {}

# Redis client (initialized on startup)
_redis_client: Optional[redis.Redis] = None
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Default 1 hour

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


def get_cache_key(question: str, use_router: bool, history: List[Dict[str, str]] = []) -> str:
    """Generate a cache key from the question and conversation history"""
    history_str = json.dumps(history, ensure_ascii=False) if history else ""
    content = f"{question.strip().lower()}:{use_router}:{history_str}"
    return f"rag:{hashlib.md5(content.encode()).hexdigest()}"


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


def get_cached_response(key: str) -> Optional[dict]:
    """Get cached response from Redis"""
    if _redis_client is None:
        return None
    try:
        data = _redis_client.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        print(f"Redis get error: {e}")
    return None


def set_cached_response(key: str, response: dict) -> None:
    """Cache response in Redis"""
    if _redis_client is None:
        return
    try:
        _redis_client.setex(key, CACHE_TTL, json.dumps(response))
    except Exception as e:
        print(f"Redis set error: {e}")


class QueryRequest(BaseModel):
    question: str
    use_router: bool = True  # Whether to use LangGraph router
    history: List[Dict[str, str]] = []  # Conversation history: [{"role": "user"/"assistant", "content": "..."}]


class QueryResponse(BaseModel):
    answer: str
    question: str
    success: bool
    error: Optional[str] = None


def init_agents():
    """Initialize all agents and vectorstore"""
    global _agents

    # Environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

    # Generation LLM
    gen_llm = ChatOpenAI(
        model=os.getenv("GEN_LLM_MODEL", "gpt-3.5-turbo"),
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
        temperature=0.0,
        max_tokens=int(os.getenv("GEN_MAX_TOKENS", "512")),
        timeout=60.0,
    )

    # Evaluation LLM
    eval_llm = ChatOpenAI(
        model=os.getenv("EVAL_LLM_MODEL", "gpt-3.5-turbo"),
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
        temperature=0.0,
        max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "1024")),
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
    retrieval_agent = RetrievalAgent(
        vectorstore=vectorstore,
        evaluation_agent=evaluation_agent,
        top_k=int(os.getenv("RETR_TOP_K", "5"))
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
    """Initialize Redis connection"""
    global _redis_client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        print(f"  - Redis connected: {redis_url}")
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
# Example: ALLOWED_ORIGINS=https://aiea-lab.com,http://localhost:3000
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
    ACTIVE_REQUESTS.inc()

    if not _agents:
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(endpoint="/query", status="error").inc()
        raise HTTPException(status_code=503, detail="Agents not initialized")

    try:
        # Check cache first
        cache_key = get_cache_key(request.question, request.use_router, request.history)
        cached = get_cached_response(cache_key)
        if cached:
            print(f"Cache HIT for: {request.question[:50]}...")
            CACHE_HITS.inc()
            REQUEST_COUNT.labels(endpoint="/query", status="success").inc()
            REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start_time)
            ACTIVE_REQUESTS.dec()
            return QueryResponse(
                answer=cached.get("answer", ""),
                question=request.question,
                success=True
            )

        print(f"Cache MISS for: {request.question[:50]}...")
        CACHE_MISSES.inc()

        # Build structured conversation context for pronoun resolution
        conv_context = build_conversation_context(request.history)
        if conv_context:
            print(f"  - Using {len(conv_context)} prior turn(s) for memory context")

        # Run pipeline with timing
        pipeline_start = time.time()
        result = run_rag_pipeline(
            question=request.question,
            retrieval_agent=_agents["retrieval"],
            reasoning_agent=_agents["reasoning"],
            generation_agent=_agents["generation"],
            evaluation_agent=_agents["evaluation"],
            use_router=request.use_router,
            visualize=False,
            conversation_context=conv_context,
        )
        PIPELINE_STAGE_LATENCY.labels(stage="full_pipeline").observe(time.time() - pipeline_start)

        answer = result.get("answer", "")

        # Cache the result
        set_cached_response(cache_key, {"answer": answer})

        REQUEST_COUNT.labels(endpoint="/query", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()

        return QueryResponse(
            answer=answer,
            question=request.question,
            success=True
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

        # Check cache first
        cache_key = get_cache_key(request.question, request.use_router, request.history)
        cached = get_cached_response(cache_key)
        conv_context = build_conversation_context(request.history)

        if cached:
            # Stream cached response word by word
            yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieved from cache...'})}\n\n"
            await asyncio.sleep(0.1)

            answer = cached.get("answer", "")
            words = answer.split()
            for i, word in enumerate(words):
                yield f"data: {json.dumps({'type': 'token', 'content': word + (' ' if i < len(words) - 1 else '')})}\n\n"
                await asyncio.sleep(0.03)  # Small delay for visual effect

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

                answer = result.get("answer", "")

                # Cache the result
                set_cached_response(cache_key, {"answer": answer})

                yield f"data: {json.dumps({'type': 'status', 'content': 'Generating response...'})}\n\n"
                await asyncio.sleep(0.1)

                # Stream answer word by word
                words = answer.split()
                for i, word in enumerate(words):
                    yield f"data: {json.dumps({'type': 'token', 'content': word + (' ' if i < len(words) - 1 else '')})}\n\n"
                    await asyncio.sleep(0.05)  # Typing effect

                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

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
    port = int(os.getenv("RAG_SERVICE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
