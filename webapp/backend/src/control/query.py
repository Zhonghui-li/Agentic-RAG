import os
from typing import Optional, Literal, Tuple
import openai
import anthropic
import google.generativeai as genai
import json
import time
import requests

# Initialize API clients
openai_client = None
anthropic_client = None
genai_initialized = False


def get_openai_client():
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        openai_client = openai.OpenAI(api_key=api_key)
    return openai_client


def get_anthropic_client():
    global anthropic_client
    if anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        anthropic_client = anthropic.Anthropic(api_key=api_key)
    return anthropic_client


def init_genai():
    global genai_initialized
    if not genai_initialized:
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        genai_initialized = True


def query_openai(prompt, system_prompt=None, model="gpt-4o-mini"):
    print(f"[DEBUG] Sending query to OpenAI using model: {model}")
    start_time = time.time()

    client = get_openai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(model=model, messages=messages)

    elapsed_time = time.time() - start_time
    print(f"[DEBUG] OpenAI response received in {elapsed_time:.2f} seconds")

    return response.choices[0].message.content


def query_claude(prompt, system_prompt=None, model="claude-3-opus-20240229"):
    print(f"[DEBUG] Sending query to Claude using model: {model}")
    start_time = time.time()

    client = get_anthropic_client()

    messages = [{"role": "user", "content": prompt}]
    request_params = {"model": model, "messages": messages, "max_tokens": 2000}

    if system_prompt:
        request_params["system"] = system_prompt

    response = client.messages.create(**request_params)

    elapsed_time = time.time() - start_time
    print(f"[DEBUG] Claude response received in {elapsed_time:.2f} seconds")

    return response.content[0].text


def query_gemini(prompt, system_prompt=None, model="gemini-1.5-pro-latest"):
    print(f"[DEBUG] Sending query to Gemini using model: {model}")
    start_time = time.time()

    init_genai()

    if not model.startswith("models/"):
        model_name = f"models/{model}"
    else:
        model_name = model

    if model_name == "models/gemini-pro":
        model_name = "models/gemini-1.5-pro-latest"

    try:
        genai_model = genai.GenerativeModel(model_name)

        if system_prompt:
            prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        response = genai_model.generate_content(prompt)

        elapsed_time = time.time() - start_time
        print(f"[DEBUG] Gemini response received in {elapsed_time:.2f} seconds")

        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini error: {str(e)}")
        raise


def query_rag_service(query_str: str, use_router: bool = False, history: list = []) -> str:
    """Query the RAG Service for agentic RAG responses."""
    RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")

    print(f"[DEBUG] Sending query to RAG Service at {RAG_SERVICE_URL}")
    start_time = time.time()

    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/query",
            json={
                "question": query_str,
                "use_router": use_router,
                "history": history
            },
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        elapsed_time = time.time() - start_time
        print(f"[DEBUG] RAG Service response received in {elapsed_time:.2f} seconds")

        if result.get("success"):
            return result.get("answer", "")
        else:
            error_msg = result.get("error", "Unknown error from RAG Service")
            print(f"[ERROR] RAG Service error: {error_msg}")
            raise Exception(f"RAG Service error: {error_msg}")

    except requests.exceptions.ConnectionError:
        raise Exception(f"RAG Service unavailable. Please ensure it's running at {RAG_SERVICE_URL}")
    except requests.exceptions.Timeout:
        raise Exception("RAG Service request timed out")
    except Exception as e:
        print(f"[ERROR] RAG Service error: {str(e)}")
        raise


def stream_rag_service(query_str: str, use_router: bool = False, history: list = []):
    """Stream responses from the RAG Service using Server-Sent Events."""
    RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")

    print(f"[DEBUG] Streaming from RAG Service at {RAG_SERVICE_URL}")

    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/query/stream",
            json={
                "question": query_str,
                "use_router": use_router,
                "history": history
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    yield decoded + '\n\n'

    except requests.exceptions.ConnectionError:
        error_msg = f"RAG Service unavailable at {RAG_SERVICE_URL}"
        print(f"[ERROR] {error_msg}")
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
    except Exception as e:
        print(f"[ERROR] RAG Service streaming error: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


def query(
    query_str: str,
    method: str = "std",
    provider: str = "openai",
    model: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Generate a response using the specified method and provider.

    Args:
        query_str: The query text
        method: The method to use (std or cot)
        provider: The provider to use (openai, claude, or gemini)
        model: Optional specific model to use

    Returns:
        Tuple of (response_text, actual_provider, actual_model)
    """
    print(f"[DEBUG] Query: {query_str[:100]}... | Method: {method} | Provider: {provider}")

    actual_provider = provider

    if model is None:
        if provider.lower() == "openai":
            model = "gpt-4o-mini"
        elif provider.lower() == "claude":
            model = "claude-3-opus-20240229"
        elif provider.lower() == "gemini":
            model = "gemini-1.5-pro-latest"

    actual_model = model

    # Build prompt based on method
    if method == "cot":
        prompt = f"""Think step by step to answer the following question.

Question: {query_str}

Reasoning:"""
        system_prompt = "You are a helpful assistant that reasons step by step before answering."
    else:
        # Standard direct query
        prompt = f"Question: {query_str}"
        system_prompt = None

    try:
        if provider.lower() == "openai":
            response = query_openai(prompt, system_prompt, model)
        elif provider.lower() == "claude":
            response = query_claude(prompt, system_prompt, model)
        elif provider.lower() == "gemini":
            try:
                response = query_gemini(prompt, system_prompt, model)
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "429" in error_str:
                    print(f"[WARNING] Gemini quota exceeded, falling back to OpenAI")
                    actual_provider = "openai (fallback from gemini)"
                    actual_model = "gpt-3.5-turbo"
                    response = query_openai(prompt, system_prompt, "gpt-3.5-turbo")
                else:
                    raise
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return response, actual_provider, actual_model
    except Exception as e:
        print(f"[ERROR] Error generating response: {str(e)}")
        raise


DEFAULT_PROVIDERS = {
    "openai": {
        "default_model": "gpt-4o-mini",
        "available_models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    },
    "claude": {
        "default_model": "claude-3-opus-20240229",
        "available_models": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
    },
    "gemini": {
        "default_model": "gemini-1.5-pro-latest",
        "available_models": [
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
        ],
    },
    "rag": {
        "default_model": "rag-agent",
        "available_models": ["rag-agent"],
        "description": "Agentic RAG pipeline with multi-step reasoning",
    },
}


def get_available_providers():
    return DEFAULT_PROVIDERS
