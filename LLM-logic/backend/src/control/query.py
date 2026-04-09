import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import Optional, List, Literal, Tuple
import openai
import anthropic
import google.generativeai as genai
import json
import subprocess
import time
import requests

# Lazy-loaded retriever to avoid heavy initialization on import
_retriever = None


def _init_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    ucsc_passages_df = pd.read_csv(
        "src/../dataset/passages.csv",
        index_col=0,
    )
    ucsc_passage_data_loader = DataFrameLoader(
        ucsc_passages_df, page_content_column="passage"
    )
    ucsc_passage_data = ucsc_passage_data_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = text_splitter.split_documents(ucsc_passage_data)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    _retriever = db.as_retriever()
    return _retriever

# Initialize API clients
openai_client = None
anthropic_client = None
genai_initialized = False


def format_docs(docs):
    """Format documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


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

    # Create messages array - only include user message
    messages = [{"role": "user", "content": prompt}]

    # Create request parameters
    request_params = {"model": model, "messages": messages, "max_tokens": 2000}

    # Add system as a top-level parameter if provided
    if system_prompt:
        request_params["system"] = system_prompt
        print(f"[DEBUG] Using system prompt with Claude: {system_prompt[:50]}...")

    response = client.messages.create(**request_params)

    elapsed_time = time.time() - start_time
    print(f"[DEBUG] Claude response received in {elapsed_time:.2f} seconds")

    return response.content[0].text


def query_rag_service(query_str: str, use_router: bool = True, history: list = []) -> str:
    """
    Query the RAG Service for agentic RAG responses.

    Args:
        query_str: The question to ask
        use_router: Whether to use LangGraph router (default True)
        history: Conversation history as list of {"role": "user"/"assistant", "content": "..."} dicts

    Returns:
        The answer from the RAG pipeline
    """
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
            timeout=120  # RAG pipeline may take longer
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
        print(f"[ERROR] Could not connect to RAG Service at {RAG_SERVICE_URL}")
        raise Exception(f"RAG Service unavailable. Please ensure it's running at {RAG_SERVICE_URL}")
    except requests.exceptions.Timeout:
        print(f"[ERROR] RAG Service request timed out")
        raise Exception("RAG Service request timed out")
    except Exception as e:
        print(f"[ERROR] RAG Service error: {str(e)}")
        raise


def query_gemini(prompt, system_prompt=None, model="gemini-1.5-pro-latest"):
    print(f"[DEBUG] Sending query to Gemini using model: {model}")
    start_time = time.time()

    init_genai()

    # Add "models/" prefix if it doesn't already have it
    if not model.startswith("models/"):
        model_name = f"models/{model}"
    else:
        model_name = model

    # Use one of the available models from the list
    if model_name == "models/gemini-pro":
        # Replace with an available model
        model_name = "models/gemini-1.5-pro-latest"
        print(f"[DEBUG] Replacing unsupported model with: {model_name}")

    try:
        genai_model = genai.GenerativeModel(model_name)

        if system_prompt:
            # Gemini doesn't directly support system prompts like the others,
            # so we include it as part of the user's message
            prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        response = genai_model.generate_content(prompt)

        elapsed_time = time.time() - start_time
        print(f"[DEBUG] Gemini response received in {elapsed_time:.2f} seconds")

        return response.text
    except Exception as e:
        error_str = str(e)
        print(f"[ERROR] Gemini error: {error_str}")

        print(
            "[DEBUG] Available Gemini models include: models/gemini-1.5-pro-latest, models/gemini-1.5-flash-latest"
        )
        # Re-raise the exception for the caller to handle
        raise


def proslm_query(query_str: str):
    retriever = _init_retriever()
    retrieved_docs = retriever.invoke(query_str)
    context = format_docs(retrieved_docs)
    client = get_openai_client()

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": f"""
                Given this query and context, return the query in concise first order logic (FOL). 
                
                Example: 
                Query: What classes does the boating center offer to community members?, 
                Context: "UC Santa Cruz Community Boating Center is a community-based boating program that connects students and community members to the Monterey Bay through recreational activities including sailing, rowing and kayaking.
                UC Santa Cruz Community Boating Center staff and veteran participants provide a familiarity with the equipment, skills, and water safety that comes from a lifetime of enjoying these sports.
                The Community Boating Center also connects people who are interested in these sports to one another, whether you are a beginner seeking to learn something new or a experienced sailor, rower, and/or kayaker looking to improve your skills and get time on the water. Whether you've never been on a boat before, or want to build on old skills, we have a class for you! From 2-week junior summer camps to quarterly physical education classes for UC Santa Cruz Students, you'll be cruising around Monterey Bay with ease.

                Our club exists to give community members, UC Santa Cruz students, faculty, and staff access to equipment that makes it possible to enjoy our beautiful Monterey Bay. The club provides members a unique opportunity to use any number of sailing and rowing vessels under the weekend supervision of the Boating Center dockmaster.
                Our boats:
                Sailing vessels for weekend use currently include RS Quests, Lasers, RS Visions, a SC27, a Santana 22, a J 24, an Olson 30, and an O'Day 34.
                The rowing fleet exists of eight Edon training shells, two MAAS AEROS 20s, two MAAS 24s, Bay 21, Pegasus, WinTech 30' double, and a custom-built wooden double. 
                Want to take a boat out? Make sure to pay your membership dues and boat fees online ahead of time. The boating center will no longer be accepting cash payments on-site.
                Sailing Hours:
                Friday 4:00-7:00 PM (during daylight savings)
                Saturday 12:00-5:00 PM
                Sunday 12:00-5:00 PM
                Members of the Community Boating Club have access to sailboats and rowing shells during club hours. Members must be approved for boat use by the boating directors before checking out a rowing shall or sailboat. Check our calendar for the next available checkout day. If you do not have prior sailing or rowing experience, we offer classes to build your confidence on the water! For class information, click here.
                Sailors and rowers are evaluated based on experience and allowed boat use accordingly.

                Boating Club Hours
                The Community Boating Center is open 2 days a week.
                Saturday: 12:00 PM - 5:00 PM
                Sunday: 12:00 PM - 5:00 PM
                .

                FOL: ClassesOfferedByBoating Club = Sailing ∧ Rowing ∧ Kayaking
                
                Prompt: 
                Query: {query_str}, 
                Context: {context}
                .
                
                FOL:
                """,
            }
        ],
    )

    fol = response.choices[0].message.content

    print(f"FOL: {fol}")

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": f"""
                Convert this first order logic query to a concise prolog database.  
                            
                Example:
                FOL: 
                ∀x (CanBeMemberOfBoatingClub(x) ↔ CommunityMember(x) ∨ UCSCStudent(x) ∨ UCSCFaculty(x) ∨ UCSCStaff(x))
                
                Prolog:
                % Facts
                community_member(community_member).
                ucsc_student(ucsc_student).
                ucsc_faculty(ucsc_faculty).
                ucsc_staff(ucsc_staff).

                % Rule encoding the equivalence
                can_be_member_of_boating_club(X) :-
                    community_member(X);
                    ucsc_student(X);
                    ucsc_faculty(X);
                    ucsc_staff(X).
                
                Prompt:
                FOL: 
                {fol}.
                
                Prolog:
                [output]
                
                Omit ```prolog ``` and any intro.
                """,
            }
        ],
    )

    prolog = response.choices[0].message.content

    with open("rules.pl", "w") as f:
        f.write(prolog)

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": f"""
                Here I have a prolog rule and query, return the prolog query to run.
                Store all valid results in a variable X. 
                
                Example:
                Query: Who can be a member of the Boating Club?,
                Rulebase:
                % Facts
                community_member(community_member).
                ucsc_student(ucsc_student).
                ucsc_faculty(ucsc_faculty).
                ucsc_staff(ucsc_staff).

                % Rule encoding the equivalence
                can_be_member_of_boating_club(X) :-
                    community_member(X);
                    ucsc_student(X);
                    ucsc_faculty(X);
                    ucsc_staff(X).
                
                Prolog Query:
                findall(Y, can_be_member_of_boating_club(Y), X), write(X), nl
                
                Query: 
                {query_str}, 
                Rulebase: 
                {prolog} 
                Only output the prolog query without an intro or '?- '. Format the query as such: Query, write(X), nl
                """,
            }
        ],
    )

    prolog_query = response.choices[0].message.content

    # File containing Prolog code
    prolog_file = "rules.pl"

    print(f"Prolog Query: {prolog_query}")

    # This assumes your .pl file defines a predicate called `query/0`
    result = subprocess.run(
        ["swipl", "-s", prolog_file, "-g", prolog_query, "-t", "halt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": f"Here is the prolog output and error pipes from a given query, if there was a prolog error, say there was an error, else, answer the query in natural language. Original query: {query_str}, prolog query: {prolog_query}, stdout: {result.stdout}, stderr: {result.stderr}.  If there is no errors don't mention it.",
            }
        ],
    )

    result = response.choices[0].message.content

    return result


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
        method: The method to use (std, rag, or cot)
        provider: The provider to use (openai, claude, or gemini)
        model: Optional specific model to use with the provider

    Returns:
        Tuple of (response_text, actual_provider, actual_model)
    """
    print(f"\n[DEBUG] ====== LLM REQUEST ======")
    print(f"[DEBUG] Query: {query_str[:100]}...")
    print(f"[DEBUG] Method: {method}")
    print(f"[DEBUG] Provider: {provider}")
    print(f"[DEBUG] Model: {model}")

    # Track the actual provider and model used (for fallback cases)
    actual_provider = provider

    # Handle RAG provider separately (it doesn't use standard LLM flow)
    if provider.lower() == "rag":
        print(f"[DEBUG] Using RAG Agent pipeline")
        try:
            response = query_rag_service(query_str, use_router=True)
            return response, "rag", "rag-agent"
        except Exception as e:
            print(f"[ERROR] RAG Service error: {str(e)}")
            raise

    # Set default models if not specified
    if model is None:
        if provider.lower() == "openai":
            model = "gpt-4o-mini"
        elif provider.lower() == "claude":
            model = "claude-3-opus-20240229"
        elif provider.lower() == "gemini":
            model = "gemini-1.5-pro-latest"

    actual_model = model

    # Methods that depend on the removed UCSC FAQ dataset are no longer available
    UNAVAILABLE_METHODS = {"rag", "cot", "proslm"}
    if method in UNAVAILABLE_METHODS:
        msg = f"The '{method}' method is not available. Please use 'std' (standard LLM) or select RAG Agent."
        print(f"[DEBUG] {msg}")
        return msg, provider, model or "N/A"

    # Get relevant documents for RAG and CoT methods
    context = ""
    if method in ["rag", "cot"]:
        print(f"[DEBUG] Retrieving documents for method: {method}")
        retriever = _init_retriever()
        retrieved_docs = retriever.invoke(query_str)
        context = format_docs(retrieved_docs)
        print(
            f"[DEBUG] Retrieved {len(retrieved_docs)} documents, total context length: {len(context)} chars"
        )

    # Build prompt based on method
    if method == "std":
        prompt = f"Question: {query_str}"
        system_prompt = None
        print(f"[DEBUG] Using standard prompt template")

    elif method == "rag":
        prompt = f"""
Answer the question based only on the following context:

{context}

Question: {query_str}
"""
        system_prompt = "You are a helpful assistant that provides accurate information based only on the context provided."
        print(f"[DEBUG] Using RAG prompt template")

    elif method == "cot":
        prompt = f"""
1. **Question**: {query_str}

2. **Retrieved Context**: 
{context}

3. **Chain of Thought Reasoning**:
    - **Step 1**: Locate data about the specific question from the provided context.
    - **Step 2**: Identify any details or comparisons relevant to the question.
    - **Step 3**: Summarize the data, averages, or ranges if applicable.
    - **Step 4**: Draw a conclusion based on the context to directly answer the question.

4. **Examples of Chain of Thought Reasoning**:
    - **Question**: Am I on the waitlist automatically?

        **Reasoning**: Based on the information from this link: https://admissions.ucsc.edu/resources-support/frequently-asked-questions, 
        placement on the UC Santa Cruz waitlist is not automatic. Applicants who are offered a spot on the waitlist must choose to opt-in if they wish to be 
        considered for admission should space become available. This opt-in process ensures that only those genuinely interested in attending are considered 
        for admission from the waitlist.
    
        **Answer**: Therefore, you are not placed on the waitlist automatically. If offered a spot, you must opt-in to be on the waitlist.

5. **Answer**: Based on the reasoning above, the answer is:
"""
        system_prompt = "You are a helpful assistant that reasons step by step to answer questions accurately based on the provided context."
        print(f"[DEBUG] Using Chain of Thought prompt template")

    else:
        # Default to standard if method is unknown
        prompt = query_str
        system_prompt = None
        print(f"[DEBUG] Using default prompt (direct query)")

    # Generate response based on provider
    try:
        response = ""

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
                    # Change the actual provider and model for the response
                    actual_provider = "openai (fallback from gemini)"
                    actual_model = "gpt-3.5-turbo"
                    # Use OpenAI as fallback
                    response = query_openai(prompt, system_prompt, "gpt-3.5-turbo")
                else:
                    # Re-raise other errors
                    raise
        else:
            error_msg = f"Unsupported provider: {provider}"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        print(f"[DEBUG] Response received from provider: {actual_provider}")
        print(f"[DEBUG] Response length: {len(response)} characters")
        print(f"[DEBUG] First 100 chars of response: {response[:100]}...")
        print(f"[DEBUG] ====== LLM RESPONSE COMPLETE ======\n")

        return response, actual_provider, actual_model
    except Exception as e:
        print(f"[ERROR] Error generating response: {str(e)}")
        raise


# Default configuration for providers and models
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
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    },
    "rag": {
        "default_model": "rag-agent",
        "available_models": ["rag-agent"],
        "description": "Agentic RAG pipeline with multi-step reasoning",
    },
}


def get_available_providers():
    """Return information about available providers and models."""
    return DEFAULT_PROVIDERS


def stream_rag_service(query_str: str, use_router: bool = True, history: list = []):
    """
    Stream responses from the RAG Service using Server-Sent Events.

    Args:
        query_str: The question to ask
        use_router: Whether to use LangGraph router (default True)
        history: Conversation history as list of {"role": "user"/"assistant", "content": "..."} dicts

    Yields:
        SSE formatted strings
    """
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
