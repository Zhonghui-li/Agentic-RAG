# agents/multi_query.py
"""Generate query variants and decomposed sub-queries via LLM for multi-query retrieval."""

from typing import List

_DECOMPOSE_PROMPT = (
    "You are an expert at breaking down complex multi-hop questions into complementary sub-queries.\n"
    "Given the question below, generate {n} COMPLEMENTARY sub-queries that together cover all the "
    "information needed to answer it. Each sub-query should target a DIFFERENT aspect or entity "
    "(e.g. one for a person, one for a location, one for a time period, one for a relationship).\n"
    "Do NOT rephrase the same question. Each sub-query must retrieve DIFFERENT evidence.\n"
    "Return ONLY the sub-queries, one per line.\n\n"
    "Question: {query}\n\n"
    "Sub-queries:"
)

_PROMPT_TEMPLATE = (
    "You are a helpful assistant that generates alternative search queries.\n"
    "Given the original question below, generate {n} alternative versions that "
    "capture the same information need but use different wording, emphasis, or "
    "decomposition. Return ONLY the alternative queries, one per line.\n\n"
    "Original question: {query}\n\n"
    "Alternative queries:"
)


def generate_query_variants(
    query: str,
    llm,
    n_variants: int = 2,
) -> List[str]:
    """
    Use *llm* to produce ``n_variants`` alternative phrasings of *query*.

    Returns:
        A list starting with the **original query**, followed by up to
        ``n_variants`` generated alternatives (duplicates / blanks removed).
    """
    prompt = _PROMPT_TEMPLATE.format(n=n_variants, query=query)

    try:
        response = llm.invoke(prompt)
        # LangChain ChatModel returns AIMessage; plain LM returns str
        text = getattr(response, "content", None) or str(response)
    except Exception as e:
        print(f"[MultiQuery] LLM call failed ({e}); falling back to original query only.")
        return [query]

    # Parse lines — skip blanks, numbering prefixes like "1." or "- "
    variants: List[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        # strip leading numbering / bullets
        for prefix in ("1.", "2.", "3.", "4.", "-", "*"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line and line != query:
            variants.append(line)

    # Dedupe while preserving order
    seen = {query}
    unique: List[str] = []
    for v in variants[:n_variants]:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    result = [query] + unique
    print(f"[MultiQuery] {len(result)} queries: {result}")
    return result


def decompose_query(
    query: str,
    llm,
    n_subqueries: int = 5,
) -> List[str]:
    """
    Decompose a complex multi-hop question into complementary sub-queries (PAR2-RAG Stage 1).

    Unlike generate_query_variants() which rephrases, this function decomposes the question
    into sub-queries targeting DIFFERENT aspects/entities, maximising coverage breadth.

    Returns:
        A list of sub-queries (original query NOT prepended — these are decomposed aspects).
        Falls back to [query] on LLM failure.
    """
    prompt = _DECOMPOSE_PROMPT.format(n=n_subqueries, query=query)

    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
    except Exception as e:
        print(f"[Decompose] LLM call failed ({e}); falling back to original query only.")
        return [query]

    sub_queries: List[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        for prefix in ("1.", "2.", "3.", "4.", "5.", "-", "*"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line and line != query:
            sub_queries.append(line)

    # Dedupe, cap at n_subqueries
    seen: set = set()
    unique: List[str] = []
    for q in sub_queries[:n_subqueries]:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    result = unique if unique else [query]
    print(f"[Decompose] {len(result)} sub-queries: {result}")
    return result
