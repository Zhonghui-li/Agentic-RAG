# agents/esc.py
"""
Evidence Sufficiency Controller (ESC) — PAR2-RAG Stage 2.

Given the original question and currently retrieved documents, decides:
  - STOP  : evidence is sufficient to generate a final answer
  - CONTINUE + follow-up query : evidence has gaps, retrieve more
"""

from typing import Tuple, List

_ESC_PROMPT = """You are evaluating whether retrieved evidence is sufficient to answer a multi-hop question.

Question: {question}

Retrieved Evidence (summary):
{context_summary}

Task:
1. Determine if the evidence above is sufficient to fully answer the question.
2. If YES → respond with exactly: STOP
3. If NO → respond with exactly:
   CONTINUE
   <one focused follow-up query to fill the most important missing gap>

Rules:
- STOP only if all key facts needed to answer are present in the evidence.
- CONTINUE query must target a SPECIFIC missing fact, not re-ask the original question.
- Be concise. Do not explain your reasoning.

Response:"""

_CONTEXT_SUMMARY_MAX_CHARS = 3000  # truncate context sent to ESC to save tokens


def _summarise_docs(docs: List) -> str:
    """Extract text from docs and truncate for ESC prompt."""
    parts = []
    for i, d in enumerate(docs):
        text = getattr(d, "page_content", None) or (d.get("page_content", "") if isinstance(d, dict) else str(d))
        parts.append(f"[Doc {i+1}] {text[:400]}")
    combined = "\n".join(parts)
    return combined[:_CONTEXT_SUMMARY_MAX_CHARS]


class EvidenceSufficiencyController:
    """
    Lightweight ESC: prompt-based STOP / CONTINUE decision using an LLM.
    Intentionally kept stateless — call check() at each refinement hop.
    """

    def __init__(self, llm, max_hops: int = 4):
        """
        Args:
            llm: any LangChain-compatible LLM (ChatOpenAI, etc.)
            max_hops: hard ceiling on refinement hops regardless of ESC decision
        """
        self.llm = llm
        self.max_hops = max_hops

    def check(
        self,
        question: str,
        docs: List,
        current_hop: int,
    ) -> Tuple[str, str]:
        """
        Decide whether to stop or continue retrieval.

        Args:
            question: original user question
            docs: currently accumulated documents (C_t)
            current_hop: 0-indexed hop counter (Stage 2)

        Returns:
            (action, follow_up_query)
            action ∈ {"STOP", "CONTINUE"}
            follow_up_query: non-empty only when action == "CONTINUE"
        """
        # Hard ceiling — always stop at max_hops
        if current_hop >= self.max_hops:
            print(f"[ESC] hop={current_hop} >= max_hops={self.max_hops} → STOP (budget)")
            return "STOP", ""

        context_summary = _summarise_docs(docs)
        prompt = _ESC_PROMPT.format(
            question=question,
            context_summary=context_summary,
        )

        try:
            response = self.llm.invoke(prompt)
            text = (getattr(response, "content", None) or str(response)).strip()
        except Exception as e:
            print(f"[ESC] LLM call failed ({e}) → STOP (fallback)")
            return "STOP", ""

        lines = [l.strip() for l in text.splitlines() if l.strip()]

        if not lines:
            return "STOP", ""

        if lines[0].upper() == "STOP":
            print(f"[ESC] hop={current_hop} → STOP")
            return "STOP", ""

        if lines[0].upper() == "CONTINUE":
            follow_up = lines[1] if len(lines) > 1 else ""
            if not follow_up:
                print(f"[ESC] hop={current_hop} → CONTINUE but no follow-up query → STOP (fallback)")
                return "STOP", ""
            print(f"[ESC] hop={current_hop} → CONTINUE | follow-up: {follow_up}")
            return "CONTINUE", follow_up

        # Unexpected format — default to STOP
        print(f"[ESC] unexpected response: '{text[:80]}' → STOP (fallback)")
        return "STOP", ""
