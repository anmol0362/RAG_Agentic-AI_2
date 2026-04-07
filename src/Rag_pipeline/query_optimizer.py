import os
import json
from typing import List, Dict, Any

from openai import AzureOpenAI
from src.Rag_pipeline.observability import trace


# =========================================================
# CONFIG
# =========================================================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

QUERY_OPTIMIZER_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")


# =========================================================
# CLIENT
# =========================================================
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)


# =========================================================
# MAIN OPTIMIZER
# =========================================================

def optimize_query(query: str) -> Dict[str, Any]:
    """
    Returns:
    {
      "original_query": ...,
      "expanded_queries": [...],
      "hyde_query": "...",
      "sub_queries": [...]
    }
    """

    trace("QUERY OPTIMIZATION START", {
        "original_query": query
    })

    system_prompt = """
You are a query optimizer for an aviation RAG system.

Your job:
1. Generate multiple retrieval-friendly rewrites of the user's question.
2. Generate a hypothetical ideal answer passage (HyDE style) that would likely appear in relevant documents.
3. If the question is complex, decompose it into smaller retrieval-friendly sub-questions.

Return ONLY valid JSON in this exact structure:
{
  "expanded_queries": [
    "query variant 1",
    "query variant 2",
    "query variant 3"
  ],
  "hyde_query": "a short hypothetical answer-like passage useful for retrieval",
  "sub_queries": [
    "sub-question 1",
    "sub-question 2"
  ]
}

Rules:
- Keep expanded queries concise and retrieval-friendly.
- HyDE should be factual/passsage-like, not too long.
- If decomposition is unnecessary, return an empty sub_queries list.
- Output JSON only.
"""

    user_prompt = f"""
USER QUERY:
{query}
""".strip()

    response = client.chat.completions.create(
        model=QUERY_OPTIMIZER_DEPLOYMENT,
        temperature=0.2,
        max_tokens=1000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content
    trace("QUERY OPTIMIZER RAW RESPONSE", raw)

    try:
        parsed = json.loads(raw)
    except Exception as e:
        trace("QUERY OPTIMIZER PARSE ERROR", str(e))
        parsed = {
            "expanded_queries": [],
            "hyde_query": "",
            "sub_queries": []
        }

    result = {
        "original_query": query,
        "expanded_queries": parsed.get("expanded_queries", []),
        "hyde_query": parsed.get("hyde_query", ""),
        "sub_queries": parsed.get("sub_queries", [])
    }

    trace("QUERY OPTIMIZATION RESULT", result)
    return result


def build_retrieval_queries(query: str) -> List[str]:
    """
    Final retrieval query set used for Azure Search.
    Deduplicated and ordered.
    """
    optimized = optimize_query(query)

    all_queries = [optimized["original_query"]]

    all_queries.extend(optimized.get("expanded_queries", []))

    if optimized.get("hyde_query"):
        all_queries.append(optimized["hyde_query"])

    all_queries.extend(optimized.get("sub_queries", []))

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for q in all_queries:
        q_clean = q.strip()
        if q_clean and q_clean.lower() not in seen:
            deduped.append(q_clean)
            seen.add(q_clean.lower())

    trace("FINAL RETRIEVAL QUERY SET", {
        "num_queries": len(deduped),
        "queries": deduped
    })

    return deduped