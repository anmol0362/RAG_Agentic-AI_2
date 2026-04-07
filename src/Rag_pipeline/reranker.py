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

# You can use gpt-4o or a cheaper model if you later deploy one
RERANK_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

RERANK_TOP_K = 5
MAX_CHARS_PER_CHUNK = 1200


# =========================================================
# CLIENT
# =========================================================
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)


# =========================================================
# HELPERS
# =========================================================

def _compact_candidate(r: Dict[str, Any], rank: int) -> str:
    text = (r.get("content") or "")[:MAX_CHARS_PER_CHUNK]
    meta = {
        "rank": rank,
        "id": r.get("id"),
        "source": r.get("source"),
        "page": r.get("page"),
        "table": r.get("table"),
        "chunk_type": r.get("chunk_type"),
        "retrieval_source": r.get("retrieval_source"),
        "azure_score": r.get("score"),
    }
    return f"CANDIDATE {rank}\nMETA: {json.dumps(meta, ensure_ascii=False)}\nTEXT:\n{text}\n"


def rerank_results(query: str, candidates: List[Dict[str, Any]], top_k: int = RERANK_TOP_K) -> List[Dict[str, Any]]:
    """
    Rerank retrieved candidates using Azure OpenAI.
    Returns the best top_k chunks.
    """

    if not candidates:
        return []

    trace("RERANK START", {
        "query": query,
        "num_candidates": len(candidates),
        "top_k": top_k,
        "rerank_model": RERANK_DEPLOYMENT
    })

    candidate_blocks = []
    for i, c in enumerate(candidates, start=1):
        candidate_blocks.append(_compact_candidate(c, i))

    candidates_text = "\n\n".join(candidate_blocks)

    system_prompt = """
You are a retrieval reranker for an aviation RAG system.

Your task:
- Rank the candidate chunks by how useful they are for answering the user's question.
- Prefer chunks that are directly relevant, specific, procedural, factual, and answer-bearing.
- Penalize vague, generic, repetitive, or weakly related chunks.
- Use the user's question as the only relevance target.

Return ONLY valid JSON in this exact format:
{
  "ranked_candidates": [
    {"rank": 1, "candidate_number": 3, "score": 9.7, "reason": "highly relevant"},
    {"rank": 2, "candidate_number": 1, "score": 9.2, "reason": "direct procedure"},
    {"rank": 3, "candidate_number": 7, "score": 8.9, "reason": "supporting evidence"}
  ]
}

Rules:
- candidate_number refers to the numbered CANDIDATE blocks.
- Higher score = more relevant.
- Rank best first.
- Return up to the top 8 best candidates.
- Output JSON only, no markdown.
"""

    user_prompt = f"""
USER QUESTION:
{query}

CANDIDATE CHUNKS:
{candidates_text}
""".strip()

    response = client.chat.completions.create(
        model=RERANK_DEPLOYMENT,
        temperature=0,
        max_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content

    trace("RERANK RAW RESPONSE", raw)

    try:
        parsed = json.loads(raw)
        ranked = parsed.get("ranked_candidates", [])
    except Exception as e:
        trace("RERANK PARSE ERROR", str(e))
        return candidates[:top_k]

    selected = []
    seen = set()

    for item in ranked:
        idx = item.get("candidate_number")
        if not isinstance(idx, int):
            continue
        if idx < 1 or idx > len(candidates):
            continue
        if idx in seen:
            continue

        seen.add(idx)
        chosen = dict(candidates[idx - 1])
        chosen["rerank_score"] = item.get("score")
        chosen["rerank_reason"] = item.get("reason")
        chosen["rerank_rank"] = item.get("rank")
        selected.append(chosen)

        if len(selected) >= top_k:
            break

    # Fallback if model returned too few
    if len(selected) < top_k:
        for i, c in enumerate(candidates, start=1):
            if i not in seen:
                fallback = dict(c)
                fallback["rerank_score"] = None
                fallback["rerank_reason"] = "fallback_from_retrieval_order"
                fallback["rerank_rank"] = len(selected) + 1
                selected.append(fallback)
                if len(selected) >= top_k:
                    break

    trace("RERANK FINAL SELECTION", [
        {
            "rerank_rank": r.get("rerank_rank"),
            "id": r.get("id"),
            "source": r.get("source"),
            "azure_score": r.get("score"),
            "rerank_score": r.get("rerank_score"),
            "rerank_reason": r.get("rerank_reason"),
            "retrieval_source": r.get("retrieval_source")
        }
        for r in selected
    ])

    return selected