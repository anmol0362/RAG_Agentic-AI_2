import json
from typing import Any, Dict, List
import os
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()

TRACE = True


def trace(title: str, obj: Any = None):
    if not TRACE:
        return

    print("\n" + "=" * 110)
    print(f"[TRACE] {title}")
    print("=" * 110)

    if obj is not None:
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=2, ensure_ascii=False)[:10000])
        else:
            print(obj)


def print_pipeline_parameters(
    final_top_k: int,
    temperature: float,
    max_tokens: int,
    chat_deployment: str,
    doc_embedding_deployment: str,
    row_embedding_deployment: str,
    doc_index_name: str,
    row_index_name: str,
    doc_top_k: int,
    row_top_k: int,
    search_endpoint: str,
    openai_endpoint: str,
):
    trace("PIPELINE PARAMETERS", {
        "RAG_PARAMS": {
            "FINAL_TOP_K": final_top_k,
            "TEMPERATURE": temperature,
            "MAX_TOKENS": max_tokens,
            "CHAT_DEPLOYMENT": chat_deployment,
        },
        "RETRIEVAL_PARAMS": {
            "DOC_EMBEDDING_DEPLOYMENT": doc_embedding_deployment,
            "ROW_EMBEDDING_DEPLOYMENT": row_embedding_deployment,
            "DOC_INDEX_NAME": doc_index_name,
            "ROW_INDEX_NAME": row_index_name,
            "DOC_TOP_K": doc_top_k,
            "ROW_TOP_K": row_top_k,
        },
        "ENDPOINTS": {
            "AZURE_SEARCH_ENDPOINT": search_endpoint,
            "AZURE_OPENAI_ENDPOINT": openai_endpoint,
        }
    })


def print_query_observability(query: str):
    trace("QUERY OBSERVABILITY", {
        "query": query,
        "query_length_chars": len(query),
        "query_word_count": len(query.split())
    })


def print_retrieval_observability(results: List[Dict[str, Any]]):
    trace("RETRIEVAL OBSERVABILITY", [
        {
            "rank": i + 1,
            "id": r.get("id"),
            "score": r.get("score"),
            "source": r.get("source"),
            "page": r.get("page"),
            "table": r.get("table"),
            "chunk_type": r.get("chunk_type"),
            "retrieval_source": r.get("retrieval_source"),
            "chunk_size": r.get("chunk_size")
        }
        for i, r in enumerate(results)
    ])


def print_context_observability(context: str):
    trace("CONTEXT OBSERVABILITY", {
        "context_length_chars": len(context),
        "context_word_count": len(context.split()),
        "context_preview": context[:3000]
    })


def print_prompt_observability(system_prompt: str, user_prompt: str):
    trace("PROMPT OBSERVABILITY", {
        "system_prompt_length_chars": len(system_prompt),
        "user_prompt_length_chars": len(user_prompt),
        "system_prompt_preview": system_prompt[:1500],
        "user_prompt_preview": user_prompt[:3000]
    })


def print_llm_usage_observability(
    model: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int
):
    trace("LLM USAGE OBSERVABILITY", {
        "model": model,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    })


def print_answer_observability(answer: str):
    trace("ANSWER OBSERVABILITY", {
        "answer_length_chars": len(answer),
        "answer_word_count": len(answer.split()),
        "answer_preview": answer[:3000]
    })
