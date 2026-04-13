import os
from typing import Dict, Any
from dotenv import load_dotenv
import langfuse

load_dotenv()
from src.Rag_pipeline.evaluation import evaluate_rag
from src.Rag_pipeline.observability import langfuse
from src.Retrieval.azure_retrieval import retrieve, build_context
from src.Rag_pipeline.prompts import SYSTEM_PROMPT, build_user_prompt
from src.Rag_pipeline.generator import generate_answer, CHAT_DEPLOYMENT, TEMPERATURE, MAX_TOKENS
from src.Rag_pipeline.observability import (
    print_pipeline_parameters,
    print_query_observability,
    print_retrieval_observability,
    print_context_observability,
)


FINAL_TOP_K = 8

# Retrieval params (must match retrieval file)
DOC_EMBEDDING_DEPLOYMENT = os.getenv("DOC_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
ROW_EMBEDDING_DEPLOYMENT = os.getenv("ROW_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
DOC_INDEX_NAME = "docs-index"
ROW_INDEX_NAME = "rows-index"
DOC_TOP_K = 5
ROW_TOP_K = 5

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")


def run_rag(query: str) -> Dict[str, Any]:
    trace_obj = None

    print_pipeline_parameters(
        final_top_k=FINAL_TOP_K,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        chat_deployment=CHAT_DEPLOYMENT,
        doc_embedding_deployment=DOC_EMBEDDING_DEPLOYMENT,
        row_embedding_deployment=ROW_EMBEDDING_DEPLOYMENT,
        doc_index_name=DOC_INDEX_NAME,
        row_index_name=ROW_INDEX_NAME,
        doc_top_k=DOC_TOP_K,
        row_top_k=ROW_TOP_K,
        search_endpoint=AZURE_SEARCH_ENDPOINT,
        openai_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    print_query_observability(query)

    results = retrieve(query=query, final_top_k=FINAL_TOP_K)
    print_retrieval_observability(results)

    context = build_context(results)
    print_context_observability(context)

    user_prompt = build_user_prompt(query, context)
    answer = generate_answer(SYSTEM_PROMPT, user_prompt)

    eval_result = evaluate_rag(query, context, answer, trace_obj=trace_obj)
    print("\n" + "=" * 110)
    print("EVALUATION RESULT")
    print("=" * 110)
    print(eval_result)

    try:
        langfuse.flush()
    except Exception as e:
        print(f"[Langfuse Flush Error] {e}")

    langfuse.flush()

    return {
        "query": query,
        "answer": answer,
        "context": context,
        "results": results,
        "evaluation": eval_result
    }

if __name__ == "__main__":
    print("\n" + "=" * 110)
    print("AVIATION RAG CHATBOT")
    print("Type 'exit' to stop")
    print("=" * 110)

    while True:
        try:
            query = input("\n🔍 Your question: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("\n[INFO] Goodbye!")
                break

            output = run_rag(query)

            print("\n" + "=" * 110)
            print("FINAL ANSWER")
            print("=" * 110)
            print(output["answer"])
            print("=" * 110)

        except KeyboardInterrupt:
            print("\n[INFO] Goodbye!")
            break