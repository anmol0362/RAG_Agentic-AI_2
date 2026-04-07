import os
from typing import List, Dict, Any

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from src.Rag_pipeline.observability import trace
from src.Rag_pipeline.reranker import rerank_results
from src.Rag_pipeline.query_optimizer import build_retrieval_queries


# =========================================================
# WHY THIS FILE EXISTS:
# At query time, this file:
#   1. Takes the user's question
#   2. Optimizes it into multiple search queries
#   3. Embeds each query into a vector
#   4. Searches Azure AI Search using hybrid search
#      (vector similarity + keyword search combined)
#   5. Deduplicates results
#   6. Reranks using GPT-4o
#   7. Returns top chunks for the LLM context
#
# NEW: select_fields now includes content_type and section
# so retrieval results carry this info into the context.
# =========================================================

# -------------------------
# CONFIG
# -------------------------
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

DOC_EMBEDDING_DEPLOYMENT = os.getenv("DOC_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

AZURE_SEARCH_ENDPOINT  = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

DOC_INDEX_NAME    = "docs-index"
DOC_VECTOR_FIELD  = "contentVector"
DOC_TOP_K_PER_QUERY = 5
FINAL_TOP_K = 5


# -------------------------
# CLIENTS
# -------------------------
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)


def get_search_client(index_name: str) -> SearchClient:
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    )


# =========================================================
# QUERY EMBEDDING
# Converts the user's question into a 3072-dim vector
# so it can be compared against stored chunk vectors.
# =========================================================

def embed_query(query: str, deployment_name: str) -> List[float]:
    trace("Embedding Query", {
        "deployment_name": deployment_name,
        "query": query,
        "query_length_chars": len(query),
        "query_word_count": len(query.split())
    })

    response = openai_client.embeddings.create(
        model=deployment_name,
        input=[query]
    )

    embedding = response.data[0].embedding

    trace("Embedding Generated", {
        "deployment_name": deployment_name,
        "embedding_dimensions": len(embedding),
        "embedding_preview_first_8": embedding[:8]
    })

    return embedding


# =========================================================
# SEARCH HELPER
# Performs hybrid search = vector search + keyword search
# combined by Azure Search automatically.
# Vector search finds semantically similar chunks.
# Keyword search finds exact term matches.
# Combined = better recall than either alone.
# =========================================================

def _search_index(
    query: str,
    index_name: str,
    embedding_deployment: str,
    vector_field: str,
    top_k: int,
    select_fields: List[str],
    retrieval_source: str
) -> List[Dict[str, Any]]:

    client = get_search_client(index_name)
    query_vector = embed_query(query, embedding_deployment)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields=vector_field
    )

    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=top_k,
        select=select_fields
    )

    output = []
    for r in results:
        item = {
            "retrieval_source": retrieval_source,
            "retrieval_query":  query,
            "score":            float(r["@search.score"]),
            "id":               r.get("id"),
            "content":          r.get("content"),
            "doc_id":           r.get("doc_id"),
            "source":           r.get("source"),
            "file_type":        r.get("file_type"),
            "chunk_type":       r.get("chunk_type"),
            "chunk_size":       r.get("chunk_size"),
            "content_type":     r.get("content_type", "text"),  # NEW
            "section":          r.get("section", ""),           # NEW
        }

        if "page" in r:
            item["page"] = r.get("page")

        output.append(item)

    return output


# =========================================================
# DOC SEARCH
# Searches the docs-index for relevant text and table chunks.
# =========================================================

def search_docs(query: str) -> List[Dict[str, Any]]:
    trace("SEARCH DOCS", {
        "query": query,
        "index": DOC_INDEX_NAME,
        "top_k": DOC_TOP_K_PER_QUERY
    })

    results = _search_index(
        query=query,
        index_name=DOC_INDEX_NAME,
        embedding_deployment=DOC_EMBEDDING_DEPLOYMENT,
        vector_field=DOC_VECTOR_FIELD,
        top_k=DOC_TOP_K_PER_QUERY,
        retrieval_source="docs-index",
        select_fields=[
            "id", "content", "doc_id", "source",
            "file_type", "chunk_type", "page", "chunk_size",
            "content_type", "section"   # NEW fields
        ]
    )

    trace("DOC RESULTS", [
        {
            "id":           x["id"],
            "score":        x["score"],
            "source":       x.get("source"),
            "page":         x.get("page"),
            "content_type": x.get("content_type"),  # NEW
            "section":      x.get("section"),        # NEW
            "query_used":   x.get("retrieval_query")
        }
        for x in results
    ])

    return results


# =========================================================
# DEDUPLICATION
# When we run 5 different query variants, the same chunk
# might come back multiple times. We keep only the best
# scoring version of each unique chunk id.
# =========================================================

def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_id = {}

    for r in results:
        rid = r["id"]
        if rid not in best_by_id or r["score"] > best_by_id[rid]["score"]:
            best_by_id[rid] = r

    deduped = sorted(best_by_id.values(), key=lambda x: x["score"], reverse=True)

    trace("DEDUPLICATED RESULTS", {
        "before": len(results),
        "after": len(deduped)
    })

    return deduped


# =========================================================
# MAIN RETRIEVAL
# Full pipeline: query optimization → search → dedup → rerank
# =========================================================

def retrieve(query: str, final_top_k: int = FINAL_TOP_K) -> List[Dict[str, Any]]:
    trace("HYBRID RETRIEVAL START", {
        "query": query,
        "strategy": "query optimization + hybrid retrieval + reranking",
        "doc_top_k_per_query": DOC_TOP_K_PER_QUERY,
        "final_top_k_after_rerank": final_top_k
    })

    # Step 1: Generate multiple query variants
    retrieval_queries = build_retrieval_queries(query)

    # Step 2: Search for each query variant
    all_doc_results = []
    for rq in retrieval_queries:
        all_doc_results.extend(search_docs(rq))

    trace("RETRIEVAL BEFORE DEDUP", {
        "doc_results_total": len(all_doc_results)
    })

    # Step 3: Deduplicate
    candidates = deduplicate_results(all_doc_results)

    trace("CANDIDATES PRE-RERANK", [
        {
            "id":           r["id"],
            "score":        r["score"],
            "content_type": r.get("content_type"),
            "section":      r.get("section"),
            "source":       r.get("source")
        }
        for r in candidates[:25]
    ])

    # Step 4: Rerank using GPT-4o
    final_results = rerank_results(
        query=query,
        candidates=candidates,
        top_k=final_top_k
    )

    trace("HYBRID RETRIEVAL COMPLETE", {
        "final_results_count": len(final_results),
        "reranking_used": True,
        "query_optimization_used": True
    })

    return final_results


# =========================================================
# CONTEXT BUILDER
# Formats retrieved chunks into a readable context string
# that gets passed to the LLM.
# NEW: includes content_type and section in metadata line.
# =========================================================

def build_context(results: List[Dict[str, Any]]) -> str:
    trace("BUILD CONTEXT START", {"num_results": len(results)})

    context_parts = []

    for i, r in enumerate(results, start=1):
        meta = []

        if r.get("source"):
            meta.append(f"source={r['source']}")
        if r.get("page") is not None:
            meta.append(f"page={r['page']}")
        if r.get("content_type"):
            meta.append(f"content_type={r['content_type']}")   # NEW
        if r.get("section"):
            meta.append(f"section={r['section']}")             # NEW
        if r.get("chunk_type"):
            meta.append(f"type={r['chunk_type']}")
        if r.get("rerank_score") is not None:
            meta.append(f"rerank_score={r['rerank_score']}")

        meta_str = " | ".join(meta)

        context_parts.append(
            f"[Chunk {i}]\n"
            f"{meta_str}\n"
            f"{r.get('content', '')}\n"
        )

    context = "\n".join(context_parts)

    trace("FINAL CONTEXT BUILT", {
        "context_length_chars": len(context),
        "context_preview": context[:2000]
    })

    return context


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    test_query = "What caused the engine fire and what should the pilot do during takeoff?"
    results = retrieve(test_query)
    context = build_context(results)
    print("\n" + "=" * 100)
    print("FINAL RETRIEVED CONTEXT")
    print("=" * 100)
    print(context)