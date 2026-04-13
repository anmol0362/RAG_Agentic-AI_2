import os
from typing import List, Dict, Any

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from src.Rag_pipeline.observability import trace
from src.Rag_pipeline.reranker import rerank_results
from src.Rag_pipeline.query_optimizer import build_retrieval_queries

# -------------------------
# CONFIG
# -------------------------
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

DOC_EMBEDDING_DEPLOYMENT = os.getenv("DOC_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

AZURE_SEARCH_ENDPOINT  = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

DOC_INDEX_NAME      = "docs-index"
DOC_VECTOR_FIELD    = "contentVector"
DOC_TOP_K_PER_QUERY = 5
FINAL_TOP_K         = 5

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
# PARENT FETCHER
# After retrieving child chunks, fetch their parent
# for full context to send to LLM
# =========================================================

def fetch_parent_text(parent_id: str, search_client: SearchClient) -> str:
    """
    Fetch parent chunk text from Azure Search by parent_id.
    Child chunks are used for retrieval (small, precise).
    Parent chunks are used for LLM context (large, complete).
    """
    if not parent_id:
        return ""
    try:
        results = search_client.search(
            search_text="",
            filter=f"id eq '{parent_id}'",
            select=["content", "id", "section", "page"],
            top=1
        )
        for r in results:
            return r.get("content", "")
    except Exception as e:
        print(f"[WARN] Could not fetch parent {parent_id}: {e}")
    return ""


# =========================================================
# SEARCH HELPER
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
            "chunk_level":      r.get("chunk_level", "single"),  # NEW
            "parent_id":        r.get("parent_id"),              # NEW
            "chunk_size":       r.get("chunk_size"),
            "content_type":     r.get("content_type", "text"),
            "section":          r.get("section", ""),
        }

        if "page" in r:
            item["page"] = r.get("page")

        output.append(item)

    return output


# =========================================================
# DOC SEARCH
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
            "file_type", "chunk_type", "chunk_level",  # NEW
            "parent_id", "page", "chunk_size",          # NEW
            "content_type", "section"
        ]
    )

    trace("DOC RESULTS", [
        {
            "id":          x["id"],
            "score":       x["score"],
            "chunk_level": x.get("chunk_level"),
            "parent_id":   x.get("parent_id"),
            "source":      x.get("source"),
            "page":        x.get("page"),
            "section":     x.get("section"),
        }
        for x in results
    ])

    return results


# =========================================================
# DEDUPLICATION
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
        "after":  len(deduped)
    })

    return deduped


# =========================================================
# MAIN RETRIEVAL
# =========================================================

def retrieve(query: str, final_top_k: int = FINAL_TOP_K) -> List[Dict[str, Any]]:
    trace("HYBRID RETRIEVAL START", {
        "query":    query,
        "strategy": "query optimization + hybrid retrieval + parent-child + reranking",
        "doc_top_k_per_query":      DOC_TOP_K_PER_QUERY,
        "final_top_k_after_rerank": final_top_k
    })

    # Step 1: Generate multiple query variants
    retrieval_queries = build_retrieval_queries(query)

    # Step 2: Search child chunks for each query variant
    all_doc_results = []
    for rq in retrieval_queries:
        all_doc_results.extend(search_docs(rq))

    # Step 3: Deduplicate
    candidates = deduplicate_results(all_doc_results)

    # Step 4: Rerank
    final_results = rerank_results(
        query=query,
        candidates=candidates,
        top_k=final_top_k
    )

    # =========================================================
    # Step 5: PARENT FETCH
    # For every child chunk retrieved → fetch its parent
    # Replace child content with parent content for LLM
    # This gives LLM full context instead of tiny fragments
    # =========================================================
    doc_search_client = get_search_client(DOC_INDEX_NAME)
    seen_parents = set()

    for result in final_results:
        chunk_level = result.get("chunk_level", "single")
        parent_id   = result.get("parent_id")

        if chunk_level == "child" and parent_id:
            if parent_id not in seen_parents:
                parent_text = fetch_parent_text(parent_id, doc_search_client)
                if parent_text:
                    result["child_content"] = result["content"]  # keep child for reference
                    result["content"]       = parent_text         # replace with parent
                    seen_parents.add(parent_id)
                    print(f"[PARENT FETCH] ✅ {parent_id[:60]}...")
                else:
                    print(f"[PARENT FETCH] ⚠️  Parent not found: {parent_id[:60]}...")
            else:
                # Already fetched this parent — deduplicate
                result["content"] = ""

    # Remove results where parent was already used
    final_results = [r for r in final_results if r.get("content")]

    trace("HYBRID RETRIEVAL COMPLETE", {
        "final_results_count": len(final_results),
        "reranking_used":           True,
        "parent_child_used":        True,
        "query_optimization_used":  True
    })

    return final_results


# =========================================================
# CONTEXT BUILDER
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
            meta.append(f"content_type={r['content_type']}")
        if r.get("section"):
            meta.append(f"section={r['section']}")
        if r.get("chunk_level"):
            meta.append(f"level={r['chunk_level']}")
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
        "context_preview":      context[:2000]
    })

    return context


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    test_query = "What caused the accident?"
    results    = retrieve(test_query)
    context    = build_context(results)
    print("\n" + "=" * 100)
    print("FINAL RETRIEVED CONTEXT")
    print("=" * 100)
    print(context)