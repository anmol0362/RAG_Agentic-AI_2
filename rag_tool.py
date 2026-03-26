from src.vectorstore import FaissVectorStore

store = FaissVectorStore(
    persist_dir="faiss_store",
    deployment_name="text-embedding-3-small"
)

store.load()


def rag_search(query: str, top_k: int = 3):
    results = store.query(query, top_k=top_k)

    return [
        {
            "text": r["metadata"]["text"],
            "distance": r["distance"]
        }
        for r in results
    ]