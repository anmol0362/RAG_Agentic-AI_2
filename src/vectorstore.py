import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from dotenv import load_dotenv
from src.embedding import EmbeddingPipeline

load_dotenv()


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        azure_endpoint: str = None,
        api_key: str = None,
        deployment_name: str = "text-embedding-3-large",
        chunk_size: int = 800,     # 🔥 updated
        chunk_overlap: int = 100
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")

        if not self.azure_endpoint or not self.api_key:
            raise ValueError("❌ Missing Azure credentials in .env")

        self.azure_endpoint = self.azure_endpoint.strip().rstrip("/")
        print("DEBUG ENDPOINT:", self.azure_endpoint)

        self.emb_pipe = EmbeddingPipeline(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            deployment_name=deployment_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # 🔹 Build
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents...")

        chunks = self.emb_pipe.chunk_documents(documents)
        embeddings = self.emb_pipe.embed_chunks(chunks)

        metadatas = [{"text": chunk.page_content} for chunk in chunks]

        self.add_embeddings(embeddings.astype("float32"), metadatas)
        self.save()

    # 🔹 Add embeddings (FIXED)
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]):
        dim = embeddings.shape[1]

        # 🔥 recreate index if mismatch
        if self.index is None or self.index.d != dim:
            print("[INFO] Creating new FAISS index...")
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors")

    # 🔹 Save
    def save(self):
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))

        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

        print("✅ [INFO] Saved FAISS index")

    # 🔹 Load
    def load(self):
        self.index = faiss.read_index(os.path.join(self.persist_dir, "faiss.index"))

        with open(os.path.join(self.persist_dir, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)

        print("⚡ [INFO] Loaded existing FAISS index")

    # 🔹 Search (FIXED)
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = query_embedding.astype("float32")

        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                f"❌ Dimension mismatch: query={query_embedding.shape[1]} vs index={self.index.d}"
            )

        D, I = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"distance": float(dist), "metadata": meta})

        return results

    # 🔹 Query
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Query: {query_text}")

        response = self.emb_pipe.client.embeddings.create(
            input=query_text,
            model=self.emb_pipe.deployment_name
        )

        query_embedding = np.array([response.data[0].embedding]).astype("float32")

        return self.search(query_embedding, top_k)


# 🚀 RUN
if __name__ == "__main__":
    from src.Data_Ingestion.data_loader import load_all_documents
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    faiss_path = os.path.join("faiss_store", "faiss.index")
    meta_path = os.path.join("faiss_store", "metadata.pkl")

    store = FaissVectorStore(
        persist_dir="faiss_store",
        deployment_name="text-embedding-3-large"
    )

    # 🔥 IMPORTANT FIX (NO REBUILD LOOP)
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        print("⚡ Loading existing vector DB...")
        store.load()
    else:
        print("🚀 First time build...")
        docs = load_all_documents(str(BASE_DIR / "data"))
        store.build_from_documents(docs)

    results = store.query("What is attention mechanism?", top_k=3)

    print("\n🔍 Results:")
    for r in results:
        print("-", r["metadata"]["text"][:200])