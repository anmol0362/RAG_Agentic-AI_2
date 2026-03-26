from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

faiss_path = os.path.join("faiss_store", "faiss.index")
meta_path = os.path.join("faiss_store", "metadata.pkl")

store = FaissVectorStore(
    persist_dir="faiss_store",
    deployment_name="text-embedding-3-small"
)

if os.path.exists(faiss_path) and os.path.exists(meta_path):
    print("⚡ FAISS already exists. Skipping rebuild.")
else:
    print("🚀 First time build...")
    docs = load_all_documents(str(BASE_DIR / "data"))
    store.build_from_documents(docs)
    print("✅ RAG built successfully.")