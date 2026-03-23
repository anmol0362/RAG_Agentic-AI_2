from typing import List, Any
import numpy as np
import time
import os
from openai import AzureOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingPipeline:
  def __init__(
    self,
    azure_endpoint: str,
    api_key: str,
    deployment_name: str = "text-embedding-3-small",
    chunk_size: int = 800,        # 🔥 bigger chunks
    chunk_overlap: int = 100,
    batch_size: int = 7           # 🔥 safe batch
  ):
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.deployment_name = deployment_name
    self.batch_size = batch_size

    # ✅ clean endpoint
    azure_endpoint = azure_endpoint.strip().rstrip("/")

    self.client = AzureOpenAI(
      api_key=api_key,
      api_version="2024-02-01",
      azure_endpoint=azure_endpoint
    )

    print(f"[INFO] Using embedding model: {deployment_name}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Chunk size: {chunk_size}")

  # 🔹 Step 1: Chunk documents
  def chunk_documents(self, documents: List[Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
      chunk_size=self.chunk_size,
      chunk_overlap=self.chunk_overlap,
      length_function=len,
      separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"[INFO] Documents: {len(documents)}")
    print(f"[INFO] Chunks created: {len(chunks)}")

    return chunks

  # 🔹 Step 2: Embed with retry (VERY IMPORTANT)
  def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
    texts = [chunk.page_content for chunk in chunks]

    print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

    all_embeddings = []

    for i in range(0, len(texts), self.batch_size):
      batch = texts[i:i + self.batch_size]

      success = False

      for attempt in range(5):   # 🔥 retry logic
        try:
          response = self.client.embeddings.create(
            input=batch,
            model=self.deployment_name
          )

          batch_embeddings = [item.embedding for item in response.data]
          all_embeddings.extend(batch_embeddings)

          print(f"[INFO] Batch {i // self.batch_size + 1} done")

          success = True
          break

        except Exception as e:
          print(f"[RETRY {attempt+1}] batch {i // self.batch_size + 1} failed: {e}")
          time.sleep(5)

      if not success:
        print(f"[ERROR] Skipping batch {i // self.batch_size + 1}")

    embeddings_array = np.array(all_embeddings, dtype="float32")

    print(f"[INFO] Final embeddings shape: {embeddings_array.shape}")

    return embeddings_array