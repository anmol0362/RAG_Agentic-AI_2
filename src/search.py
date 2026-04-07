import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_deployment: str = "text-embedding-3-large",
        chat_deployment: str = "gpt-4o-mini"   # 🔥 better default
    ):
        # 🔐 Load env
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not self.azure_endpoint or not self.api_key:
            raise ValueError("❌ Missing Azure credentials in .env")

        self.azure_endpoint = self.azure_endpoint.strip().rstrip("/")

        # 🔹 Vector store
        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            deployment_name=embedding_deployment
        )

        # 🔹 Load only (NO rebuild here 🔥)
        print("⚡ Loading vector DB...")
        self.vectorstore.load()

        # 🔹 Chat client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-02-01",
            azure_endpoint=self.azure_endpoint
        )

        self.chat_deployment = chat_deployment

        print(f"[INFO] Chat model: {chat_deployment}")

    # 🔹 Search + summarize (IMPROVED)
    def search_and_summarize(self, query: str, top_k: int = 4) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        # 🔥 clean + limit context
        texts = []
        for r in results:
            if r["metadata"] and r["metadata"].get("text"):
                texts.append(r["metadata"]["text"][:500])  # 🔥 trim long chunks

        context = "\n\n".join(texts)

        if not context.strip():
            return "No relevant documents found."

        # 🔥 MUCH BETTER PROMPT
        prompt = f"""
You are a precise AI assistant.

Use ONLY the provided context to answer.

If the answer is not in the context, say:
"I could not find this in the provided data."

---

Question:
{query}

---

Context:
{context}

---

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.chat_deployment,
            messages=[
                {"role": "system", "content": "Answer strictly from context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1   # 🔥 less hallucination
        )

        return response.choices[0].message.content.strip()


# 🚀 RUN
if __name__ == "__main__":

    rag_search = RAGSearch(
        persist_dir="faiss_store",
        embedding_deployment="text-embedding-3-large",
        chat_deployment="gpt-4o-mini"   # 🔥 recommended
    )

    query = "What is attention mechanism?"
    answer = rag_search.search_and_summarize(query, top_k=4)

    print("\n🧠 Answer:\n", answer)