from rag_tool import rag_search

results = rag_search("What is attention mechanism?")

for i, r in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(r["text"][:500])