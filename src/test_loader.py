from src.data_loader import load_all_documents

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
docs = load_all_documents(str(BASE_DIR / "data"))

print("Total documents loaded:", len(docs))

if len(docs) > 0:
    print("\n--- FIRST DOCUMENT ---\n")
    print(docs[0].page_content[:500])

    print("\n--- LAST DOCUMENT ---\n")
    print(docs[-1].page_content[:500])