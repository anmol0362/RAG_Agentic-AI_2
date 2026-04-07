import os
import json
from typing import List, Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from dotenv import load_dotenv
load_dotenv()

# =========================================================
# WHY THIS FILE EXISTS:
# After embeddings are generated and saved locally,
# this file uploads them into Azure AI Search index
# so they can be searched at query time.
#
# NEW: payload now includes content_type and section fields
# so Azure knows what type of content each chunk is.
# =========================================================

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

DOC_INDEX_NAME    = "docs-index"
EMBEDDED_DOCS_PATH = "data/embedded/embedded_docs.json"
EMBEDDED_ROWS_PATH = "data/embedded/embedded_rows.json"
UPLOAD_BATCH_SIZE  = 500


# =========================================================
# HELPERS
# =========================================================

def load_json(path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSON file. Returns empty list if file is empty
    or contains something other than a list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Splits a large list into smaller batches.
    Azure Search has limits on how many docs per upload call.
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def get_search_client(index_name: str) -> SearchClient:
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    )


# =========================================================
# PAYLOAD BUILDER
# Converts embedded records into the format Azure Search expects.
# Field names here MUST match exactly what's in build_index.py.
#
# NEW FIELDS:
#   content_type → "text" or "table"
#   section      → e.g. "3.2 Probable Cause"
# =========================================================

def build_docs_payload(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = []

    for i, r in enumerate(records):
        payload.append({
            # Required key field
            "id": str(r.get("chunk_id", f"doc_{i}")),

            # Main searchable text
            "content": str(r.get("text", "")),

            # Metadata
            "doc_id":       str(r.get("doc_id", "")),
            "source":       str(r.get("source", "")),
            "file_type":    str(r.get("file_type", "")),
            "chunk_type":   str(r.get("chunk_type", "")),

            # NEW: content type and section
            "content_type": str(r.get("content_type", "text")),
            "section":      str(r.get("section", "")),

            "page":         int(r.get("page", 0) or 0),
            "chunk_size":   int(r.get("chunk_size", len(str(r.get("text", ""))))),

            # The embedding vector (3072 floats)
            "contentVector": r.get("embedding", [])
        })

    return payload


# =========================================================
# UPLOAD FUNCTION
# Uploads documents to Azure Search in batches.
# Prints success/fail count per batch.
# =========================================================

def upload_documents(index_name: str, documents: List[Dict[str, Any]]):
    if not documents:
        print(f"[INFO] No documents to upload for index: {index_name}")
        return

    client = get_search_client(index_name)
    batches = batch_list(documents, UPLOAD_BATCH_SIZE)

    print(f"\nUploading to index : {index_name}")
    print(f"Total documents    : {len(documents)}")
    print(f"Total batches      : {len(batches)}")

    total_success = 0
    for i, batch in enumerate(batches, start=1):
        results = client.upload_documents(documents=batch)
        success_count = sum(1 for r in results if r.succeeded)
        total_success += success_count
        print(f"Batch {i}/{len(batches)} → Uploaded {success_count}/{len(batch)}")

    print(f"✅ Total uploaded: {total_success}/{len(documents)}")


# =========================================================
# MAIN
# =========================================================

def run_upload():
    print("\n" + "=" * 80)
    print("AZURE AI SEARCH UPLOAD")
    print("=" * 80)
    print(f"Endpoint       : {AZURE_SEARCH_ENDPOINT}")
    print(f"Doc Index      : {DOC_INDEX_NAME}")
    print(f"Docs Path      : {EMBEDDED_DOCS_PATH}")
    print("=" * 80)

    # --- Load embedded docs ---
    if not os.path.exists(EMBEDDED_DOCS_PATH):
        raise FileNotFoundError(f"Embedded docs not found: {EMBEDDED_DOCS_PATH}")

    docs = load_json(EMBEDDED_DOCS_PATH)
    print(f"\nLoaded embedded docs : {len(docs)}")

    # --- Print content type breakdown ---
    text_count  = sum(1 for r in docs if r.get("content_type") == "text")
    table_count = sum(1 for r in docs if r.get("content_type") == "table")
    print(f"  Text chunks        : {text_count}")
    print(f"  Table chunks       : {table_count}")

    # --- Build and upload payload ---
    docs_payload = build_docs_payload(docs)
    upload_documents(DOC_INDEX_NAME, docs_payload)

    print("\n🎉 Upload complete.")


if __name__ == "__main__":
    run_upload()