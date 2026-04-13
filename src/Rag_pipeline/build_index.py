import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

load_dotenv()

# =========================================================
# WHY THIS FILE EXISTS:
# Before uploading embeddings to Azure AI Search,
# we need to define the INDEX SCHEMA — basically telling
# Azure what fields each document has and how to search them.
#
# NEW FIELDS ADDED:
#   content_type → "text" or "table" — lets you filter by type
#   section      → section heading from the document
#                  e.g. "3.2 Probable Cause"
#                  lets you filter search to specific sections
# =========================================================

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")

DOC_INDEX_NAME = "docs-index"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-large


def get_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    )


def get_vector_search_config() -> VectorSearch:
    return VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw"
        )]
    )


def build_docs_index() -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="file_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True, facetable=True),

        # NEW: content_type → "text" or "table"
        # Filter example: content_type eq 'table'
        SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True, facetable=True),

        # NEW: section → e.g. "3.2 Probable Cause"
        # Filter example: section eq '3.2 Probable Cause'
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="chunk_level", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="parent_id",   type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="chunk_size", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSION,
            vector_search_profile_name="myHnswProfile"
        ),
    ]

    return SearchIndex(
        name=DOC_INDEX_NAME,
        fields=fields,
        vector_search=get_vector_search_config()
    )


def create_or_update_index(index: SearchIndex):
    client = get_index_client()
    name = index.name
    existing_indexes = [i.name for i in client.list_indexes()]

    if name in existing_indexes:
        print(f"[INFO] Index '{name}' already exists → updating...")
        client.create_or_update_index(index)
        print(f"✅ Index updated: {name}")
    else:
        print(f"[INFO] Index '{name}' not found → creating...")
        client.create_or_update_index(index)
        print(f"✅ Index created: {name}")


def delete_index_if_exists(index_name: str):
    """
    Deletes index completely. Use when schema changes.
    WARNING: Deletes ALL documents. Cannot be undone.
    """
    client = get_index_client()
    existing_indexes = [i.name for i in client.list_indexes()]

    if index_name in existing_indexes:
        client.delete_index(index_name)
        print(f"🗑️  Deleted index: {index_name}")
    else:
        print(f"[INFO] Index '{index_name}' does not exist.")


def run_build_index():
    print("\n" + "=" * 80)
    print("AZURE AI SEARCH — BUILD INDEX")
    print("=" * 80)
    print(f"Endpoint       : {AZURE_SEARCH_ENDPOINT}")
    print(f"Doc Index      : {DOC_INDEX_NAME}")
    print(f"Embedding Dim  : {EMBEDDING_DIMENSION}")
    print(f"New Fields     : content_type, section")
    print("=" * 80)

    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_ADMIN_KEY:
        raise ValueError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_ADMIN_KEY in .env")

    print(f"\n[STEP 1] Building docs index: {DOC_INDEX_NAME}")
    docs_index = build_docs_index()
    create_or_update_index(docs_index)

    print("\n" + "=" * 80)
    print("✅ Index is ready.")
    print("👉 Now run upload_to_search.py to upload your embeddings.")
    print("=" * 80)


if __name__ == "__main__":
    run_build_index()