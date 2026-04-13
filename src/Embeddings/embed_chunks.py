import os
import json
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# EMBEDDING MODEL CONFIG
# =========================================================

DOC_EMBEDDING_MODEL = "text-embedding-3-large"
ROW_EMBEDDING_MODEL = "text-embedding-3-large"

DOC_EMBEDDING_DIM = 1536
ROW_EMBEDDING_DIM = 1536

DOC_BATCH_SIZE = 16
ROW_BATCH_SIZE = 64

NORMALIZE_VECTORS = True
VECTOR_DTYPE = np.float32
SIMILARITY_METRIC = "cosine"

MAX_RETRIES = 5
RETRY_SLEEP_SECONDS = 2

INPUT_CHUNK_KEY = "text"
INPUT_CHUNK_TYPE_KEY = "chunk_type"

DOC_CHUNK_TYPES = {"structure_text"}
ROW_CHUNK_TYPES = {"row"}

SAVE_OUTPUT_DIR = "data/embedded"

# =========================================================
# AZURE BLOB CONFIG
# =========================================================

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "documents")

BLOB_INPUT_FILE  = "chunks/chunked_records.jsonl"
BLOB_OUTPUT_DOCS = "embedded/embedded_docs.json"
BLOB_OUTPUT_ROWS = "embedded/embedded_rows.json"
BLOB_OUTPUT_ALL  = "embedded/embedded_all.json"

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)

# =========================================================
# AZURE OPENAI CONFIG
# =========================================================

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

DOC_EMBEDDING_DEPLOYMENT = os.getenv("DOC_EMBEDDING_DEPLOYMENT", AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
ROW_EMBEDDING_DEPLOYMENT = os.getenv("ROW_EMBEDDING_DEPLOYMENT", AZURE_OPENAI_EMBEDDING_DEPLOYMENT)

print("\n" + "=" * 80)
print("AZURE EMBEDDING CONFIG DEBUG")
print("=" * 80)
print(f"ENDPOINT           : {AZURE_OPENAI_ENDPOINT}")
print(f"API VERSION        : {AZURE_OPENAI_API_VERSION}")
print(f"EMBED DEPLOYMENT   : {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
print(f"KEY PRESENT        : {'YES' if AZURE_OPENAI_API_KEY else 'NO'}")
print(f"KEY PREFIX         : {AZURE_OPENAI_API_KEY[:6] if AZURE_OPENAI_API_KEY else 'NO KEY'}")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# =========================================================
# BLOB HELPERS
# =========================================================

def download_blob_to_temp(blob_name: str) -> Path:
    blob_client = container_client.get_blob_client(blob_name)
    if not blob_client.exists():
        raise FileNotFoundError(
            f"Blob not found: '{blob_name}' in container '{AZURE_STORAGE_CONTAINER}'"
        )
    suffix = Path(blob_name).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    with open(temp_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print(f"[BLOB DOWNLOAD] {blob_name} -> {temp_path}")
    return temp_path


def upload_file_to_blob(local_file_path: str, blob_name: str, max_retries: int = 5):
    blob_client = container_client.get_blob_client(blob_name)
    file_size = os.path.getsize(local_file_path)
    print(f"[BLOB UPLOAD] File size: {file_size / (1024*1024):.2f} MB")

    for attempt in range(1, max_retries + 1):
        try:
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    max_concurrency=1,          # reduce to 1 to avoid socket thrashing
                    connection_timeout=300,     # 5 min connection timeout
                    read_timeout=600,           # 10 min read/write timeout
                )
            print(f"[BLOB UPLOAD] ✅ {local_file_path} -> {blob_name}")
            return

        except Exception as e:
            wait = 2 ** attempt
            print(f"[RETRY {attempt}/{max_retries}] Upload failed: {e}. Retrying in {wait}s...")
            if attempt == max_retries:
                raise
            time.sleep(wait)
# =========================================================
# PRINT ALL EMBEDDING PARAMETERS
# =========================================================

def print_embedding_parameters():
    print("\n" + "=" * 80)
    print("EMBEDDING PARAMETERS")
    print("=" * 80)

    print("\n[MODELS]")
    print(f"DOC_EMBEDDING_MODEL         : {DOC_EMBEDDING_MODEL}")
    print(f"ROW_EMBEDDING_MODEL         : {ROW_EMBEDDING_MODEL}")
    print(f"DOC_EMBEDDING_DEPLOYMENT    : {DOC_EMBEDDING_DEPLOYMENT}")
    print(f"ROW_EMBEDDING_DEPLOYMENT    : {ROW_EMBEDDING_DEPLOYMENT}")

    print("\n[VECTOR SETTINGS]")
    print(f"DOC_EMBEDDING_DIM           : {DOC_EMBEDDING_DIM}")
    print(f"ROW_EMBEDDING_DIM           : {ROW_EMBEDDING_DIM}")
    print(f"NORMALIZE_VECTORS           : {NORMALIZE_VECTORS}")
    print(f"VECTOR_DTYPE                : {VECTOR_DTYPE}")
    print(f"SIMILARITY_METRIC           : {SIMILARITY_METRIC}")

    print("\n[BATCH SETTINGS]")
    print(f"DOC_BATCH_SIZE              : {DOC_BATCH_SIZE}")
    print(f"ROW_BATCH_SIZE              : {ROW_BATCH_SIZE}")

    print("\n[ROUTING]")
    print(f"DOC_CHUNK_TYPES             : {DOC_CHUNK_TYPES}")
    print(f"ROW_CHUNK_TYPES             : {ROW_CHUNK_TYPES}")
    print(f"INPUT_CHUNK_KEY             : {INPUT_CHUNK_KEY}")
    print(f"INPUT_CHUNK_TYPE_KEY        : {INPUT_CHUNK_TYPE_KEY}")

    print("\n[RETRY SETTINGS]")
    print(f"MAX_RETRIES                 : {MAX_RETRIES}")
    print(f"RETRY_SLEEP_SECONDS         : {RETRY_SLEEP_SECONDS}")

    print("\n[BLOB PATHS]")
    print(f"BLOB INPUT                  : {BLOB_INPUT_FILE}")
    print(f"BLOB OUTPUT DOCS            : {BLOB_OUTPUT_DOCS}")
    print(f"BLOB OUTPUT ROWS            : {BLOB_OUTPUT_ROWS}")
    print(f"BLOB OUTPUT ALL             : {BLOB_OUTPUT_ALL}")

    print("=" * 80 + "\n")


# =========================================================
# HELPERS
# =========================================================

def ensure_output_dir():
    Path(SAVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_chunks(path):
    print(f"\nLoading chunks from: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                print(f"[ERROR] Failed to parse line {line_num}: {e}")
    print(f"[LOAD] Total chunks loaded: {len(records)}")
    if records:
        print(f"[LOAD] Sample keys: {list(records[0].keys())}")
    return records


def save_json(data: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    return str(text or "").strip()


def split_by_chunk_type(records: List[Dict[str, Any]]):
    doc_records = []
    row_records = []
    for record in records:
        chunk_type = str(record.get(INPUT_CHUNK_TYPE_KEY, "")).strip().lower()
        if chunk_type in DOC_CHUNK_TYPES:
            doc_records.append(record)
        elif chunk_type in ROW_CHUNK_TYPES:
            row_records.append(record)
        else:
            doc_records.append(record)
    return doc_records, row_records


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# =========================================================
# VECTOR PROCESSING
# =========================================================

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def process_embeddings(embeddings: List[List[float]]) -> np.ndarray:
    arr = np.array(embeddings, dtype=VECTOR_DTYPE)
    if NORMALIZE_VECTORS:
        arr = l2_normalize(arr)
    return arr


def print_vector_stats(vectors: np.ndarray, label: str):
    print(f"\n[{label} VECTOR STATS]")
    print(f"Shape              : {vectors.shape}")
    print(f"Dtype              : {vectors.dtype}")
    print(f"Min value          : {vectors.min():.6f}")
    print(f"Max value          : {vectors.max():.6f}")
    print(f"Mean value         : {vectors.mean():.6f}")
    if len(vectors) > 0:
        print(f"First vector norm  : {np.linalg.norm(vectors[0]):.6f}")


# =========================================================
# EMBEDDING API CALL
# =========================================================

def get_embeddings(texts: List[str], deployment_name: str) -> List[List[float]]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=deployment_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SECONDS)
    return []


# =========================================================
# EMBED RECORDS
# =========================================================

def embed_records(
    records: List[Dict[str, Any]],
    deployment_name: str,
    expected_dim: int,
    batch_size: int,
    embedding_type: str
) -> List[Dict[str, Any]]:
    if not records:
        return []

    print("\n" + "=" * 80)
    print(f"STARTING {embedding_type.upper()} EMBEDDING")
    print("=" * 80)
    print(f"Deployment Name    : {deployment_name}")
    print(f"Expected Dimension : {expected_dim}")
    print(f"Batch Size         : {batch_size}")
    print(f"Total Records      : {len(records)}")

    embedded_records = []
    batches = batch_list(records, batch_size)

    for batch_idx, batch in enumerate(batches, start=1):
        texts = [clean_text(r.get(INPUT_CHUNK_KEY, "")) for r in batch]
        print(f"\n[Batch {batch_idx}/{len(batches)}] Embedding {len(batch)} records...")
        raw_embeddings = get_embeddings(texts, deployment_name)
        vectors = process_embeddings(raw_embeddings)

        if vectors.shape[1] != expected_dim:
            raise ValueError(
                f"Dimension mismatch for {embedding_type}: "
                f"expected {expected_dim}, got {vectors.shape[1]}"
            )

        print_vector_stats(vectors, f"{embedding_type.upper()} BATCH {batch_idx}")

        for record, vector in zip(batch, vectors):
            new_record = dict(record)
            new_record["embedding_model"] = deployment_name
            new_record["embedding_type"] = embedding_type
            new_record["embedding_dimension"] = int(vector.shape[0])
            new_record["normalized"] = NORMALIZE_VECTORS
            new_record["similarity_metric"] = SIMILARITY_METRIC
            new_record["embedding"] = vector.tolist()
            embedded_records.append(new_record)

    print(f"\n✅ Finished {embedding_type.upper()} embeddings: {len(embedded_records)}")
    return embedded_records


# =========================================================
# MAIN
# =========================================================

def run_embedding_pipeline():
    print_embedding_parameters()
    ensure_output_dir()

    # Download input from Blob
    print(f"[INFO] Downloading chunks from Blob: {BLOB_INPUT_FILE}")
    temp_input_path = download_blob_to_temp(BLOB_INPUT_FILE)

    records = load_chunks(temp_input_path)
    print(f"Loaded total chunks: {len(records)}")

    doc_records, row_records = split_by_chunk_type(records)

    print("\n" + "=" * 80)
    print("CHUNK ROUTING SUMMARY")
    print("=" * 80)
    print(f"Document Chunks     : {len(doc_records)}")
    print(f"Row Chunks          : {len(row_records)}")
    print("=" * 80)

    embedded_docs = embed_records(
        records=doc_records,
        deployment_name=DOC_EMBEDDING_DEPLOYMENT,
        expected_dim=DOC_EMBEDDING_DIM,
        batch_size=DOC_BATCH_SIZE,
        embedding_type="document"
    )

    embedded_rows = embed_records(
        records=row_records,
        deployment_name=ROW_EMBEDDING_DEPLOYMENT,
        expected_dim=ROW_EMBEDDING_DIM,
        batch_size=ROW_BATCH_SIZE,
        embedding_type="row"
    )

    docs_out     = os.path.join(SAVE_OUTPUT_DIR, "embedded_docs.json")
    rows_out     = os.path.join(SAVE_OUTPUT_DIR, "embedded_rows.json")
    combined_out = os.path.join(SAVE_OUTPUT_DIR, "embedded_all.json")

    # Always save and upload docs
    save_json(embedded_docs, docs_out)

    # Only save and upload rows if not empty
    if len(embedded_rows) > 0:
        save_json(embedded_rows, rows_out)
    else:
        print("[SKIP] No row embeddings, skipping rows save")

    # Save and upload combined
    combined = embedded_docs + embedded_rows
    save_json(combined, combined_out)

    # Upload to Blob
    print("\n[INFO] Uploading embedded outputs to Blob...")

    upload_file_to_blob(docs_out, BLOB_OUTPUT_DOCS)

    if len(embedded_rows) > 0:
        upload_file_to_blob(rows_out, BLOB_OUTPUT_ROWS)
    else:
        print("[SKIP] No row embeddings, skipping rows upload")

    if len(combined) > 0:
        upload_file_to_blob(combined_out, BLOB_OUTPUT_ALL)
    else:
        print("[SKIP] Combined is empty, skipping upload")

    print("\n" + "=" * 80)
    print("SAVE SUMMARY")
    print("=" * 80)
    print(f"Embedded Docs  : {docs_out}  ->  {BLOB_OUTPUT_DOCS}")
    if len(embedded_rows) > 0:
        print(f"Embedded Rows  : {rows_out}  ->  {BLOB_OUTPUT_ROWS}")
    else:
        print(f"Embedded Rows  : SKIPPED (no row chunks)")
    print(f"Combined       : {combined_out}  ->  {BLOB_OUTPUT_ALL}")
    print("=" * 80)

    # Cleanup temp file
    try:
        temp_input_path.unlink(missing_ok=True)
    except Exception:
        pass

    print("\n🎉 Embedding pipeline completed successfully.")


if __name__ == "__main__":
    run_embedding_pipeline()