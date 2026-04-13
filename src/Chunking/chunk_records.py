import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from src.Chunking.chunk_text import chunk_record

load_dotenv()

# -------------------------
# AZURE BLOB STORAGE CONFIG
# -------------------------
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "raw-records")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")

blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)
container_client = blob_service_client.get_container_client(
    AZURE_STORAGE_CONTAINER
)

# -------------------------
# FILE CONFIG
# -------------------------
BLOB_INPUT_FILE = "cleaned/cleaned_records.jsonl"
OUTPUT_PATH = Path("data/chunks/chunked_records.jsonl")
BLOB_OUTPUT_FILE = "chunks/chunked_records.jsonl"


def list_blobs_in_container(prefix=None):
    print(f"[INFO] Listing blobs in container: {AZURE_STORAGE_CONTAINER}")
    blobs = container_client.list_blobs(name_starts_with=prefix)
    found = False
    for blob in blobs:
        print(f" - {blob.name}")
        found = True
    if not found:
        print("[INFO] No blobs found.")


def download_blob_to_temp(blob_name: str) -> Path:
    """
    Download a blob file to a temporary local file.
    """
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():
        raise FileNotFoundError(
            f"Blob not found: '{blob_name}' in container '{AZURE_STORAGE_CONTAINER}'"
        )

    suffix = Path(blob_name).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)

    with open(temp_path, "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())

    print(f"[BLOB DOWNLOAD] {blob_name} -> {temp_path}")
    return temp_path


import time
from azure.storage.blob import BlobBlock
import uuid

def upload_file_to_blob(local_file_path: str, blob_name: str, max_retries: int = 5):
    """
    Upload a local file to Azure Blob Storage using manual block upload
    to handle large files and timeouts, with retry logic.
    """
    blob_client = container_client.get_blob_client(blob_name)
    chunk_size = 4 * 1024 * 1024  # 4MB chunks

    for attempt in range(1, max_retries + 1):
        try:
            block_list = []

            with open(local_file_path, "rb") as data:
                while True:
                    chunk = data.read(chunk_size)
                    if not chunk:
                        break

                    block_id = str(uuid.uuid4()).replace("-", "")[:32]
                    # Pad to exactly 32 chars (required by Azure)
                    block_id = block_id.ljust(32, "0")
                    import base64
                    block_id_b64 = base64.b64encode(block_id.encode()).decode()

                    blob_client.stage_block(block_id=block_id_b64, data=chunk, length=len(chunk))
                    block_list.append(BlobBlock(block_id=block_id_b64))

            blob_client.commit_block_list(block_list)
            print(f"[BLOB UPLOAD] {local_file_path} -> {blob_name} ({len(block_list)} blocks)")
            return  # success

        except Exception as e:
            wait = 2 ** attempt
            print(f"[RETRY {attempt}/{max_retries}] Upload failed: {e}. Retrying in {wait}s...")
            if attempt == max_retries:
                print(f"[ERROR] Upload failed after {max_retries} attempts.")
                raise
            time.sleep(wait)


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print(f"[INFO] Using container: {AZURE_STORAGE_CONTAINER}")
    list_blobs_in_container(prefix="cleaned/")

    print(f"[INFO] Downloading cleaned records from Blob: {BLOB_INPUT_FILE}")
    temp_input_path = download_blob_to_temp(BLOB_INPUT_FILE)

    print(f"[INFO] Loading cleaned records from: {temp_input_path}")
    records = load_jsonl(temp_input_path)
    print(f"[INFO] Loaded {len(records)} cleaned records")

    all_chunks = []

    for i, record in enumerate(records, start=1):
        chunks = chunk_record(record)

        if chunks:
            chunk_type = chunks[0].get("chunk_type", "unknown")
        else:
            chunk_type = "none"

        print(
            f"[INFO] Chunking record {i}/{len(records)} | "
            f"source={record.get('source')} | "
            f"type={record.get('file_type')} | "
            f"chunking={chunk_type} | "
            f"chunks_created={len(chunks)}"
        )

        all_chunks.extend(chunks)

    print(f"[INFO] Total chunks created: {len(all_chunks)}")

    # Save locally
    save_jsonl(all_chunks, OUTPUT_PATH)
    print(f"[INFO] Saved chunked records locally to: {OUTPUT_PATH}")

    # Upload chunked records to Blob
    upload_file_to_blob(str(OUTPUT_PATH), BLOB_OUTPUT_FILE)
    print(f"[INFO] Uploaded chunked_records.jsonl to Blob -> {BLOB_OUTPUT_FILE}")

    print("[INFO] Verifying chunk blob upload...")
    list_blobs_in_container(prefix="chunks/")

    # Cleanup temp file
    try:
        temp_input_path.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()