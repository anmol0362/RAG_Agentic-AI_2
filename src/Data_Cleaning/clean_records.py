import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from src.Data_Cleaning.clean_text import clean_record

load_dotenv()

# -------------------------
# AZURE BLOB STORAGE CONFIG
# -------------------------
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "documents")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")

blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)
container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)

# -------------------------
# FILE CONFIG
# -------------------------
BLOB_INPUT_FILE = "raw_records/raw_records.jsonl"   # FIXED
OUTPUT_PATH = Path("data/cleaned/cleaned_records.jsonl")
BLOB_OUTPUT_FILE = "cleaned/cleaned_records.jsonl"


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

def upload_file_to_blob(local_file_path: str, blob_name: str, max_retries: int = 5):
    """
    Upload a local file to Azure Blob Storage using chunked upload with retry logic.
    """
    blob_client = container_client.get_blob_client(blob_name)

    for attempt in range(1, max_retries + 1):
        try:
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    max_concurrency=4  # parallel upload
                )
            print(f"[BLOB UPLOAD] {local_file_path} -> {blob_name}")
            return  # success, exit

        except Exception as e:
            wait = 2 ** attempt  # exponential backoff: 2, 4, 8, 16, 32 sec
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
    list_blobs_in_container(prefix="raw_records/")

    print(f"[INFO] Downloading raw records from Blob: {BLOB_INPUT_FILE}")
    temp_input_path = download_blob_to_temp(BLOB_INPUT_FILE)

    print(f"[INFO] Loading records from: {temp_input_path}")
    records = load_jsonl(temp_input_path)
    print(f"[INFO] Loaded {len(records)} records")

    cleaned_records = []
    skipped = 0

    for i, record in enumerate(records, start=1):
        print(
            f"[INFO] Cleaning record {i}/{len(records)} | "
            f"source={record.get('source')} | type={record.get('file_type')}"
        )

        cleaned = clean_record(record)

        if cleaned.get("text", "").strip():
            cleaned_records.append(cleaned)
        else:
            skipped += 1
            print(f"[SKIPPED] Empty after cleaning: {record.get('source')}")

    print(f"[INFO] Cleaned records: {len(cleaned_records)}")
    print(f"[INFO] Skipped empty records: {skipped}")

    # Save locally
    save_jsonl(cleaned_records, OUTPUT_PATH)
    print(f"[INFO] Saved cleaned records locally to: {OUTPUT_PATH}")

    # Upload cleaned file back to Blob
    upload_file_to_blob(str(OUTPUT_PATH), BLOB_OUTPUT_FILE)
    print(f"[INFO] Uploaded cleaned_records.jsonl back to Blob -> {BLOB_OUTPUT_FILE}")

    print("[INFO] Verifying cleaned blob upload...")
    list_blobs_in_container(prefix="cleaned/")

    # Cleanup temp file
    try:
        temp_input_path.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()