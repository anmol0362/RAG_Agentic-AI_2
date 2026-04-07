from pathlib import Path

from src.Data_Ingestion.data_loader import (
    load_all_documents,
    upload_file_to_blob
)
from src.Data_Ingestion.deduplication import (
    is_duplicate_file,
    get_file_hash,
    register_file
)

LOCAL_RAW_RECORDS_PATH = "data/raw_records/raw_records.jsonl"
BLOB_RAW_RECORDS_PATH = "raw_records/raw_records.jsonl"


if __name__ == "__main__":
    data_dir = "data"

    # ---------------------------------------------------------
    # STEP 1: FIND ALL PDFs / FILES
    # ---------------------------------------------------------
    pdfs = list(Path(data_dir).glob("**/*.pdf"))

    print(f"\n[DEBUG MAIN] PDFs found: {len(pdfs)}")
    for p in pdfs:
        print(" -", p)

    # ---------------------------------------------------------
    # STEP 2: CHECK DEDUP STATUS BEFORE PROCESSING
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEDUPLICATION CHECK")
    print("=" * 80)

    new_files = []
    skipped_files = []

    for p in pdfs:
        file_path = str(p)
        file_hash = get_file_hash(file_path)

        print(f"\n[FILE] {file_path}")
        print(f"[HASH] {file_hash}")

        if is_duplicate_file(file_path):
            print("[STATUS] DUPLICATE → SKIP")
            skipped_files.append(file_path)
        else:
            print("[STATUS] NEW FILE → WILL PROCESS")
            new_files.append(file_path)

    print("\n" + "=" * 80)
    print("DEDUP SUMMARY")
    print("=" * 80)
    print(f"New files to process: {len(new_files)}")
    print(f"Duplicate files skipped: {len(skipped_files)}")

    if not new_files:
        print("\n[INFO] No new files found.")

        if Path(LOCAL_RAW_RECORDS_PATH).exists():
            print(f"[INFO] Existing raw records found locally: {LOCAL_RAW_RECORDS_PATH}")

            upload_file_to_blob(
                local_file_path=LOCAL_RAW_RECORDS_PATH,
                blob_name=BLOB_RAW_RECORDS_PATH
            )

            print(f"[INFO] Uploaded existing raw records to Blob -> {BLOB_RAW_RECORDS_PATH}")
        else:
            print("[INFO] No existing raw_records.jsonl found locally. Nothing to process.")

        exit()

    # ---------------------------------------------------------
    # STEP 3: RUN LOADER ONLY IF NEW FILES EXIST
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING DOCUMENT LOADER")
    print("=" * 80)

    records = load_all_documents(
        data_dir=data_dir,
        save_path=LOCAL_RAW_RECORDS_PATH
    )

    # ---------------------------------------------------------
    # STEP 4: UPLOAD RAW RECORDS TO BLOB
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("UPLOADING RAW RECORDS TO BLOB")
    print("=" * 80)

    upload_file_to_blob(
        local_file_path=LOCAL_RAW_RECORDS_PATH,
        blob_name=BLOB_RAW_RECORDS_PATH
    )

    # ---------------------------------------------------------
    # STEP 5: REGISTER NEW FILES AFTER SUCCESS
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("REGISTERING NEW FILES")
    print("=" * 80)

    for file_path in new_files:
        register_file(file_path)
        print(f"[REGISTERED] {file_path}")

    # ---------------------------------------------------------
    # STEP 6: PRINT SAMPLE OUTPUT
    # ---------------------------------------------------------
    print("\n===== SAMPLE OUTPUT =====")

    for i, r in enumerate(records[:5], start=1):
        print(f"\n--- Record {i} ---")
        print(r)

    print(f"\nTotal records: {len(records)}")