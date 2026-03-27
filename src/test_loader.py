from src.data_loader import load_all_documents

if __name__ == "__main__":
    records = load_all_documents(
        data_dir="data",
        save_path="data/extracted/test_output.jsonl"
    )

    print("\n===== SAMPLE OUTPUT =====")

    for i, r in enumerate(records[:5], start=1):
        print(f"\n--- Record {i} ---")
        print(r)

    print(f"\nTotal records: {len(records)}")