from pathlib import Path
from typing import List, Dict, Any
import os
import uuid
import json

import pandas as pd
import pyodbc
import pytesseract
from src.pdf_ocr_azure import extract_text_from_pdf
from PIL import Image
from pdf2image import convert_from_path

from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


# -------------------------
# OCR CONFIG (EDIT THESE)
# -------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\Library\bin"   # change if needed


def make_record(
    source: str,
    file_type: str,
    text: str,
    page: int = None,
    table: str = None
) -> Dict[str, Any]:
    return {
        "doc_id": str(uuid.uuid4()),
        "source": source,
        "file_type": file_type,
        "page": page,
        "table": table,
        "text": text.strip() if text else ""
    }


def save_jsonl(records: List[Dict[str, Any]], output_path: str):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved JSONL to: {output_file}")


def extract_text_from_image(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"[ERROR] OCR Image {image_path}: {e}")
        return ""


def load_all_documents(data_dir: str, save_path: str = None) -> List[Dict[str, Any]]:
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    print("[DEBUG] Current working dir:", os.getcwd())

    records = []

    MAX_FILES_PER_TYPE = 10
    MAX_DOCS = 2000

    # -------------------------
    # PDF FILES
    # -------------------------
    print("\n===== PROCESSING PDFs =====")
    pdf_files = list(data_path.glob("**/*.pdf"))[:MAX_FILES_PER_TYPE]
    print(f"[DEBUG] Using {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"[PDF] Loading with Azure OCR: {pdf_file.name}")
        try:
            text = extract_text_from_pdf(str(pdf_file))

            records.append(
                make_record(
                    source=pdf_file.name,
                    file_type="pdf",
                    text=text
                )
            )

        except Exception as e:
            print(f"[ERROR] PDF {pdf_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # -------------------------
    # IMAGE FILES
    # -------------------------
    print("\n===== PROCESSING IMAGES =====")
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(data_path.glob(f"**/{ext}")))

    image_files = image_files[:MAX_FILES_PER_TYPE]
    print(f"[DEBUG] Using {len(image_files)} image files")

    for image_file in image_files:
        print(f"[IMG] OCR Processing: {image_file.name}")
        text = extract_text_from_image(str(image_file))

        records.append(
            make_record(
                source=image_file.name,
                file_type="image",
                text=text
            )
        )

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # -------------------------
    # TXT FILES
    # -------------------------
    print("\n===== PROCESSING TXT =====")
    txt_files = list(data_path.glob("**/*.txt"))[:MAX_FILES_PER_TYPE]

    for txt_file in txt_files:
        print(f"[TXT] Loading: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                records.append(
                    make_record(
                        source=txt_file.name,
                        file_type="txt",
                        text=doc.page_content
                    )
                )

        except Exception as e:
            print(f"[ERROR] TXT {txt_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # -------------------------
    # CSV FILES
    # -------------------------
    print("\n===== PROCESSING CSV =====")
    csv_files = list(data_path.glob("**/*.csv"))[:MAX_FILES_PER_TYPE]

    for csv_file in csv_files:
        print(f"[CSV] Loading: {csv_file.name}")
        try:
            loader = CSVLoader(str(csv_file))
            docs = loader.load()

            for doc in docs:
                records.append(
                    make_record(
                        source=csv_file.name,
                        file_type="csv",
                        text=doc.page_content
                    )
                )
        except Exception as e:
            print(f"[ERROR] CSV {csv_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # -------------------------
    # EXCEL FILES
    # -------------------------
    print("\n===== PROCESSING EXCEL =====")
    xlsx_files = list(data_path.glob("**/*.xlsx"))[:MAX_FILES_PER_TYPE]

    for xlsx_file in xlsx_files:
        print(f"[XLSX] Loading: {xlsx_file.name}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            docs = loader.load()

            for doc in docs:
                records.append(
                    make_record(
                        source=xlsx_file.name,
                        file_type="xlsx",
                        text=doc.page_content
                    )
                )
        except Exception as e:
            print(f"[ERROR] Excel {xlsx_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # -------------------------
    # WORD FILES
    # -------------------------
    print("\n===== PROCESSING DOCX =====")
    docx_files = list(data_path.glob("**/*.docx"))[:MAX_FILES_PER_TYPE]

    for docx_file in docx_files:
        print(f"[DOCX] Loading: {docx_file.name}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            docs = loader.load()

            for doc in docs:
                records.append(
                    make_record(
                        source=docx_file.name,
                        file_type="docx",
                        text=doc.page_content
                    )
                )
        except Exception as e:
            print(f"[ERROR] DOCX {docx_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # -------------------------
    # JSON FILES
    # -------------------------
    print("\n===== PROCESSING JSON =====")
    json_files = list(data_path.glob("**/*.json"))[:MAX_FILES_PER_TYPE]

    for json_file in json_files:
        print(f"[JSON] Loading: {json_file.name}")
        try:
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".",
                text_content=True
            )
            docs = loader.load()

            for doc in docs:
                records.append(
                    make_record(
                        source=json_file.name,
                        file_type="json",
                        text=doc.page_content
                    )
                )
        except Exception as e:
            print(f"[ERROR] JSON {json_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # -------------------------
    # MDB FILES
    # -------------------------
    print("\n===== PROCESSING MDB =====")
    mdb_files = list(data_path.glob("**/*.mdb"))[:3]
    print(f"[DEBUG] Using {len(mdb_files)} MDB files")

    IMPORTANT_TABLES = ["narratives", "events", "Findings"]
    MAX_ROWS_PER_TABLE = 80

    for mdb_file in mdb_files:
        print(f"[MDB] Loading: {mdb_file.name}")

        try:
            conn_str = (
                r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                rf"DBQ={mdb_file};"
            )

            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()

            table_names = [
                row.table_name
                for row in cursor.tables()
                if row.table_type == "TABLE"
            ]

            for table_name in table_names:
                if table_name not in IMPORTANT_TABLES:
                    continue

                try:
                    print(f"[MDB] Reading table: {table_name}")
                    df = pd.read_sql(
                        f"SELECT TOP {MAX_ROWS_PER_TABLE} * FROM [{table_name}]",
                        conn
                    )

                    for _, row in df.iterrows():
                        row_text = " | ".join(
                            [str(val) for val in row.values if pd.notna(val)]
                        )

                        records.append(
                            make_record(
                                source=mdb_file.name,
                                file_type="mdb",
                                text=row_text,
                                table=table_name
                            )
                        )

                    print(f"[MDB] Loaded {len(df)} rows from {table_name}")

                except Exception as table_error:
                    print(f"[ERROR] Table {table_name}: {table_error}")

            conn.close()

        except Exception as e:
            print(f"[ERROR] MDB {mdb_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # -------------------------
    # FINAL CLEANUP
    # -------------------------
    records = [r for r in records if r["text"].strip()]
    records = records[:MAX_DOCS]

    print("\n===== FINAL SUMMARY =====")
    print(f"[INFO] Final record count: {len(records)}")

    if save_path:
        save_jsonl(records, save_path)

    return records