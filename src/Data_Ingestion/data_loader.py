from pathlib import Path
from typing import List, Dict, Any
import os
import uuid
import json
import re

import fitz  # PyMuPDF - used for PDF text and table extraction
import pandas as pd
import pyodbc
import pytesseract
from PIL import Image

from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

# =========================================================
# WHY THIS FILE EXISTS:
# This is the FIRST step in the RAG pipeline.
# It reads every file (PDF, TXT, CSV etc.) and converts
# them into a list of "records" — structured dicts that
# contain the text + metadata (page, source, type, section).
#
# KEY IMPROVEMENTS IN THIS VERSION:
#   1. TOC pages skipped (Table of Contents)
#   2. Abbreviation pages skipped (CFM, CFR, CRM lists)
#   3. Figure/Table list pages skipped
#   4. Tables extracted separately from text (row by row)
#   5. Section headers detected and stored as metadata
#   6. content_type field added ("text" or "table")
# =========================================================

# -------------------------
# AZURE BLOB STORAGE CONFIG
# -------------------------
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "documents")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")

try:
    conn_parts = dict(
        part.split("=", 1)
        for part in AZURE_STORAGE_CONNECTION_STRING.split(";")
        if "=" in part
    )
    account_name = conn_parts.get("AccountName", "UNKNOWN")
except Exception:
    account_name = "UNKNOWN"

print(f"[DEBUG] Azure Storage Account: {account_name}")
print(f"[DEBUG] Azure Container      : {AZURE_STORAGE_CONTAINER}")

blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)
container_client = blob_service_client.get_container_client(
    AZURE_STORAGE_CONTAINER
)

# -------------------------
# OUTPUT PATHS
# -------------------------
LOCAL_RAW_RECORDS_PATH = "data/raw_records/raw_records.jsonl"
BLOB_RAW_RECORDS_PATH  = "raw_records/raw_records.jsonl"

# -------------------------
# OCR CONFIG
# -------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# PRETTY PRINT HELPERS
# =========================================================

def print_block(title: str):
    print("\n" + "=" * 110)
    print(f"[PIPELINE] {title}")
    print("=" * 110)


def print_summary(label: str, value):
    print(f"[SUMMARY] {label}: {value}")


# =========================================================
# RECORD FACTORY
# Every piece of extracted content becomes a "record" dict.
# All records have the same keys so downstream scripts
# (cleaning, chunking, embedding) can process them uniformly.
#
# Fields:
#   doc_id           → unique ID for this record
#   source           → relative path to the source file
#   file_type        → "pdf", "txt", "csv" etc.
#   page             → page number (PDF only)
#   table            → table name (MDB/PDF tables)
#   extraction_method→ "text", "table_extraction", "ocr"
#   content_type     → "text" or "table"
#   section          → nearest section heading on this page
#   text             → the actual content
# =========================================================

def make_record(
    source: str,
    file_type: str,
    text: str,
    page: int = None,
    table_name: str = None,
    extraction_method: str = "text",
    content_type: str = "text",
    section: str = None,
) -> Dict[str, Any]:
    return {
        "doc_id":            str(uuid.uuid4()),
        "source":            source,
        "file_type":         file_type,
        "page":              page,
        "table":             table_name,
        "extraction_method": extraction_method,
        "content_type":      content_type,
        "section":           section or "",
        "text":              text.strip() if text else ""
    }


# =========================================================
# NOISE PAGE DETECTOR
# Detects pages that have no useful content for RAG.
# These pages are skipped entirely during extraction.
#
# Types of noise pages detected:
#   1. Table of Contents  → dotted lines, page numbers
#   2. Abbreviations list → short lines like "CFM → company flight manual"
#   3. List of Figures    → "Figure 1 - Aircraft diagram...v"
#   4. List of Tables     → similar to above
#
# How abbreviation detection works:
#   Count the ratio of short lines (< 60 chars) on the page.
#   If 70%+ of lines are short AND there are 10+ lines
#   → likely an abbreviations or glossary page → skip it.
# =========================================================

def is_noise_page(text: str) -> bool:
    """
    Returns True if this page should be skipped.
    Covers: TOC, abbreviations, figures list, tables list.
    """
    if not text or len(text.strip()) < 50:
        return True  # nearly empty page → skip

    # --- Pattern-based TOC detection ---
    toc_patterns = [
        r"\.{5,}",                              # dotted lines like .......
        r"(?m)^\s*\d{1,3}\s*$",                # line that is just a page number
        r"(?i)\btable\s+of\s+contents\b",      # literal "table of contents"
        r"(?i)\blist\s+of\s+(figures|tables)\b" # "list of figures" or "list of tables"
    ]

    toc_match_count = sum(
        1 for pattern in toc_patterns
        if re.search(pattern, text, re.MULTILINE)
    )

    if toc_match_count >= 2:
        return True

    # --- Abbreviation/Glossary page detection ---
    # These pages have many very short lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 10:
        short_lines = sum(1 for l in lines if len(l) < 60)
        short_line_ratio = short_lines / len(lines)

        # 70%+ short lines = likely abbreviations page
        if short_line_ratio > 0.70:
            return True

    # --- Executive Summary header-only pages ---
    # Very short pages with only a title
    words = text.split()
    if len(words) < 30:
        return True

    return False


# =========================================================
# SECTION HEADER DETECTOR
# NTSB report has numbered sections like:
#   "1.7 Meteorological Information"
#   "3.2 Probable Cause"
#   "2.1 History of the Flight"
#
# We detect these and store them as metadata on every chunk.
# This enables filtered search:
#   filter="section eq '3.2 Probable Cause'"
# =========================================================

def detect_section_header(text: str) -> str:
    """
    Finds the first line that looks like a section heading.
    Returns the heading string, or empty string if none found.
    """
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        # Match: "1.7 Something" or "3.2.1 Something"
        if re.match(r"^\d+(\.\d+)+\s+[A-Z]", line) and len(line) < 120:
            return line
    return ""


# =========================================================
# TEXT CLEANER
# Basic cleanup before saving raw records.
# More thorough cleaning happens in build_cleaning.py.
# =========================================================

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)    # collapse multiple spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 consecutive newlines
    return text.strip()


# =========================================================
# TABLE EXTRACTOR
# PyMuPDF's find_tables() detects table structures in PDFs.
# For each table, we convert to pandas DataFrame, then
# convert each ROW to a readable "field: value | field: value" string.
#
# Why row-by-row?
#   Each row is independently searchable.
#   Mixing all rows into one chunk loses row identity.
#   Example row: "Injuries: 2 | Fatalities: 0 | Date: Feb 12 2009"
# =========================================================

def extract_tables_from_page(
    page: fitz.Page,
    source: str,
    page_num: int,
    section: str
) -> List[Dict[str, Any]]:
    """
    Extracts all tables from a single PDF page.
    Returns one record per table row.
    """
    records = []

    try:
        table_finder = page.find_tables()

        for table_idx, table in enumerate(table_finder.tables, start=1):
            df = table.to_pandas()

            if df.empty:
                continue

            # Clean column names
            df.columns = [str(c).strip() for c in df.columns]

            for row_idx, row in df.iterrows():
                parts = []
                for col, val in row.items():
                    val_str = str(val).strip()
                    # Skip empty, None, nan values
                    if val_str and val_str.lower() not in {"nan", "none", ""}:
                        parts.append(f"{col}: {val_str}")

                row_text = " | ".join(parts)

                # Only keep rows with meaningful content
                if len(row_text.strip()) > 20:
                    records.append(make_record(
                        source=source,
                        file_type="pdf",
                        text=row_text,
                        page=page_num,
                        table_name=f"table_{table_idx}_row_{row_idx}",
                        extraction_method="table_extraction",
                        content_type="table",
                        section=section
                    ))

        if records:
            print(f"    [TABLE] Page {page_num}: {len(records)} table rows extracted")

    except Exception as e:
        print(f"    [TABLE ERROR] Page {page_num}: {e}")

    return records


# =========================================================
# PDF EXTRACTOR (MAIN)
# Processes one PDF file page by page.
#
# For each page:
#   1. Skip if noise page (TOC, abbreviations, figures list)
#   2. Detect section header → store as metadata
#   3. Extract tables separately → content_type="table"
#   4. Extract remaining text → content_type="text"
#
# This separation is critical for quality:
#   Text chunks → paragraph chunker (split into pieces)
#   Table chunks → table chunker (keep rows as-is)
# =========================================================

def extract_pdf(pdf_file: Path, data_path: Path) -> List[Dict[str, Any]]:
    records = []
    source = str(pdf_file.relative_to(data_path))
    current_section = ""

    # Counters for summary
    skipped_noise = 0
    text_records = 0
    table_records = 0

    print(f"\n[PDF] Processing: {pdf_file.name}")

    try:
        doc = fitz.open(str(pdf_file))
        print(f"[PDF] Total pages: {len(doc)}")

        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text()
            text = clean_ocr_text(raw_text)

            # -------------------------------------------------
            # STEP 1: Skip noise pages
            # TOC, abbreviations, figures list etc.
            # These add noise to retrieval without useful content.
            # -------------------------------------------------
            if is_noise_page(text):
                skipped_noise += 1
                print(f"    [SKIP NOISE] Page {page_num} - {len(text.split())} words")
                continue

            # -------------------------------------------------
            # STEP 2: Detect section header
            # Updates current_section if a new heading is found.
            # This gets attached to all records from this page.
            # -------------------------------------------------
            detected = detect_section_header(text)
            if detected:
                current_section = detected
                print(f"    [SECTION] Page {page_num}: {current_section}")

            # -------------------------------------------------
            # STEP 3: Extract tables from this page
            # Tables are extracted first and stored separately
            # with content_type="table"
            # -------------------------------------------------
            page_table_records = extract_tables_from_page(
                page=page,
                source=source,
                page_num=page_num,
                section=current_section
            )
            records.extend(page_table_records)
            table_records += len(page_table_records)

            # -------------------------------------------------
            # STEP 4: Extract text from this page
            # Only if text is long enough to be meaningful.
            # Short pages (< 200 chars) are usually
            # cover pages, blank pages, or chapter dividers.
            # -------------------------------------------------
            if len(text) > 200:
                records.append(make_record(
                    source=source,
                    file_type="pdf",
                    text=text,
                    page=page_num,
                    extraction_method="text",
                    content_type="text",
                    section=current_section
                ))
                text_records += 1

        print(f"[PDF] Done: {pdf_file.name}")
        print(f"    Text records  : {text_records}")
        print(f"    Table records : {table_records}")
        print(f"    Skipped pages : {skipped_noise}")

    except Exception as e:
        print(f"[ERROR] PDF {pdf_file}: {e}")

    return records


# =========================================================
# IMAGE EXTRACTOR (OCR)
# Uses Tesseract OCR to extract text from image files.
# =========================================================

def extract_text_from_image(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"[ERROR] OCR Image {image_path}: {e}")
        return ""


# =========================================================
# BLOB / FILE HELPERS
# =========================================================

def save_jsonl(records: List[Dict[str, Any]], output_path: str):
    """
    Saves records as JSONL (one JSON object per line).
    JSONL is used instead of JSON because:
      - Easier to stream large files
      - Can process line by line without loading everything
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved JSONL to: {output_file} ({len(records)} records)")


def ensure_container_exists():
    try:
        container_client.create_container()
        print(f"[INFO] Created container: {AZURE_STORAGE_CONTAINER}")
    except Exception:
        print(f"[INFO] Container already exists: {AZURE_STORAGE_CONTAINER}")


def upload_file_to_blob(local_file_path: str, blob_name: str):
    try:
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=blob_name
        )
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"[BLOB UPLOAD] {local_file_path} -> {blob_name}")
    except Exception as e:
        print(f"[ERROR] Blob upload failed for {local_file_path}: {e}")
        raise


def list_blobs_in_container(prefix=None):
    print(f"[INFO] Listing blobs in container: {AZURE_STORAGE_CONTAINER}")
    blobs = container_client.list_blobs(name_starts_with=prefix)
    found = False
    for blob in blobs:
        print(f" - {blob.name}")
        found = True
    if not found:
        print("[INFO] No blobs found.")


# =========================================================
# MAIN DOCUMENT LOADER
# Called by test_loader.py to process all files.
# Handles: PDF, TXT, CSV, XLSX, DOCX, MDB, Images
# =========================================================

def load_all_documents(data_dir: str, save_path: str = None) -> List[Dict[str, Any]]:
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path   : {data_path}")
    print(f"[DEBUG] Working dir : {os.getcwd()}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    records = []

    MAX_FILES_PER_TYPE  = None
    MAX_DOCS            = None
    IMPORTANT_MDB_TABLES = ["narratives", "events", "Findings"]

    def limit_files(files):
        return files if MAX_FILES_PER_TYPE is None else files[:MAX_FILES_PER_TYPE]

    # =========================================================
    # PDF FILES
    # =========================================================
    print("\n===== PROCESSING PDFs =====")
    pdf_files = limit_files(sorted(data_path.glob("**/*.pdf")))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        pdf_records = extract_pdf(pdf_file, data_path)
        records.extend(pdf_records)

    # Summary after PDFs
    pdf_text  = sum(1 for r in records if r.get("content_type") == "text")
    pdf_table = sum(1 for r in records if r.get("content_type") == "table")
    print(f"\n[PDF SUMMARY] Text: {pdf_text} | Tables: {pdf_table}")
    print(f"[SUMMARY] Total records so far: {len(records)}")

    # =========================================================
    # TXT FILES
    # Plain text — loaded as single records per file
    # =========================================================
    print("\n===== PROCESSING TXT =====")
    txt_files = limit_files(sorted(data_path.glob("**/*.txt")))
    print(f"[DEBUG] Found {len(txt_files)} TXT files")

    for txt_file in txt_files:
        print(f"[TXT] Loading: {txt_file}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                records.append(make_record(
                    source=str(txt_file.relative_to(data_path)),
                    file_type="txt",
                    text=doc.page_content,
                    extraction_method="text",
                    content_type="text"
                ))
        except Exception as e:
            print(f"[ERROR] TXT {txt_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # =========================================================
    # CSV FILES
    # Each row = one record with content_type="table"
    # =========================================================
    print("\n===== PROCESSING CSV =====")
    csv_files = limit_files(sorted(data_path.glob("**/*.csv")))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        print(f"[CSV] Loading: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            docs = loader.load()
            for doc in docs:
                records.append(make_record(
                    source=str(csv_file.relative_to(data_path)),
                    file_type="csv",
                    text=doc.page_content,
                    content_type="table"
                ))
        except Exception as e:
            print(f"[ERROR] CSV {csv_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # =========================================================
    # EXCEL FILES
    # =========================================================
    print("\n===== PROCESSING EXCEL =====")
    xlsx_files = limit_files(sorted(data_path.glob("**/*.xlsx")))
    print(f"[DEBUG] Found {len(xlsx_files)} XLSX files")

    for xlsx_file in xlsx_files:
        print(f"[XLSX] Loading: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            docs = loader.load()
            for doc in docs:
                records.append(make_record(
                    source=str(xlsx_file.relative_to(data_path)),
                    file_type="xlsx",
                    text=doc.page_content,
                    content_type="table"
                ))
        except Exception as e:
            print(f"[ERROR] Excel {xlsx_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # =========================================================
    # WORD FILES
    # =========================================================
    print("\n===== PROCESSING DOCX =====")
    docx_files = limit_files(sorted(data_path.glob("**/*.docx")))
    print(f"[DEBUG] Found {len(docx_files)} DOCX files")

    for docx_file in docx_files:
        print(f"[DOCX] Loading: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            docs = loader.load()
            for doc in docs:
                records.append(make_record(
                    source=str(docx_file.relative_to(data_path)),
                    file_type="docx",
                    text=doc.page_content,
                    content_type="text"
                ))
        except Exception as e:
            print(f"[ERROR] DOCX {docx_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # =========================================================
    # MDB FILES (Microsoft Access Database)
    # Each row in each table = one record
    # =========================================================
    print("\n===== PROCESSING MDB =====")
    mdb_files = limit_files(sorted(data_path.glob("**/*.mdb")))
    print(f"[DEBUG] Found {len(mdb_files)} MDB files")

    for mdb_file in mdb_files:
        print(f"[MDB] Loading: {mdb_file}")
        try:
            conn_str = (
                r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                rf"DBQ={mdb_file};"
            )
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            table_names = [
                row.table_name for row in cursor.tables()
                if row.table_type == "TABLE"
            ]

            for tbl in table_names:
                if IMPORTANT_MDB_TABLES and tbl not in IMPORTANT_MDB_TABLES:
                    continue
                try:
                    df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
                    for _, row in df.iterrows():
                        row_text = " | ".join([
                            str(val) for val in row.values if pd.notna(val)
                        ])
                        if row_text.strip():
                            records.append(make_record(
                                source=str(mdb_file.relative_to(data_path)),
                                file_type="mdb",
                                text=row_text,
                                table_name=tbl,
                                content_type="table"
                            ))
                    print(f"[MDB] Loaded {len(df)} rows from {tbl}")
                except Exception as te:
                    print(f"[ERROR] MDB table {tbl}: {te}")

            conn.close()
        except Exception as e:
            print(f"[ERROR] MDB {mdb_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # =========================================================
    # IMAGE FILES (OCR)
    # =========================================================
    print("\n===== PROCESSING IMAGES =====")
    image_files = limit_files(sorted(
        list(data_path.glob("**/*.png")) +
        list(data_path.glob("**/*.jpg")) +
        list(data_path.glob("**/*.jpeg"))
    ))
    print(f"[DEBUG] Found {len(image_files)} image files")

    for image_file in image_files:
        print(f"[IMAGE] Loading: {image_file}")
        try:
            text = extract_text_from_image(str(image_file))
            text = clean_ocr_text(text)
            if text.strip():
                records.append(make_record(
                    source=str(image_file.relative_to(data_path)),
                    file_type=image_file.suffix.lower().replace(".", ""),
                    text=text,
                    extraction_method="ocr",
                    content_type="text"
                ))
        except Exception as e:
            print(f"[ERROR] IMAGE {image_file}: {e}")

    print(f"[SUMMARY] Total records so far: {len(records)}")

    # =========================================================
    # FINAL CLEANUP
    # Remove records with empty text
    # =========================================================
    records = [r for r in records if r["text"] and str(r["text"]).strip()]

    if MAX_DOCS is not None:
        records = records[:MAX_DOCS]

    # Final breakdown
    print("\n===== FINAL SUMMARY =====")
    print(f"[INFO] Total records  : {len(records)}")
    print(f"[INFO] Text records   : {sum(1 for r in records if r.get('content_type') == 'text')}")
    print(f"[INFO] Table records  : {sum(1 for r in records if r.get('content_type') == 'table')}")

    if save_path:
        save_jsonl(records, save_path)

    return records


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("[INFO] data_loader.py is ready. Run test_loader.py to execute ingestion.")