from pathlib import Path
from typing import List, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import (
  PyMuPDFLoader,
  TextLoader,
  CSVLoader,
  Docx2txtLoader,
  JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

import pandas as pd
import pyodbc
import os




def load_all_documents(data_dir: str) -> List[Any]:

  data_path = Path(data_dir).resolve()
  print(f"[DEBUG] Data path: {data_path}")
  print("[DEBUG] Current working dir:", os.getcwd())

  documents = []

  # 🔥 LIMIT FILES (important)
  MAX_FILES_PER_TYPE = 10

  # -------------------------
  # PDF FILES (HIGH VALUE)
  # -------------------------
  pdf_files = list(data_path.glob("**/*.pdf"))[:MAX_FILES_PER_TYPE]
  print(f"[DEBUG] Using {len(pdf_files)} PDF files")

  for pdf_file in pdf_files:
    try:
      loader = PyMuPDFLoader(str(pdf_file))
      documents.extend(loader.load())
    except Exception as e:
      print(f"[ERROR] PDF {pdf_file}: {e}")

  # -------------------------
  # TXT FILES
  # -------------------------
  txt_files = list(data_path.glob("**/*.txt"))[:MAX_FILES_PER_TYPE]

  for txt_file in txt_files:
    try:
      loader = TextLoader(str(txt_file), encoding="utf-8")
      documents.extend(loader.load())
    except Exception as e:
      print(f"[ERROR] TXT {txt_file}: {e}")

  # -------------------------
  # CSV FILES
  # -------------------------
  csv_files = list(data_path.glob("**/*.csv"))[:MAX_FILES_PER_TYPE]

  for csv_file in csv_files:
    try:
      loader = CSVLoader(str(csv_file))
      documents.extend(loader.load())
    except Exception as e:
      print(f"[ERROR] CSV {csv_file}: {e}")

  # -------------------------
  # EXCEL FILES
  # -------------------------
  xlsx_files = list(data_path.glob("**/*.xlsx"))[:MAX_FILES_PER_TYPE]

  for xlsx_file in xlsx_files:
    try:
      loader = UnstructuredExcelLoader(str(xlsx_file))
      documents.extend(loader.load())
    except Exception as e:
      print(f"[ERROR] Excel {xlsx_file}: {e}")

  # -------------------------
  # WORD FILES
  # -------------------------
  docx_files = list(data_path.glob("**/*.docx"))[:MAX_FILES_PER_TYPE]

  for docx_file in docx_files:
    try:
      loader = Docx2txtLoader(str(docx_file))
      documents.extend(loader.load())
    except Exception as e:
      print(f"[ERROR] DOCX {docx_file}: {e}")

  # -------------------------
  # JSON FILES
  # -------------------------
  json_files = list(data_path.glob("**/*.json"))[:MAX_FILES_PER_TYPE]

  for json_file in json_files:
    try:
      loader = JSONLoader(
        file_path=str(json_file),
        jq_schema=".",
        text_content=True
      )
      documents.extend(loader.load())
    except Exception as e:
      print(f"[ERROR] JSON {json_file}: {e}")

  # -------------------------
  # MDB FILES (STRICT FILTER)
  # -------------------------
  mdb_files = list(data_path.glob("**/*.mdb"))[:3]   # 🔥 only 3 DBs
  print(f"[DEBUG] Using {len(mdb_files)} MDB files")

  IMPORTANT_TABLES = ["narratives", "events", "Findings"]
  MAX_ROWS_PER_TABLE = 80   # 🔥 reduced

  for mdb_file in mdb_files:
    print(f"[DEBUG] Loading MDB: {mdb_file}")

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
          df = pd.read_sql(
            f"SELECT TOP {MAX_ROWS_PER_TABLE} * FROM [{table_name}]",
            conn
          )

          for _, row in df.iterrows():
            row_text = " | ".join(
              [str(val) for val in row.values if pd.notna(val)]
            )

            documents.append(
              Document(
                page_content=row_text,
                metadata={
                  "source": str(mdb_file),
                  "table": table_name
                }
              )
            )

          print(f"[DEBUG] Loaded {len(df)} rows from {table_name}")

        except Exception as table_error:
          print(f"[ERROR] Table {table_name}: {table_error}")

      conn.close()

    except Exception as e:
      print(f"[ERROR] MDB {mdb_file}: {e}")

  # 🔥 FINAL GLOBAL LIMIT (VERY IMPORTANT)
  MAX_DOCS = 2000
  documents = documents[:MAX_DOCS]

  print(f"[INFO] Final document count: {len(documents)}")

  return documents