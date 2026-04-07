import re
import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from torch import chunk

#from src.Chunking.chunk_records import extract_section_from_chunk

# =========================================================
# WHY THIS FILE EXISTS:
# After extraction, each record has raw text.
# This file splits that text into smaller "chunks"
# that are the right size for embedding and retrieval.
#
# NEW: We now route by content_type:
#   content_type="text"  → paragraph chunker (split into pieces)
#   content_type="table" → table chunker (keep each row as-is)
#   content_type="mdb"   → mdb chunker (structured row data)
# =========================================================

# =========================================================
# CHUNKING PARAMETERS
# Tune these to change chunk size and overlap
# =========================================================
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*\d+(\.\d+)*\s+[A-Z][^\n]{3,80}$",
    re.MULTILINE
)

def extract_section_from_chunk(chunk_text: str, fallback_section: str) -> str:
    for line in chunk_text.split("\n"):
        line = line.strip()
        if SECTION_HEADER_PATTERN.match(line):
            return line
    return fallback_section

MAX_CHUNK_SIZE = 1200        # max characters per chunk
CHUNK_OVERLAP = 150          # overlap between consecutive chunks
MIN_CHUNK_LENGTH = 80        # discard chunks smaller than this
OVERSIZED_BLOCK_THRESHOLD = 1200

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# -------------------------
# FALLBACK SPLITTER
# Used when a single paragraph is too large
# LangChain's splitter handles it with multiple separator levels
# -------------------------
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS
)


# =========================================================
# HASH FUNCTION
# Used to detect and skip duplicate chunks
# SHA256 gives a unique fingerprint for each chunk's text
# =========================================================

def compute_chunk_hash(text: str) -> str:
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# =========================================================
# TEXT NORMALIZATION
# Clean up whitespace before chunking
# =========================================================

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 newlines
    text = re.sub(r"[ \t]+", " ", text)     # collapse spaces
    return text.strip()


# =========================================================
# BLOCK SPLITTER
# Splits text into paragraph blocks using double newlines
# Each paragraph becomes a "block" that we then merge
# =========================================================

def split_into_blocks(text: str) -> List[str]:
    text = normalize_text(text)
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


# =========================================================
# OVERLAP APPLIER
# Adds the last 150 chars of previous chunk to start of next.
# Why? So context isn't lost at chunk boundaries.
# Example: if chunk 1 ends with "...the captain pulled back"
#          chunk 2 starts with "...the captain pulled back\n\n[new content]"
# =========================================================

def apply_overlap(chunks: List[str]) -> List[str]:
    if not chunks:
        return []

    final_chunks = [chunks[0]]

    for i in range(1, len(chunks)):
        prev = final_chunks[-1]
        curr = chunks[i]
        overlap = prev[-CHUNK_OVERLAP:] if len(prev) > CHUNK_OVERLAP else prev
        merged = f"{overlap}\n\n{curr}".strip()
        final_chunks.append(merged)

    return final_chunks


# =========================================================
# DEDUPLICATION
# Removes exact duplicate chunks using SHA256 hashing.
# Duplicates can occur when the same text appears on
# multiple pages (e.g. headers, repeated sections).
# =========================================================

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    seen = set()
    unique = []
    for chunk in chunks:
        h = compute_chunk_hash(chunk)
        if h in seen:
            print(f"[SKIP DUPLICATE CHUNK] {chunk[:80]}...")
            continue
        seen.add(h)
        unique.append(chunk)
    return unique


# =========================================================
# PDF TEXT CHUNKER
# Used for content_type="text" records.
# Strategy:
#   1. Split by paragraphs
#   2. Merge small paragraphs together up to MAX_CHUNK_SIZE
#   3. Split oversized paragraphs with fallback splitter
#   4. Apply overlap between chunks
#   5. Deduplicate
# =========================================================

def chunk_pdf_text(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = str(record.get("text", "") or "").strip()
    if not text:
        return []

    blocks = split_into_blocks(text)
    chunks = []
    current = ""

    for block in blocks:
        candidate = f"{current}\n\n{block}".strip() if current else block

        if len(candidate) <= MAX_CHUNK_SIZE:
            # Block fits → keep merging
            current = candidate
        else:
            if current:
                chunks.append(current.strip())

            if len(block) > OVERSIZED_BLOCK_THRESHOLD:
                # Single paragraph too big → use fallback splitter
                sub_chunks = fallback_splitter.split_text(block)
                chunks.extend([
                    c.strip() for c in sub_chunks
                    if len(c.strip()) >= MIN_CHUNK_LENGTH
                ])
                current = ""
            else:
                current = block

    if current:
        chunks.append(current.strip())

    # Apply sliding window overlap
    chunks = apply_overlap(chunks)

    # Remove tiny chunks
    chunks = [c for c in chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]

    # Remove duplicates
    chunks = deduplicate_chunks(chunks)

    # Build output records
    output = []
    for i, chunk in enumerate(chunks, start=1):
        output.append({
            "chunk_id":           f"{record.get('doc_id')}_{i}",
            "doc_id":             record.get("doc_id"),
            "source":             record.get("source"),
            "file_type":          record.get("file_type"),
            "page":               record.get("page"),
            "table":              record.get("table"),
            "extraction_method":  record.get("extraction_method"),
            "chunk_type":         "structure_text",
            "content_type":       record.get("content_type", "text"),  # NEW
            "section":            record.get("section", ""),            # NEW
            "chunk_size":         len(chunk),
            "overlap":            CHUNK_OVERLAP,
            "text":               chunk
        })

    return output


# =========================================================
# TABLE CHUNKER
# Used for content_type="table" records.
#
# WHY different from text chunker?
# Tables are already row-by-row from data_loader.py.
# Each row is self-contained structured data like:
#   "Injuries: 2 | Fatalities: 0 | Aircraft: DHC-8"
# We DON'T want to split or merge these rows.
# We DON'T want overlap (rows are independent).
# We just validate and pass them through as-is.
# =========================================================

def chunk_table_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = str(record.get("text", "") or "").strip()

    if not text or len(text) < 20:
        return []

    # Table rows are kept as single chunks — no splitting needed
    return [{
        "chunk_id":           f"{record.get('doc_id')}_1",
        "doc_id":             record.get("doc_id"),
        "source":             record.get("source"),
        "file_type":          record.get("file_type"),
        "page":               record.get("page"),
        "table":              record.get("table"),
        "extraction_method":  record.get("extraction_method", "table_extraction"),
        "chunk_type":         "table_row",    # distinct chunk_type for tables
        "content_type":       "table",        # NEW
        "section":            record.get("section", ""),        "chunk_size":         len(text),
        "overlap":            0,              # no overlap for table rows
        "text":               text
    }]


# =========================================================
# MDB CHUNKER
# Used for Microsoft Access Database records.
# Each MDB row has structured_data (field: value pairs).
# We convert that to a readable text string and keep as 1 chunk.
# =========================================================

def normalize_field_name(field: str) -> str:
    return field.replace("_", " ").strip().title()


def build_structured_text(structured_data: Dict[str, Any]) -> str:
    parts = []
    for key, value in structured_data.items():
        if value is None or str(value).strip() == "":
            continue
        key_name = normalize_field_name(key)
        parts.append(f"{key_name}: {value}")
    return ". ".join(parts).strip() + "." if parts else ""


def chunk_mdb_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    structured_data = record.get("structured_data", {}) or {}
    base_text = str(record.get("text", "") or "").strip()

    text = build_structured_text(structured_data)
    if not text:
        text = base_text
    if not text:
        return []

    return [{
        "chunk_id":           f"{record.get('doc_id')}_1",
        "doc_id":             record.get("doc_id"),
        "source":             record.get("source"),
        "file_type":          record.get("file_type"),
        "page":               record.get("page"),
        "table":              record.get("table"),
        "chunk_type":         "row",
        "content_type":       "table",   # MDB rows are table data
        "section":            "",
        "chunk_size":         len(text),
        "overlap":            0,
        "text":               text,
        "structured_data":    structured_data
    }]


# =========================================================
# ROUTER
# This is the main function called by build_chunks.py.
# It decides which chunker to use based on:
#   1. content_type field (NEW — from data_loader.py)
#   2. file_type field (fallback for MDB)
#
# Routing logic:
#   file_type="mdb"          → mdb chunker
#   content_type="table"     → table chunker (keep rows as-is)
#   everything else          → text chunker (split into pieces)
# =========================================================

def chunk_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    file_type    = str(record.get("file_type", "") or "").lower()
    content_type = str(record.get("content_type", "") or "").lower()

    # MDB rows → structured field chunker
    if file_type == "mdb":
        return chunk_mdb_record(record)

    # PDF/CSV table rows → keep as-is, no splitting
    if content_type == "table":
        return chunk_table_record(record)

    # Everything else → paragraph text chunker
    return chunk_pdf_text(record)