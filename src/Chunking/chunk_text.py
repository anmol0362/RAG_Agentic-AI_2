import re
import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# PARENT-CHILD CHUNKING STRATEGY
# Child chunks → small, used for retrieval (semantic search)
# Parent chunks → large, sent to LLM for full context
# Each child stores parent_id to fetch parent at query time
# =========================================================

SECTION_HEADER_PATTERN = re.compile(
    r"^\s*\d+(\.\d+)*\s+[A-Z][^\n]{3,80}$",
    re.MULTILINE
)

# Parent = large context window
PARENT_CHUNK_SIZE = 3000
PARENT_OVERLAP    = 200

# Child = small retrieval unit
CHILD_CHUNK_SIZE  = 400
CHILD_OVERLAP     = 50
MIN_CHUNK_LENGTH  = 80

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_OVERLAP,
    separators=SEPARATORS
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_OVERLAP,
    separators=SEPARATORS
)


def compute_chunk_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_section_from_chunk(chunk_text: str, fallback_section: str) -> str:
    for line in chunk_text.split("\n"):
        line = line.strip()
        if SECTION_HEADER_PATTERN.match(line):
            return line
    return fallback_section


def deduplicate_chunks(chunks: List[str]) -> List[str]:
    seen = set()
    unique = []
    for chunk in chunks:
        h = compute_chunk_hash(chunk)
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique


# =========================================================
# PARENT-CHILD TEXT CHUNKER
# =========================================================

def chunk_pdf_text(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = str(record.get("text", "") or "").strip()
    if not text:
        return []

    text    = normalize_text(text)
    doc_id  = record.get("doc_id")
    section = record.get("section", "")

    # Step 1: Split into PARENT chunks
    parent_texts = parent_splitter.split_text(text)
    parent_texts = deduplicate_chunks(parent_texts)

    output = []

    for p_idx, parent_text in enumerate(parent_texts, start=1):
        if len(parent_text.strip()) < MIN_CHUNK_LENGTH:
            continue

        parent_id      = f"{doc_id}_parent_{p_idx}"
        parent_section = extract_section_from_chunk(parent_text, section)

        # Store PARENT chunk
        output.append({
            "chunk_id":          parent_id,
            "parent_id":         None,
            "chunk_level":       "parent",
            "doc_id":            doc_id,
            "source":            record.get("source"),
            "file_type":         record.get("file_type"),
            "page":              record.get("page"),
            "table":             record.get("table"),
            "extraction_method": record.get("extraction_method"),
            "chunk_type":        "parent_text",
            "content_type":      record.get("content_type", "text"),
            "section":           parent_section,
            "chunk_size":        len(parent_text),
            "overlap":           PARENT_OVERLAP,
            "text":              parent_text.strip()
        })

        # Step 2: Split PARENT into CHILD chunks
        child_texts = child_splitter.split_text(parent_text)
        child_texts = [c.strip() for c in child_texts if len(c.strip()) >= MIN_CHUNK_LENGTH]
        child_texts = deduplicate_chunks(child_texts)

        for c_idx, child_text in enumerate(child_texts, start=1):
            child_section = extract_section_from_chunk(child_text, parent_section)

            output.append({
                "chunk_id":          f"{parent_id}_child_{c_idx}",
                "parent_id":         parent_id,
                "chunk_level":       "child",
                "doc_id":            doc_id,
                "source":            record.get("source"),
                "file_type":         record.get("file_type"),
                "page":              record.get("page"),
                "table":             record.get("table"),
                "extraction_method": record.get("extraction_method"),
                "chunk_type":        "child_text",
                "content_type":      record.get("content_type", "text"),
                "section":           child_section,
                "chunk_size":        len(child_text),
                "overlap":           CHILD_OVERLAP,
                "text":              child_text
            })

    return output


# =========================================================
# TABLE CHUNKER (unchanged)
# =========================================================

def chunk_table_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = str(record.get("text", "") or "").strip()
    if not text or len(text) < 20:
        return []

    return [{
        "chunk_id":          f"{record.get('doc_id')}_1",
        "parent_id":         None,
        "chunk_level":       "single",
        "doc_id":            record.get("doc_id"),
        "source":            record.get("source"),
        "file_type":         record.get("file_type"),
        "page":              record.get("page"),
        "table":             record.get("table"),
        "extraction_method": record.get("extraction_method", "table_extraction"),
        "chunk_type":        "table_row",
        "content_type":      "table",
        "section":           record.get("section", ""),
        "chunk_size":        len(text),
        "overlap":           0,
        "text":              text
    }]


# =========================================================
# MDB CHUNKER (unchanged)
# =========================================================

def normalize_field_name(field: str) -> str:
    return field.replace("_", " ").strip().title()


def build_structured_text(structured_data: Dict[str, Any]) -> str:
    parts = []
    for key, value in structured_data.items():
        if value is None or str(value).strip() == "":
            continue
        parts.append(f"{normalize_field_name(key)}: {value}")
    return ". ".join(parts).strip() + "." if parts else ""


def chunk_mdb_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    structured_data = record.get("structured_data", {}) or {}
    text = build_structured_text(structured_data) or str(record.get("text", "") or "").strip()
    if not text:
        return []

    return [{
        "chunk_id":        f"{record.get('doc_id')}_1",
        "parent_id":       None,
        "chunk_level":     "single",
        "doc_id":          record.get("doc_id"),
        "source":          record.get("source"),
        "file_type":       record.get("file_type"),
        "page":            record.get("page"),
        "table":           record.get("table"),
        "chunk_type":      "row",
        "content_type":    "table",
        "section":         "",
        "chunk_size":      len(text),
        "overlap":         0,
        "text":            text,
        "structured_data": structured_data
    }]


# =========================================================
# ROUTER
# =========================================================

def chunk_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    file_type    = str(record.get("file_type", "") or "").lower()
    content_type = str(record.get("content_type", "") or "").lower()

    if file_type == "mdb":
        return chunk_mdb_record(record)
    if content_type == "table":
        return chunk_table_record(record)
    return chunk_pdf_text(record)