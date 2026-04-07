import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any


HASH_REGISTRY_PATH = "data/cleaned/file_hash_registry.json"


def ensure_registry_exists():
    Path(HASH_REGISTRY_PATH).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(HASH_REGISTRY_PATH):
        with open(HASH_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def load_hash_registry() -> Dict[str, Any]:
    ensure_registry_exists()

    with open(HASH_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_hash_registry(registry: Dict[str, Any]):
    with open(HASH_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def is_duplicate_file(file_path: str) -> bool:
    registry = load_hash_registry()
    file_hash = compute_file_hash(file_path)

    return file_hash in registry


def register_file(file_path: str):
    registry = load_hash_registry()
    file_hash = compute_file_hash(file_path)

    registry[file_hash] = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path
    }

    save_hash_registry(registry)


def get_file_hash(file_path: str) -> str:
    return compute_file_hash(file_path)