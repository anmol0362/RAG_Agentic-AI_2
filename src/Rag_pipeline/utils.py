import json
from typing import Any

TRACE = True


def trace(title: str, obj: Any = None):
    if not TRACE:
        return

    print("\n" + "=" * 100)
    print(f"[TRACE] {title}")
    print("=" * 100)

    if obj is not None:
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=2, ensure_ascii=False)[:8000])
        else:
            print(obj)