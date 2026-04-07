SYSTEM_PROMPT = """
You are an aviation safety assistant.

Rules:
- Use ONLY the provided context.
- Do NOT hallucinate.
- If unsure → say "insufficient data".
- Prefer procedural clarity.
- Keep answers structured.
"""


def build_user_prompt(query: str, context: str) -> str:
    return f"""
USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Provide a grounded answer using ONLY the context above.
""".strip()