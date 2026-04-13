import os
from openai import AzureOpenAI

from src.Rag_pipeline.observability import (
    print_prompt_observability,
    print_llm_usage_observability,
    print_answer_observability,
)

# CONFIG
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")  # ← changed

TEMPERATURE = 0.2
MAX_TOKENS  = 1200

# CLIENT
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# SYSTEM PROMPT
BASE_SYSTEM_PROMPT = """You are an expert aviation safety analyst assistant.
You answer questions strictly based on the provided document context.

STRICT RULES:
- Answer ONLY about the PRIMARY accident being investigated in the report
- Do NOT mix information from referenced or comparison accidents
- If the context mentions multiple accidents, focus only on the main one
- If the answer is not in the context, say "I don't have enough information"
- Be precise with numbers, dates, and names
- Cite page numbers when available"""


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    # Combine base prompt with any additional system prompt
    full_system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{system_prompt}".strip()

    print_prompt_observability(full_system_prompt, user_prompt)

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": full_system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )

    answer = response.choices[0].message.content

    print_llm_usage_observability(
        model=CHAT_DEPLOYMENT,
        finish_reason=response.choices[0].finish_reason,
        prompt_tokens=getattr(response.usage, "prompt_tokens", None),
        completion_tokens=getattr(response.usage, "completion_tokens", None),
        total_tokens=getattr(response.usage, "total_tokens", None)
    )

    print_answer_observability(answer)

    return answer