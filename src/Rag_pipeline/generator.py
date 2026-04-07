import os
from openai import AzureOpenAI

from src.Rag_pipeline.observability import (
    print_prompt_observability,
    print_llm_usage_observability,
    print_answer_observability,
)

# CONFIG
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

TEMPERATURE = 0.2
MAX_TOKENS = 1200

# CLIENT
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    print_prompt_observability(system_prompt, user_prompt)

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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