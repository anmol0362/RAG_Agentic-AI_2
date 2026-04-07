from typing import Dict, Any
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from src.Rag_pipeline.observability import langfuse

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


def evaluate_rag(query: str, context: str, answer: str, trace_obj=None) -> Dict[str, Any]:
    eval_prompt = f"""
You are evaluating a RAG system.

Score the answer from 1 to 5 on:

1. answer_relevance → Did it answer the user's question?
2. faithfulness → Is the answer grounded in the provided context?
3. context_relevance → Was the retrieved context useful?
4. completeness → Did the answer cover the important parts?

Return ONLY valid JSON in this format:
{{
  "answer_relevance": 0,
  "faithfulness": 0,
  "context_relevance": 0,
  "completeness": 0,
  "notes": "short explanation"
}}

QUESTION:
{query}

CONTEXT:
{context[:12000]}

ANSWER:
{answer}
"""

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a strict RAG evaluator. Output ONLY valid JSON."
            },
            {"role": "user", "content": eval_prompt}
        ]
    )

    import json
    print("\n" + "=" * 110)
    print("RAW EVAL RESPONSE")
    print("=" * 110)
    print(response.choices[0].message.content)

    result = json.loads(response.choices[0].message.content)

    if trace_obj:
        trace_obj.score(name="answer_relevance", value=result["answer_relevance"])
        trace_obj.score(name="faithfulness", value=result["faithfulness"])
        trace_obj.score(name="context_relevance", value=result["context_relevance"])
        trace_obj.score(name="completeness", value=result["completeness"])

    langfuse.flush()
    return result