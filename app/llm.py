import os
from typing import List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)


SYSTEM_PROMPT = (
    "You are a helpful assistant answering strictly from the provided context. "
    "If the answer is not present, say: 'I don't know from the document.' "
    "Keep answers concise and cite page numbers when possible."
)


def format_prompt(context_blocks: List[str], question: str) -> str:
    context = "\n\n".join([f"[CONTEXT]\n{c}" for c in context_blocks])
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"{context}\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[ANSWER]"
    )


def generate_answer(
    context_blocks: List[str],
    question: str,
    max_new_tokens: int = 400,
    temperature: float = 0.2,
) -> str:
    prompt = format_prompt(context_blocks, question)
    # Use text generation (works for instruct/chat models)
    # First API call with reasoning
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": SYSTEM_PROMPT},
        ],
        # extra_body={"reasoning": {"enabled": True}},
        temperature=temperature,
    )
    response = response.choices[0].message.content
    print(f"Response: {response}")
    return response
