import os
from typing import List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = "MiniMaxAI/MiniMax-M2"


client = InferenceClient(model=HF_MODEL_ID, token=HF_API_TOKEN)


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
    resp = client.text_generation(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    print(f"Response: {resp}")
    return resp
