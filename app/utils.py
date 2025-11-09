from pathlib import Path
from typing import List, Tuple
import pdfplumber


def extract_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def chunk_text(
    text: list[str],
    chunk_size: int = 450,
    overlap: int = 60,
    force_process: bool = False,
) -> dict:
    # Combine all text strings in the list
    combined_text = " ".join(text)
    words = combined_text.split()
    n = len(words)

    # If less than 100 words, mark as small chunk to combine with next page
    # Unless force_process is True (for end-of-file case)
    if n < 100 and not force_process:
        return {
            "text": text,  # Return original list to accumulate
            "small_chunk": True,
            "words_length": n,
            "chunks": [],
        }

    # If less than 100 words but force_process is True, return as single chunk
    if n < 100 and force_process:
        return {
            "text": [combined_text],  # Return as single chunk
            "small_chunk": False,
            "words_length": n,
            "chunks": [combined_text],
        }

    # If less than chunk_size, return as single chunk
    if n < chunk_size:
        return {
            "text": [combined_text],  # Return as single chunk
            "small_chunk": False,
            "words_length": n,
            "chunks": [combined_text],
        }

    # Otherwise, create chunks with overlap
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return {
        "text": chunks,  # Return list of chunks
        "small_chunk": False,
        "words_length": n,
        "chunks": chunks,
    }


# while start < n:
#     end = min(start + chunk_size, n)
#     chunk = " ".join(words[start:end]).strip()
#     if chunk:
#         chunks.append(chunk)
#     if end == n:
#         break
#     start = max(0, end - overlap)
#     print(f"Chunks: {chunks}")
#     return chunks
