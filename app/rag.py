import os
import uuid
from typing import Dict, List, Tuple
from pathlib import Path


import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np


from .utils import extract_pdf_text, chunk_text


CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TOP_K = int(os.getenv("TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))


# _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
import chromadb

_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT_ID"),
    database=os.getenv("CHROMA_DATABASE"),
)
_collection = _client.get_or_create_collection(name="pdf_qa")


# def _embed_texts(texts: List[str]) -> List[List[float]]:
#     embs = _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
#     if hasattr(embs, "tolist"):
#         embs = embs.tolist()
#     return embs


def index_pdf(pdf_path: str, doc_id: str | None = None) -> Dict:
    doc_id = doc_id or str(uuid.uuid4())
    pages: List[Tuple[int, str]] = extract_pdf_text(pdf_path)

    ids, documents, metadatas = [], [], []

    small_text = []
    for page_num, text in pages:
        if not text.strip():
            continue
        # if page_num == 6:
        #     break

        # Combine current page with accumulated small pages
        text_to_chunk = small_text + [text] if small_text else [text]

        chunks_result = chunk_text(
            text_to_chunk,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        if chunks_result["small_chunk"] == True:
            # Page is too small, accumulate for next iteration
            small_text = text_to_chunk
            print(
                f"Page {page_num} has {chunks_result['words_length']} words - accumulating for next page -- {chunks_result['text']}"
            )
            continue
        else:
            # Process the chunks (either single chunk or multiple chunks)
            small_text.clear()  # Clear accumulated small pages

            # Get the actual chunks to process
            actual_chunks = chunks_result["chunks"]
            print(
                f"Processing page {page_num} into {len(actual_chunks)} chunk(s) ({chunks_result['words_length']} words total) -- {chunks_result['text']}"
            )

            # Add chunks to documents
            for i, chunk in enumerate(actual_chunks):
                cid = f"{doc_id}_p{page_num}_c{i}"
                ids.append(cid)
                documents.append(chunk)
                metadatas.append(
                    {
                        "doc_id": doc_id,
                        "page": page_num,
                        "chunk": i,
                        "source": Path(pdf_path).name,
                    }
                )

    # Handle any remaining accumulated small pages at the end
    if small_text:
        chunks_result = chunk_text(
            small_text, CHUNK_SIZE, CHUNK_OVERLAP, force_process=True
        )
        actual_chunks = chunks_result["chunks"]
        if actual_chunks:
            print(
                f"Processing accumulated small pages into {len(actual_chunks)} chunk(s) ({chunks_result['words_length']} words total)"
            )
            last_page_num = len(pages) - 1 if pages else 0
            for i, chunk in enumerate(actual_chunks):
                cid = f"{doc_id}_p{last_page_num}_c{i}"
                ids.append(cid)
                documents.append(chunk)
                metadatas.append(
                    {
                        "doc_id": doc_id,
                        "page": last_page_num,
                        "chunk": i,
                        "source": Path(pdf_path).name,
                    }
                )

    if not documents:
        raise ValueError("No text found in PDF. Add OCR if it's scanned.")

    # embeddings = _embed_texts(documents)
    # _collection.upsert(
    #     ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    # )
    print(f"Documents: {len(documents)}")

    _collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return {"doc_id": doc_id, "chunks": len(documents), "pages": len(pages)}


def search(query: str, doc_id: str | None = None, k: int | None = None):
    results = _collection.query(
        query_texts=[query],
        n_results=k,
        where={"doc_id": doc_id},
        include=["metadatas", "documents", "distances"],
    )
    print(f"Results: {results}")
    return results
