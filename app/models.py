from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class IndexResponse(BaseModel):
    doc_id: str
    chunks: int
    pages: int

class AskRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    top_k: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]