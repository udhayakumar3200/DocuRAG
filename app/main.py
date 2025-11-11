import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .rag import index_pdf, search
from .llm import generate_answer
from .models import IndexResponse, AskRequest, AskResponse
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="PDF QA (RAG) — HF API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/index", response_model=IndexResponse)
async def index_endpoint(file: UploadFile = File(...)):
    out_path = UPLOAD_DIR / file.filename
    with out_path.open("wb") as f:
        f.write(await file.read())
    meta = index_pdf(str(out_path))
    return IndexResponse(**meta)


@app.post("/ask")
async def ask_endpoint(payload: AskRequest):
    results = search(payload.question, payload.doc_id, payload.top_k)
    answer = generate_answer(results["documents"], payload.question)
    return {"answer": answer}


@app.get("/")
async def root():
    return JSONResponse({"ok": True, "message": "PDF QA server up (HF)."})
