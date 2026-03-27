"""
FastAPI server exposing:
  POST /ingest   upload and ingest a PDF
  POST /query    ask a question
  GET  /health   health check
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ingestion.ingest import ingest, load_vector_store
from agent.rag_agent import ask

load_dotenv()

app = FastAPI(
    title="Immigration RAG API",
    description="Grounded Q&A from PDF documents + official government websites",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global vector store (loaded once) ────────────────────────────────────────

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        try:
            _vectorstore = load_vector_store()
            print("[API] Vector store loaded from disk.")
        except Exception:
            print("[API] No vector store found. Please ingest a PDF first.")
    return _vectorstore


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer:             str
    pdf_sources:        list[dict]
    web_sources:        list[dict]
    used_web_fallback:  bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "vector_store_ready": get_vectorstore() is not None}


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF and build/update the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        ingest(tmp_path)
        global _vectorstore
        _vectorstore = load_vector_store()   # reload after ingestion
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"message": f"'{file.filename}' ingested successfully."}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ask a question against the ingested PDF (with web fallback)."""
    vs = get_vectorstore()
    if vs is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Please POST a PDF to /ingest first.",
        )

    result = ask(req.question, vs)
    return QueryResponse(**result)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)