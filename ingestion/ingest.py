"""
Loads a PDF, chunks it, embeds it with OpenAI, and stores in FAISS or Chroma.
Run once per document:
    python -m ingestion.ingest --pdf path/to/document.pdf
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 150))
VECTOR_STORE    = os.getenv("VECTOR_STORE", "faiss")   # faiss | chroma

FAISS_INDEX_PATH  = "data/faiss_index"
CHROMA_PERSIST_DIR = "data/chroma_db"

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page with metadata."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text,
                    "metadata": {
                        "source": Path(pdf_path).name,
                        "page": i + 1,
                    },
                })
    print(f"[Ingest] Extracted {len(pages)} pages from '{pdf_path}'")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split pages into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        splits = splitter.create_documents(
            texts=[page["text"]],
            metadatas=[page["metadata"]],
        )
        chunks.extend(splits)
    print(f"[Ingest] Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks):
    """Embed chunks and persist to FAISS or Chroma."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if VECTOR_STORE == "chroma":
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        print(f"[Ingest] Saved Chroma DB → {CHROMA_PERSIST_DIR}")
    else:
        store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        store.save_local(FAISS_INDEX_PATH)
        print(f"[Ingest] Saved FAISS index → {FAISS_INDEX_PATH}")

    return store


def load_vector_store():
    """Load an existing vector store from disk."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if VECTOR_STORE == "chroma":
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
    else:
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest(pdf_path: str):
    pages  = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    build_vector_store(chunks)
    print("[Ingest] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to ingest")
    args = parser.parse_args()
    ingest(args.pdf)