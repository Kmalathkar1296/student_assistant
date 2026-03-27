"""
Two-stage retrieval:
  1. Dense retrieval  – top-K from FAISS/Chroma via cosine similarity
  2. Cross-encoder rerank – reorder by relevance, keep top-N for LLM
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 10))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 4))

@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: Optional[int]
    score: float

def dense_retrieve(query: str, vectorstore, k: int = TOP_K_RETRIEVE) -> list[Document]:
    """
    Hybrid retrieval: MMR (Maximal Marginal Relevance) to balance
    relevance with diversity and reduce redundant chunks.
    """
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k = k,
        fetch_k = k*3,
        lambda_mul = 0.6,
    )
    return docs

_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        # Lightweight but effective model; swap for ms-marco-MiniLM-L-12-v2 for higher accuracy
        __reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[Retriever] Cross-encoder loaded.")
    return _reranker

def rerank(query: str, docs: list[Document], top_n: int = TOP_K_RERANK) -> list[RetrievedChunk]:
    """Score (query, chunk) pairs with a cross-encoder and return top_n."""
    reranker = get_reranker()
    pairs = [(query, docs.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, pairs),
        key = lambda x: x[1],
        reverse = True,
    )[:top_n]
    
    return [
        RetrievedChunk(
            content = doc.page_content,
            source = doc.metadata.get("source", "pdf"),
            page = doc.metadata.get("page"),
            score = float(score),
        ) 
    for doc, score in ranked
    ]

def retrieve_and_rerank(
        query:str, 
        vectorstore, 
        k_retrieval: int = TOP_K_RETRIEVE, 
        k_rerank: int = TOP_K_RERANK) -> list[RetrievedChunk]:
    """Full pipeline: dense retrieval → MMR → cross-encoder rerank."""

    docs = dense_retrieve(query, vectorstore, k=k_retrieval)
    chunks = rerank(query, docs, top_n=k_rerank)
    print(f"[Retriever] {len(docs)} retrieved → {len(chunks)} after rerank")
    return chunks

def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks into a clean string for the prompt context window."""
    parts = []
    for i, c in enumerate(chunks, 1):
        loc = f"page {c.page}" if c.page else "unknown page"
        parts.append(
            f"[Source {i}: {c.source}, {loc}  |  relevance score: {c.score:.3f}]\n{c.content}"
        )
    return "\n\n---\n\n".join(parts)
