
"""
LangGraph ReAct-style agent with three nodes:
  1. retrieve     – dense retrieval + cross-encoder rerank from PDF vector store
  2. decide       – LLM judges if retrieved context is sufficient
  3. web_fallback – Tavily search on allowed government domains (if needed)
  4. answer       – Final grounded answer generation with strict anti-hallucination prompt
 
State flows:  retrieve → decide → answer
                              └──→ web_fallback → answer
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from retrieval.retriever import retrieve_and_rerank, format_context, RetrievedChunk
from tools.web_search import web_search, format_web_results

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
SYSTEM_PROMPT = """You are a precise, factual assistant that answers questions strictly about \
immigration, visa processes, and USCIS-related government topics.
 
STRICT RULES — follow every rule without exception:
1. Answer ONLY using the provided context (PDF excerpts and/or official government web results).
2. Do NOT generate, invent, extrapolate, or infer any information not explicitly present in the context.
3. If the context does not contain enough information to answer, respond with exactly:
   "I don't have sufficient information in the provided documents or official government sources to answer this question. \
Please refer directly to uscis.gov or dhs.gov."
4. Do NOT answer questions outside immigration/USCIS scope — politely decline and redirect.
5. Always cite the source (file name + page number, or URL) for every factual claim.
6. Use plain, clear language. Do not add disclaimers beyond citing sources.
"""

class AgentState(TypedDict):
    query: str
    pdf_chunks: list[RetrievedChunk]
    web_results: list[dict]
    pdf_context: str
    web_context: str
    answer: str
    needs_web: bool

def node_retrieve(state: AgentState, vectorstore) -> AgentState:
    """Node 1: Retrieve from PDF vector store."""
    chunks = retrieve_and_rerank(state["query"], vectorstore)
    return {
        **state,
        "pdf_chunks": chunks,
        "pdf_context": format_context(chunks),
    }

def node_decide(state: AgentState) -> AgentState:
    """
    Node 2: Ask LLM if PDF context is sufficient.
    Uses a lightweight, cheap call (gpt-4o-mini) to save cost.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    decision_prompt = f"""You are a relevance judge.
    Given a user question and retrieved context, decide if the context contains \
    enough specific information to answer the question accurately.
    
    Question: {state['query']}
    
    Retrieved context:
    {state['pdf_context'] or '(none)'}
    
    Respond with EXACTLY one word: SUFFICIENT or INSUFFICIENT"""
    response = llm.revoke([HumanMessage(content=decision_prompt)])
    verdict = response.content.strip().upper()
    needs_web = "INSUFFICIENT" in verdict
    print(f"[Decide] Context verdict: {verdict} → needs_web={needs_web}")
    return {**state, "needs_web": needs_web}

def node_web_fallback(state: AgentState) -> AgentState:
    """Node 3 (conditional): Search official government websites."""
    results = web_search(state["query"])
    return {
        **state,
        "web_results": results,
        "web_context": format_web_results(results),
    }

def node_answer(state: AgentState) -> AgentState:
    """Node 4: Generate the final grounded answer."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    # Build context block from whatever sources we have
    context_parts = []
    if state.get("pdf_context"):
        context_parts.append("=== PDF Document Context ===\n" + state["pdf_context"])
    if state.get("web_context"):
        context_parts.append("=== Official Government Website Context ===\n" + state["web_context"])
    full_context = "\n\n".join(context_parts) if context_parts else "(no context available)"
    user_message = f"""Context:
    {full_context}
    
    Question: {state['query']}
    
    Answer (cite every source):"""
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    return {**state, "answer": response.content}

def route_after_decide(state: AgentState) -> Literal["web_fallback", "answer"]:
    return "web_fallback" if state["needs_web"] else "answer"


def build_rag_graph(vectorstore):
    """Build and compile the LangGraph RAG workflow."""
    # Wrap node_retrieve to close over vectorstore
    def retrieve(state):
        return node_retrieve(state, vectorstore)
    
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", node_decide)
    graph.add_node("web_fallback", node_web_fallback)
    graph.add_node("answer", node_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {"web_fallback": "web_fallback", "answer": "answer"},
    )
    graph.add_edge("web_fallback", "answer")
    graph.add_edge("answer", END)

    return graph.compile()

def ask(query:str, vectorstore) -> dict:
    """Run the full RAG pipeline for a single query."""
    app = build_rag_graph(vectorstore)

    initial_state: AgentState = {
        "query": query,
        "pdf_chunks": [],
        "web_results": [],
        "pdf_context": "",
        "web_context": "",
        "answer": "",
        "needs_web": False,
    }

    final_state = app.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "pdf_source": [
            {"source": c.source, "page": c.page, "score": round(c.score, 3)}
            for c in final_state["pdf_chunks"]
        ],
            "web_sources": [
                {
                    "title": r["title"], "url": r["url"]
                }
                for r in final_state["web_results"]
            ],
                "used_web_fallback": final_state["needs_web"],
    }