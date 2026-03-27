"""
Streamlit frontend — upload PDF, ask questions, see sourced answers.
Run:  streamlit run ui/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingestion.ingest import ingest, load_vector_store
from agent.rag_agent import ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Immigration RAG Assistant",
    page_icon="🗂️",
    layout="wide",
)

st.title("🗂️ Immigration Document Assistant")
st.caption(
    "Ask questions about your immigration document. "
    "If the answer isn't in the PDF, I'll search official government websites "
    "(uscis.gov, travel.state.gov, dhs.gov, etc.)."
)

# ── Session state ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {role, content}

# ── Sidebar: PDF upload ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Load Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded and st.button("Ingest PDF"):
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        with st.spinner("Embedding document…"):
            ingest(tmp_path)
            st.session_state.vectorstore = load_vector_store()
            st.session_state.chat_history = []  # reset chat for new doc
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        st.success(f"✅ '{uploaded.name}' ingested!")

    # Try loading existing index
    if st.session_state.vectorstore is None:
        try:
            st.session_state.vectorstore = load_vector_store()
            st.info("Existing index loaded.")
        except Exception:
            st.warning("No index found. Please upload a PDF above.")

    st.divider()
    st.markdown(
        "**Allowed web fallback domains:**\n"
        "- uscis.gov\n- travel.state.gov\n- dhs.gov\n"
        "- state.gov\n- ice.gov\n- cbp.gov"
    )

# ── Chat interface ────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about your document…", disabled=st.session_state.vectorstore is None)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            result = ask(user_input, st.session_state.vectorstore)

        answer = result["answer"]
        st.markdown(answer)

        # Show source expanders
        if result["pdf_sources"]:
            with st.expander(f"📎 PDF Sources ({len(result['pdf_sources'])})"):
                for s in result["pdf_sources"]:
                    st.markdown(f"- **{s['source']}** — page {s['page']}  *(relevance: {s['score']})*")

        if result["web_sources"]:
            with st.expander(f"🌐 Web Sources ({len(result['web_sources'])}) — government sites only"):
                for s in result["web_sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")

        if result["used_web_fallback"]:
            st.info("ℹ️ PDF didn't have enough information — answer supplemented from official government websites.")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})