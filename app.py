import streamlit as st
from rag.pipeline import ingest, answer
from rag.vectorstore import VectorStore

st.set_page_config(page_title="Chat With Your Documents", page_icon="📄")
st.title("📄 Chat With Your Documents")
st.caption("Document → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer")

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
    st.session_state.messages = []
    st.session_state.files = []

with st.sidebar:
    st.header("1. Upload")
    files = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 800, 100)
    overlap = st.slider("Chunk overlap", 0, 400, 150, 50)
    k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)

    if st.button("Process documents", type="primary") and files:
        with st.spinner("Chunking and embedding..."):
            for f in files:
                if f.name in st.session_state.files:
                    continue
                try:
                    n = ingest(st.session_state.store, f.name, f.read(), chunk_size, overlap)
                    st.session_state.files.append(f.name)
                    st.success(f"{f.name}: {n} chunks")
                except Exception:
                    st.error(f"Could not process {f.name}")

    st.write(f"Chunks in vector DB: **{st.session_state.store.count()}**")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

question = st.chat_input("Ask something about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, hits = answer(st.session_state.store, question, k)
            except Exception:
                reply, hits = "Something went wrong. Please try again.", []
        st.write(reply)
        if hits:
            with st.expander("Sources used"):
                for h in hits:
                    st.markdown(f"**{h['source']}** (chunk {h['chunk']}, similarity {h['score']})")
                    st.caption(h["text"])
    st.session_state.messages.append({"role": "assistant", "content": reply})
st.markdown(
    """
    <style>
    .footer {position: fixed; bottom: 14px; right: 16px; font-size: 20px; color: gray; z-index: 999;}
    </style>
    <div class="footer"><i>Built by Muniswami Y</i></div>
    """,
    unsafe_allow_html=True,
)