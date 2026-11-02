import os
import json
import streamlit as st

from rag.config import gemini_keys, gemini_models, get_masked_keys, EMBED_MODEL, GROQ_MODEL, GROQ_API_KEY
from rag.chunker import chunk_text, chunk_text_with_metadata
from rag.loader import load_file
from rag.pipeline import ingest_text, answer
from rag.vectorstore import VectorStore
from rag.keep_alive import start_keep_alive

# Start 24/7 keep-alive background worker on Render
start_keep_alive()

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Learning Lab | Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Header */
    .hero-container {
        padding: 1.2rem 1.6rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.06) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        margin-bottom: 1.4rem;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #4b5563;
        font-size: 0.98rem;
        margin-bottom: 0.8rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .metric-num {
        font-size: 1.35rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.1rem;
    }

    /* Badges & Tags */
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-high {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-med {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-low {
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
    }
    .badge-source {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }

    /* Chunk Visualizer Cards */
    .chunk-card {
        border: 1px solid #e5e7eb;
        border-left: 4px solid #6366f1;
        background-color: #fafafa;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
        font-family: inherit;
    }
    .chunk-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        color: #6b7280;
    }
    .chunk-text {
        font-size: 0.9rem;
        line-height: 1.5;
        color: #1f2937;
        background-color: #ffffff;
        padding: 0.7rem;
        border-radius: 6px;
        border: 1px solid #f3f4f6;
    }

    /* Step visualizer card */
    .step-box {
        border: 1px solid #e0e7ff;
        background: #ffffff;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .step-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: #3730a3;
        margin-bottom: 0.4rem;
    }

    /* Fixed Footer */
    .footer {
        position: fixed;
        bottom: 12px;
        right: 18px;
        font-size: 0.85rem;
        color: #9ca3af;
        z-index: 999;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(6px);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# State Initialization
# -----------------------------------------------------------------------------
if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files" not in st.session_state:
    st.session_state.files = []
if "file_contents" not in st.session_state:
    st.session_state.file_contents = {}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_sample_file():
    """Load the built-in sample.txt file for instant testing."""
    sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample.txt")
    if not os.path.exists(sample_path):
        sample_path = "samples/sample.txt"
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            content = f.read()
        return "sample.txt", content
    except Exception as e:
        st.error(f"Could not load sample file: {e}")
        return None, None


def reset_database():
    """Clear vector database and conversation history."""
    st.session_state.store.clear()
    st.session_state.messages = []
    st.session_state.files = []
    st.session_state.file_contents = {}
    st.session_state.pending_question = None


# -----------------------------------------------------------------------------
# Hero Banner & Quick Actions
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🧠 RAG Document Intelligence & Learning Lab</div>
        <div class="hero-subtitle">
            Experience and master <b>Retrieval-Augmented Generation</b> (RAG). Upload documents, inspect chunk boundaries, 
            explore vector similarity scores, and see how grounded context prevents LLM hallucinations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Action Bar
c_act1, c_act2, c_act3 = st.columns([2.5, 1.3, 1.2])

with c_act1:
    if st.button("⚡ 1-Click Load Sample Document (sample.txt)", type="primary", use_container_width=True):
        name, content = load_sample_file()
        if content:
            with st.spinner("Chunking, embedding, and indexing sample.txt..."):
                try:
                    num_chunks = ingest_text(st.session_state.store, name, content, chunk_size=800, overlap=150)
                    if name not in st.session_state.files:
                        st.session_state.files.append(name)
                    st.session_state.file_contents[name] = content
                    st.success(f"Loaded '{name}' successfully! ({num_chunks} vector chunks indexed)")
                except Exception as ex:
                    st.error(f"Failed to ingest sample document: {ex}")

with c_act2:
    if st.button("🗑️ Reset Database", use_container_width=True):
        reset_database()
        st.info("Vector database and chat history cleared.")
        st.rerun()

with c_act3:
    doc_count = len(st.session_state.files)
    total_chunks = st.session_state.store.count()
    st.caption(f"Status: **{doc_count}** docs | **{total_chunks}** chunks")

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-num">{len(st.session_state.files)}</div>
        <div class="metric-label">Documents Loaded</div></div>""",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-num">{st.session_state.store.count()}</div>
        <div class="metric-label">Vector Chunks in DB</div></div>""",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-num">768-D</div>
        <div class="metric-label">Embeddings ({EMBED_MODEL})</div></div>""",
        unsafe_allow_html=True,
    )
with m4:
    active_m = gemini_models()[0] if gemini_models() else "gemini"
    st.markdown(
        f"""<div class="metric-card"><div class="metric-num">{active_m.split('-')[1] if '-' in active_m else active_m}</div>
        <div class="metric-label">Primary LLM ({active_m})</div></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_inspector, tab_visualizer, tab_glossary = st.tabs([
    "💬 Chat & Document Q&A",
    "🔬 Chunk Inspector & Playground",
    "🧭 Interactive RAG Visualizer",
    "📚 RAG Concept Glossary",
])

# =============================================================================
# TAB 1: Chat & Document Q&A
# =============================================================================
with tab_chat:
    st.markdown("### 💬 Ask Questions Grounded in Your Documents")
    
    # Display Chat History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            # If assistant and has source hits
            if m["role"] == "assistant" and "hits" in m and m["hits"]:
                with st.expander(f"📑 Sources & Similarity Scores ({len(m['hits'])} chunks retrieved)"):
                    for i, h in enumerate(m["hits"], start=1):
                        score = h.get("score", 0.0)
                        pct = int(score * 100)
                        badge_cls = "badge-high" if pct >= 75 else ("badge-med" if pct >= 50 else "badge-low")
                        st.markdown(
                            f"""
                            <div style="margin-bottom: 0.6rem; padding: 0.5rem 0.75rem; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;">
                                <div>
                                    <span class="badge-pill badge-source">📄 {h.get('source', 'Document')}</span>
                                    <span class="badge-pill {badge_cls}">Cosine Similarity: {score:.3f} ({pct}% match)</span>
                                    <span style="font-size: 0.8rem; color: #6b7280;">Chunk #{h.get('chunk', 0)} ({h.get('char_count', len(h.get('text', '')))} chars)</span>
                                </div>
                                <div style="margin-top: 0.4rem; font-size: 0.88rem; color: #374151; background: #f9fafb; padding: 0.5rem; border-radius: 6px; border-left: 3px solid #6366f1;">
                                    "{h.get('text', '')}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                
                # Educational Prompt Inspector
                if "prompt" in m and m["prompt"]:
                    with st.expander("🔍 Inspect Full Prompt Sent to LLM (Learning View)"):
                        st.caption("Here is the exact prompt assembled with your retrieved context and question:")
                        st.code(m["prompt"], language="text")

    # Handle Chat Input & Pending Quick Questions
    user_input = st.chat_input("Ask a question about your documents...")
    effective_question = user_input or st.session_state.pending_question

    if effective_question:
        st.session_state.pending_question = None
        
        # Check if vector DB has documents
        if st.session_state.store.count() == 0:
            st.warning("⚠️ No documents indexed in the vector store yet! Upload a file from the sidebar or click **⚡ 1-Click Load Sample Document** above.")
        else:
            # Append and render user question
            st.session_state.messages.append({"role": "user", "content": effective_question})
            with st.chat_message("user"):
                st.write(effective_question)
                
            # Generate grounded assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching vector database & synthesizing answer..."):
                    try:
                        k_val = st.session_state.get("retrieval_k", 4)
                        reply, hits, rendered_prompt = answer(
                            st.session_state.store, effective_question, k=k_val, return_prompt=True
                        )
                    except Exception as err:
                        reply = f"Error during generation: {err}"
                        hits = []
                        rendered_prompt = ""

                st.write(reply)
                
                if hits:
                    with st.expander(f"📑 Sources & Similarity Scores ({len(hits)} chunks retrieved)"):
                        for h in hits:
                            score = h.get("score", 0.0)
                            pct = int(score * 100)
                            badge_cls = "badge-high" if pct >= 75 else ("badge-med" if pct >= 50 else "badge-low")
                            st.markdown(
                                f"""
                                <div style="margin-bottom: 0.6rem; padding: 0.5rem 0.75rem; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;">
                                    <div>
                                        <span class="badge-pill badge-source">📄 {h.get('source', 'Document')}</span>
                                        <span class="badge-pill {badge_cls}">Cosine Similarity: {score:.3f} ({pct}% match)</span>
                                        <span style="font-size: 0.8rem; color: #6b7280;">Chunk #{h.get('chunk', 0)} ({h.get('char_count', len(h.get('text', '')))} chars)</span>
                                    </div>
                                    <div style="margin-top: 0.4rem; font-size: 0.88rem; color: #374151; background: #f9fafb; padding: 0.5rem; border-radius: 6px; border-left: 3px solid #6366f1;">
                                        "{h.get('text', '')}"
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    
                    with st.expander("🔍 Inspect Full Prompt Sent to LLM (Learning View)"):
                        st.caption("Here is the exact prompt assembled with your retrieved context and question:")
                        st.code(rendered_prompt, language="text")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "hits": hits,
                    "prompt": rendered_prompt,
                })
                st.rerun()

    # Chat Export Option
    if st.session_state.messages:
        with st.expander("📥 Export Conversation History"):
            chat_md = "# RAG Chat Session Log\n\n"
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_md += f"### {role}\n{msg['content']}\n\n"
            st.download_button(
                label="Download Chat as Markdown (.md)",
                data=chat_md,
                file_name="rag_chat_history.md",
                mime="text/markdown",
            )

# =============================================================================
# TAB 2: Chunk Inspector & Live Experimenter
# =============================================================================
with tab_inspector:
    st.markdown("### 🔬 Chunk Inspector & Live Experimenter")
    st.write(
        "Chunking is the foundation of RAG. Use this playground to see in real-time how changing "
        "**Chunk Size** and **Chunk Overlap** cuts and organizes your text into segments."
    )
    
    col_insp_mode, col_insp_doc = st.columns([1.5, 2.5])
    with col_insp_mode:
        source_choice = st.radio(
            "Text Source:",
            ["Loaded Documents", "Custom Text Input"],
            horizontal=True,
        )
        
    sample_text_val = ""
    if source_choice == "Loaded Documents":
        if st.session_state.file_contents:
            selected_file = col_insp_doc.selectbox("Select document to inspect:", list(st.session_state.file_contents.keys()))
            sample_text_val = st.session_state.file_contents.get(selected_file, "")
        else:
            col_insp_doc.info("No documents uploaded yet. Load the sample document or switch to 'Custom Text Input'.")
            sample_text_val = "Retrieval-Augmented Generation (RAG) is a technique where a language model is given relevant passages from external documents before answering. This reduces hallucinations because the model answers from real context instead of guessing. A RAG system splits documents into chunks, converts each chunk into an embedding vector, and stores the vectors in a vector database."
    else:
        sample_text_val = st.text_area(
            "Paste or edit text to inspect chunking:",
            value="Retrieval-Augmented Generation (RAG) is a technique where a language model is given relevant passages from external documents before answering. This reduces hallucinations because the model answers from real context instead of guessing. A RAG system splits documents into chunks, converts each chunk into an embedding vector, and stores the vectors in a vector database. When a user asks a question, the question is embedded too, the most similar chunks are retrieved using cosine similarity, and those chunks are passed to the LLM as context. Chunk size matters. Small chunks are precise but may miss context. Large chunks keep context but add noise. Overlap between chunks helps avoid cutting an idea in half.",
            height=140,
        )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        test_size = st.slider("Experimental Chunk Size (characters)", min_value=100, max_value=1200, value=300, step=50)
    with col_s2:
        test_overlap = st.slider("Experimental Chunk Overlap (characters)", min_value=0, max_value=200, value=60, step=10)

    # Analyze chunks with metadata
    detailed_chunks = chunk_text_with_metadata(sample_text_val, size=test_size, overlap=test_overlap)

    # Metrics
    tot_chars = len(sample_text_val)
    tot_words = len(sample_text_val.split())
    num_c = len(detailed_chunks)
    avg_c = (sum(c["char_count"] for c in detailed_chunks) // num_c) if num_c > 0 else 0

    st.markdown("#### 📊 Chunking Analysis")
    im1, im2, im3, im4 = st.columns(4)
    im1.metric("Original Characters", tot_chars)
    im2.metric("Total Chunks Created", num_c)
    im3.metric("Average Chunk Length", f"{avg_c} chars")
    im4.metric("Configured Overlap", f"{test_overlap} chars")

    st.markdown("#### 🧩 Visualized Chunks")
    if not detailed_chunks:
        st.info("Provide some text above to visualize chunking.")
    else:
        for c in detailed_chunks:
            overlap_info = f"<span class='badge-pill badge-med'>Overlap with prev: {c['overlap_with_prev']} chars</span>" if c['overlap_with_prev'] > 0 else "<span class='badge-pill badge-low'>First Chunk (No overlap)</span>"
            st.markdown(
                f"""
                <div class="chunk-card">
                    <div class="chunk-meta">
                        <div>
                            <b>Chunk #{c['index']}</b> &bull; {c['char_count']} chars &bull; {c['word_count']} words
                        </div>
                        <div>
                            {overlap_info}
                        </div>
                    </div>
                    <div class="chunk-text">"{c['text']}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Educational Chunking Guide
    with st.expander("💡 Educational Guide: How to choose Chunk Size & Overlap"):
        st.markdown(
            """
            - **Small Chunks (200 - 400 chars)**:
              - *Pros:* High retrieval precision, low noise, exact fact finding.
              - *Cons:* Can lose surrounding context or narrative continuity.
            - **Medium Chunks (500 - 1000 chars)**:
              - *Recommended for most RAG applications.* Captures 1-2 complete paragraphs while staying well within LLM context limits.
            - **Large Chunks (1000+ chars)**:
              - *Pros:* Rich semantic context.
              - *Cons:* Dilutes vector embedding similarity with irrelevant sentences and wastes prompt tokens.
            - **Why Chunk Overlap Matters**:
              - Without overlap, a crucial sentence or idea might be split directly in half between Chunk $N$ and Chunk $N+1$.
              - Overlap guarantees that every boundary transition retains shared context.
            """
        )

# =============================================================================
# TAB 3: Interactive RAG Visualizer
# =============================================================================
with tab_visualizer:
    st.markdown("### 🧭 Step-by-Step RAG Visualizer")
    st.write("Understand the complete lifecycle of how a question is answered using your external documents:")

    # Step 1
    st.markdown(
        """
        <div class="step-box">
            <div class="step-header">
                <span>1️⃣</span> <span>Document Ingestion & Text Extraction</span>
            </div>
            <p>Documents (PDF, TXT, Markdown) are read into clean plain-text strings. For PDFs, <code>pypdf</code> traverses each page and extracts text streams.</p>
            <div style="font-size: 0.85rem; color: #4f46e5; background: #f5f3ff; padding: 0.4rem 0.8rem; border-radius: 6px;">
                <b>Implemented in:</b> <code>rag/loader.py: load_file()</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 2
    st.markdown(
        """
        <div class="step-box">
            <div class="step-header">
                <span>2️⃣</span> <span>Text Chunking with Sliding Window Overlap</span>
            </div>
            <p>Raw text is divided into segments of <b>Chunk Size</b> with a repeated boundary of <b>Overlap</b> characters. Splitting breaks at natural space boundaries rather than cutting words in half.</p>
            <div style="font-size: 0.85rem; color: #4f46e5; background: #f5f3ff; padding: 0.4rem 0.8rem; border-radius: 6px;">
                <b>Implemented in:</b> <code>rag/chunker.py: chunk_text()</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 3
    st.markdown(
        """
        <div class="step-box">
            <div class="step-header">
                <span>3️⃣</span> <span>Dense Vector Embeddings</span>
            </div>
            <p>Each text chunk is converted into a 768-dimensional geometric coordinate vector by <b>Google Gemini's <code>gemini-embedding-001</code></b>. Semantically similar passages end up close to each other in vector space.</p>
            <div style="font-size: 0.85rem; color: #4f46e5; background: #f5f3ff; padding: 0.4rem 0.8rem; border-radius: 6px;">
                <b>Implemented in:</b> <code>rag/embeddings.py: embed()</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 4
    st.markdown(
        """
        <div class="step-box">
            <div class="step-header">
                <span>4️⃣</span> <span>Vector Database Indexing (ChromaDB)</span>
            </div>
            <p>Embeddings and original text chunks are stored in an in-memory <b>ChromaDB</b> collection indexed with <b>Cosine Similarity</b> ($1 - \text{cosine distance}$).</p>
            <div style="font-size: 0.85rem; color: #4f46e5; background: #f5f3ff; padding: 0.4rem 0.8rem; border-radius: 6px;">
                <b>Implemented in:</b> <code>rag/vectorstore.py: VectorStore</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 5
    st.markdown(
        """
        <div class="step-box">
            <div class="step-header">
                <span>5️⃣</span> <span>Semantic Retrieval ($k$-Nearest Neighbors)</span>
            </div>
            <p>When you ask a question, the question is also converted into an embedding vector. ChromaDB calculates cosine distances across all chunks and retrieves the top-<b>k</b> highest-scoring matches.</p>
            <div style="font-size: 0.85rem; color: #4f46e5; background: #f5f3ff; padding: 0.4rem 0.8rem; border-radius: 6px;">
                <b>Implemented in:</b> <code>rag/vectorstore.py: search()</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 6
    st.markdown(
        """
        <div class="step-box">
            <div class="step-header">
                <span>6️⃣</span> <span>Prompt Augmentation & Grounded Generation</span>
            </div>
            <p>Retrieved chunks are packaged into a structured system prompt. The LLM (Gemini with Groq backup) is strictly instructed to answer <b>ONLY</b> using the provided context, preventing hallucinations.</p>
            <div style="font-size: 0.85rem; color: #4f46e5; background: #f5f3ff; padding: 0.4rem 0.8rem; border-radius: 6px;">
                <b>Implemented in:</b> <code>rag/pipeline.py: answer() & rag/llm.py</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB 4: RAG Concept Glossary & Learning Guide
# =============================================================================
with tab_glossary:
    st.markdown("### 📚 Core RAG Concepts Explained")
    st.write("A student- and developer-friendly cheat sheet for mastering Retrieval-Augmented Generation:")

    with st.expander("📐 What is an Embedding Vector?", expanded=True):
        st.markdown(
            """
            An **embedding** turns words and sentences into a list of numbers (a vector) in a high-dimensional mathematical space (e.g., 768 dimensions).
            
            - Words or sentences with similar meanings (e.g. *"puppy"* and *"canine"*) have vectors pointing in nearly the exact same direction.
            - Unlike keyword search, semantic search understands concepts, synonyms, and paraphrasing!
            """
        )

    with st.expander("🎯 What is Cosine Similarity?"):
        st.markdown(
            """
            **Cosine Similarity** measures the cosine of the angle $\\theta$ between two vectors:
            
            $$\\text{Cosine Similarity} = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$$
            
            - **1.0 (or 100%)**: The two vectors point in the identical direction (perfect semantic match).
            - **0.0**: The vectors are perpendicular (unrelated topics).
            - It measures **direction rather than length**, meaning a short sentence and a long paragraph discussing the same concept will still match!
            """
        )

    with st.expander("🛡️ How does RAG eliminate Hallucinations?"):
        st.markdown(
            """
            Standard LLMs answer from statistical memory learned during training, which can cause them to invent plausible-sounding falsehoods (**hallucination**).
            
            In RAG:
            1. Real factual excerpts are fetched directly from your verified documents.
            2. The system prompt instructs: *"Answer the question using ONLY the context below. If not found, say I couldn't find that."*
            3. The model acts as a reading-comprehension engine rather than guessing!
            """
        )

    with st.expander("🎛️ What is the Top-$k$ Parameter?"):
        st.markdown(
            """
            **$k$** dictates how many chunks to retrieve from the vector database:
            - **$k = 1$ to $2$**: Very concise, minimizes cost and prompt tokens, but might miss multi-part answers.
            - **$k = 4$ to $6$**: Optimal balance for most questions; gives the LLM enough context without cluttering the prompt.
            - **$k = 8+$**: Provides exhaustive context, but increases latency and may introduce distracting noise ("Lost in the Middle" effect).
            """
        )

    with st.expander("🗄️ Vector Database vs Relational Database (SQL)"):
        st.markdown(
            """
            | Feature | Traditional SQL DB | Vector DB (e.g. ChromaDB) |
            | :--- | :--- | :--- |
            | **Query Type** | Exact match (`WHERE status = 'active'`) | Approximate Nearest Neighbor (ANN) |
            | **Search Mechanism** | B-Tree / Hash Index | HNSW (Hierarchical Navigable Small World) graphs |
            | **Understanding** | Strict lexical match | Semantic context & synonyms |
            | **Best For** | Structured transactional data | Unstructured text, documents, images, audio |
            """
        )

# -----------------------------------------------------------------------------
# Sidebar: Upload & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration & Upload")

    # Document Upload Section
    st.markdown("### 1. Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload course notes, manuals, research papers, or documentation.",
    )

    # Tuning Sliders
    st.markdown("### 2. Pipeline Tuning")
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=300,
        max_value=2000,
        value=800,
        step=100,
        help="Size of each text segment. Smaller = precise; Larger = broader context.",
    )
    overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=400,
        value=150,
        step=25,
        help="Number of characters repeated between adjacent chunks to preserve continuity.",
    )
    k_slider = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Number of most similar chunks injected into the prompt context for the LLM.",
    )
    st.session_state["retrieval_k"] = k_slider

    # Process Action
    if st.button("🚀 Process Uploaded Files", type="primary", use_container_width=True) and uploaded_files:
        with st.spinner("Parsing, chunking, and embedding documents..."):
            for f in uploaded_files:
                if f.name in st.session_state.files:
                    continue
                try:
                    data = f.read()
                    # Store plain text content for chunk inspector
                    plain_text = load_file(f.name, data)
                    st.session_state.file_contents[f.name] = plain_text
                    
                    n = ingest_text(st.session_state.store, f.name, plain_text, chunk_size, overlap)
                    st.session_state.files.append(f.name)
                    st.success(f"✓ {f.name}: {n} chunks")
                except Exception as ex:
                    st.error(f"Error processing {f.name}: {ex}")

    # Vector DB Status
    st.markdown("---")
    st.markdown("### 🗄️ Vector Store Status")
    total_in_db = st.session_state.store.count()
    st.markdown(f"Total chunks stored: **{total_in_db}**")
    
    # Document Breakdown
    sources_map = st.session_state.store.get_sources()
    if sources_map:
        st.markdown("**Indexed Documents:**")
        for src, cnt in sources_map.items():
            st.markdown(f"- 📄 `{src}`: **{cnt}** chunks")
    else:
        st.caption("No documents in vector store yet.")

    # System Diagnostics
    st.markdown("---")
    st.markdown("### 🔑 API & Model Health")
    keys = gemini_keys()
    masked = get_masked_keys()
    if keys:
        st.success(f"✓ Gemini API: {len(keys)} key(s) configured")
        with st.expander("Key details"):
            for mk in masked:
                st.code(mk, language="text")
    else:
        st.error("⚠️ No Gemini API Key found!")
        st.caption("Add `GEMINI_API_KEY` to your `.env` or Streamlit secrets.")

    st.caption(f"Models: `{', '.join(gemini_models())}`")
    st.caption(f"Backup Groq LLM: `{'Enabled (' + GROQ_MODEL + ')' if GROQ_API_KEY else 'Disabled'}`")

# Fixed Footer Branding
st.markdown(
    """
    <div class="footer">
        <b>RAG Learning Lab</b> &bull; Built by Muniswami Y
    </div>
    """,
    unsafe_allow_html=True,
)