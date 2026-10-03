from .chunker import chunk_text
from .embeddings import embed, embed_query
from .llm import generate
from .loader import load_file

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}

Answer:"""


def ingest(store, name, data, chunk_size=800, overlap=150):
    """Document -> Chunking -> Embeddings -> Vector DB. Returns number of chunks."""
    text = load_file(name, data)
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return 0
    store.add(chunks, embed(chunks), name)
    return len(chunks)


def answer(store, question, k=4):
    """Retrieval -> LLM -> Answer. Returns (answer, retrieved_chunks)."""
    hits = store.search(embed_query(question), k=k)
    if not hits:
        return "Please upload and process a document first.", []
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    return generate(PROMPT.format(context=context, question=question)), hits
