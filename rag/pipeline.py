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
    return ingest_text(store, name, text, chunk_size=chunk_size, overlap=overlap)


def ingest_text(store, name, text, chunk_size=800, overlap=150):
    """Chunk, embed, and store raw text string in the vector store."""
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return 0
    embeddings = embed(chunks)
    store.add(chunks, embeddings, name)
    return len(chunks)


def answer(store, question, k=4, return_prompt=False):
    """Retrieval -> LLM -> Answer. Returns (answer, retrieved_chunks) or (answer, retrieved_chunks, prompt)."""
    hits = store.search(embed_query(question), k=k)
    if not hits:
        msg = "Please upload and process a document first."
        return (msg, [], "") if return_prompt else (msg, [])
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    rendered_prompt = PROMPT.format(context=context, question=question)
    reply = generate(rendered_prompt)
    if return_prompt:
        return reply, hits, rendered_prompt
    return reply, hits

