# Chat With Your Documents (RAG)

Upload PDFs or text files and ask questions. Answers come only from your documents.

**Pipeline:** Document → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer

## Tech
- Streamlit (UI)
- Gemini API (embeddings + answers), with multi-key and multi-model fallback
- ChromaDB (vector database, cosine similarity)
- pypdf (PDF text extraction)
- Optional Groq backup LLM

## Run locally
```
pip install -r requirements.txt
cp .env.example .env      # then add your keys
streamlit run app.py
```

## Deploy free (Streamlit Community Cloud)
1. Push this folder to GitHub.
2. Go to share.streamlit.io, click **Create app**, pick the repo, main file `app.py`.
3. Under **Advanced settings → Secrets**, paste:
```
GEMINI_API_KEY = "key1,key2"
GEMINI_MODELS = "gemini-3.5-flash-lite,gemini-3.8-flash"
```
4. Deploy.

## Project structure
```
app.py              Streamlit chat UI
rag/loader.py       PDF / TXT reading
rag/chunker.py      Overlapping text chunks
rag/embeddings.py   Gemini embeddings
rag/vectorstore.py  ChromaDB wrapper
rag/llm.py          Gemini + Groq answer generation
rag/pipeline.py     ingest() and answer()
samples/sample.txt  Test document
```

## Improving retrieval quality
The sidebar lets you tune chunk size, overlap and top-k. The "Sources used" panel shows which
chunks were retrieved and their similarity scores, so you can see why an answer was given.
