import uuid
import chromadb


class VectorStore:
    """Wrapper around an in-memory ChromaDB collection using cosine similarity."""

    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self._new_collection()

    def _new_collection(self):
        self.col = self.client.create_collection(
            name=f"docs_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self):
        """Reset the vector database collection."""
        try:
            self.client.delete_collection(name=self.col.name)
        except Exception:
            pass
        self._new_collection()

    def add(self, chunks, embeddings, source):
        if not chunks:
            return
        self.col.add(
            ids=[uuid.uuid4().hex for _ in chunks],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )

    def count(self):
        return self.col.count()

    def get_sources(self):
        """Return a mapping of source document names to their chunk counts."""
        if self.count() == 0:
            return {}
        data = self.col.get(include=["metadatas"])
        counts = {}
        for meta in data.get("metadatas", []):
            src = meta.get("source", "Unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def get_all_documents(self, limit=100):
        """Return indexed documents and their metadata for inspection."""
        if self.count() == 0:
            return []
        data = self.col.get(include=["documents", "metadatas"], limit=limit)
        items = []
        for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            items.append({
                "source": meta.get("source", "Unknown"),
                "chunk": meta.get("chunk", 0),
                "text": doc,
                "char_count": len(doc),
            })
        items.sort(key=lambda x: (x["source"], x["chunk"]))
        return items

    def search(self, query_embedding, k=4):
        total = self.col.count()
        if total == 0:
            return []
        k = min(k, total)
        res = self.col.query(query_embeddings=[query_embedding], n_results=k)
        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # Cosine distance to cosine similarity
            sim = max(0.0, min(1.0, 1.0 - dist))
            results.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "chunk": meta["chunk"],
                    "score": round(sim, 3),
                    "char_count": len(doc),
                }
            )
        return results

