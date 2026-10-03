import uuid
import chromadb


class VectorStore:
    """Small wrapper around an in-memory ChromaDB collection (cosine similarity)."""

    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self.col = self.client.create_collection(
            name=f"docs_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks, embeddings, source):
        self.col.add(
            ids=[uuid.uuid4().hex for _ in chunks],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )

    def count(self):
        return self.col.count()

    def search(self, query_embedding, k=4):
        k = min(k, self.col.count())
        if k == 0:
            return []
        res = self.col.query(query_embeddings=[query_embedding], n_results=k)
        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "chunk": meta["chunk"],
                    "score": round(1 - dist, 3),  # cosine similarity
                }
            )
        return results
