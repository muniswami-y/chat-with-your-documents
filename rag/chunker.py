def chunk_text(text: str, size: int = 800, overlap: int = 150):
    """Split text into overlapping chunks, breaking at spaces."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
