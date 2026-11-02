def chunk_text_with_metadata(text: str, size: int = 800, overlap: int = 150):
    """Split text into overlapping chunks, returning text and inspection metadata."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks = []
    start = 0
    idx = 0
    last_end = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            space = cleaned.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        chunk_str = cleaned[start:end].strip()
        if chunk_str:
            overlap_chars = max(0, last_end - start) if idx > 0 else 0
            chunks.append({
                "index": idx,
                "text": chunk_str,
                "char_count": len(chunk_str),
                "word_count": len(chunk_str.split()),
                "start": start,
                "end": end,
                "overlap_with_prev": overlap_chars,
            })
            idx += 1
            last_end = end
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_text(text: str, size: int = 800, overlap: int = 150):
    """Split text into overlapping chunks, breaking at spaces."""
    return [c["text"] for c in chunk_text_with_metadata(text, size, overlap)]

