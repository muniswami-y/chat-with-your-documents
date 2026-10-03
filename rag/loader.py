import io
from pypdf import PdfReader


def load_file(name: str, data: bytes) -> str:
    """Return plain text from a PDF or TXT/MD file."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if lower.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {name}")
