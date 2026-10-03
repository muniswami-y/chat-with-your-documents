import os
from dotenv import load_dotenv

load_dotenv()


def gemini_keys():
    raw = os.getenv("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def gemini_models():
    raw = os.getenv("GEMINI_MODELS", "gemini-3.5-flash-lite")
    return [m.strip() for m in raw.split(",") if m.strip()]


EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
