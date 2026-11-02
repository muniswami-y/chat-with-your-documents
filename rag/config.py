import os
from dotenv import load_dotenv

load_dotenv()


def _get_env_or_secret(key: str, default: str = "") -> str:
    """Retrieve environment variable from os.environ or streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


def gemini_keys():
    raw = _get_env_or_secret("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def gemini_models():
    raw = _get_env_or_secret("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash")
    return [m.strip() for m in raw.split(",") if m.strip()]


def get_masked_keys():
    """Return masked previews of active Gemini keys for diagnostics."""
    keys = gemini_keys()
    masked = []
    for k in keys:
        if len(k) > 10:
            masked.append(f"{k[:4]}...{k[-4:]}")
        else:
            masked.append("configured")
    return masked


EMBED_MODEL = _get_env_or_secret("EMBED_MODEL", "gemini-embedding-001")
GROQ_API_KEY = _get_env_or_secret("GROQ_API_KEY", "")
GROQ_MODEL = _get_env_or_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

