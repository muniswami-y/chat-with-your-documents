from google import genai
from google.genai import types
from .config import gemini_keys, EMBED_MODEL

BATCH = 50


def embed(texts, task="RETRIEVAL_DOCUMENT"):
    """Embed a list of texts. Tries each Gemini key until one works."""
    keys = gemini_keys()
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY set.")
    last_error = None
    for key in keys:
        try:
            client = genai.Client(api_key=key)
            out = []
            for i in range(0, len(texts), BATCH):
                res = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=texts[i:i + BATCH],
                    config=types.EmbedContentConfig(task_type=task),
                )
                out.extend(e.values for e in res.embeddings)
            return out
        except Exception as e:  # try next key
            last_error = e
    raise RuntimeError(f"Embedding failed: {last_error}")


def embed_query(text):
    return embed([text], task="RETRIEVAL_QUERY")[0]
