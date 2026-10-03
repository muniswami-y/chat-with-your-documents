import requests
from google import genai
from .config import gemini_keys, gemini_models, GROQ_API_KEY, GROQ_MODEL

FRIENDLY_ERROR = "The assistant is busy right now. Please try again in a moment."


def _gemini(prompt):
    for key in gemini_keys():
        client = genai.Client(api_key=key)
        for model in gemini_models():
            try:
                res = client.models.generate_content(model=model, contents=prompt)
                if res.text:
                    return res.text
            except Exception:
                continue  # next model / next key
    return None


def _groq(prompt):
    if not GROQ_API_KEY:
        return None
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def generate(prompt):
    """Gemini (all keys + models) first, then Groq, then a friendly message."""
    return _gemini(prompt) or _groq(prompt) or FRIENDLY_ERROR
