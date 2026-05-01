"""
gemini.py — Direct Gemini REST API calls using only Python stdlib (urllib).
No SDK, no heavy dependencies, no auth library.
"""

import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set.\n"
            "Run: export GEMINI_API_KEY=your_key\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return key


def _post(url, payload, timeout=60):
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API error {e.code}: {err_body}")
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


def embed(text, model="text-embedding-004", task_type="RETRIEVAL_QUERY"):
    """Embed a single text. Returns list[float]."""
    key = _api_key()
    url = f"{API_BASE}/models/{model}:embedContent?key={key}"
    payload = {
        "model":    f"models/{model}",
        "content":  {"parts": [{"text": text}]},
        "taskType": task_type,
    }
    return _post(url, payload)["embedding"]["values"]


def embed_batch(texts, model="text-embedding-004", task_type="RETRIEVAL_DOCUMENT"):
    """Embed multiple texts in one API call. Returns list[list[float]]."""
    key = _api_key()
    url = f"{API_BASE}/models/{model}:batchEmbedContents?key={key}"
    payload = {
        "requests": [
            {
                "model":    f"models/{model}",
                "content":  {"parts": [{"text": t}]},
                "taskType": task_type,
            }
            for t in texts
        ]
    }
    response = _post(url, payload)
    return [e["values"] for e in response["embeddings"]]


def generate(prompt, model="gemini-1.5-flash"):
    """Generate text from a prompt. Returns the response string."""
    key = _api_key()
    url = f"{API_BASE}/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = _post(url, payload)
    return response["candidates"][0]["content"]["parts"][0]["text"]
