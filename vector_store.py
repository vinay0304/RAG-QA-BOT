"""
vector_store.py — Pure-Python JSON vector store with cosine similarity.
No numpy, no external deps. Plenty fast for thousands of chunks.
"""

import json
import math
from pathlib import Path


def save(chunks, embeddings, metadata, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/index.json", "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": chunks, "embeddings": embeddings, "metadata": metadata},
            f,
        )


def load(path):
    p = Path(path) / "index.json"
    if not p.exists():
        return None, None, None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"], data["embeddings"], data["metadata"]


def _cosine(a, b):
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-9)


def search(query_vec, chunks, embeddings, metadata, top_k=3, min_score=0.25):
    scored = [(_cosine(query_vec, emb), i) for i, emb in enumerate(embeddings)]
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, i in scored[:top_k]:
        if score >= min_score:
            results.append({
                "document": chunks[i],
                "metadata": metadata[i],
                "score":    score,
            })
    return results
