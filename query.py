"""
query.py — Embed query via Gemini, search local index, generate answer.
Pure stdlib.
"""

import sys
import textwrap

from config import EMBEDDING_MODEL, GEMINI_MODEL, TOP_K, INDEX_PATH


def query_rag(question):
    import gemini
    import vector_store

    chunks, embeddings, metadata = vector_store.load(INDEX_PATH)
    if chunks is None:
        print("\nNo index found. Run: python3 main.py ingest\n")
        sys.exit(1)

    # ── Embed query ──────────────────────────────────────────────────────────
    try:
        query_vec = gemini.embed(question, model=EMBEDDING_MODEL,
                                 task_type="RETRIEVAL_QUERY")
    except Exception as e:
        print(f"\nError embedding query: {e}\n")
        return None

    # ── Retrieve ─────────────────────────────────────────────────────────────
    results = vector_store.search(query_vec, chunks, embeddings, metadata,
                                  top_k=TOP_K)
    if not results:
        print("\nNo relevant content found. Try rephrasing your question.\n")
        return None

    # ── Build prompt ─────────────────────────────────────────────────────────
    parts = []
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        parts.append(
            f"[{i}] {m['source']} (chunk {m['chunk_index'] + 1}/{m['total_chunks']})\n"
            f"{r['document']}"
        )
    context = "\n\n".join(parts)

    prompt = textwrap.dedent(f"""
        You are a precise Q&A assistant. Answer ONLY using the context below.
        Cite sources using bracket numbers like [1], [2].
        If the answer is not in the context, say:
        "The provided documents don't contain enough information to answer this."
        Be concise. Use markdown where helpful.

        Context:
        {context}

        Question: {question}

        Answer:
    """).strip()

    # ── Generate ─────────────────────────────────────────────────────────────
    try:
        answer = gemini.generate(prompt, model=GEMINI_MODEL)
    except Exception as e:
        print(f"\nError generating answer: {e}\n")
        return None

    # ── Display ──────────────────────────────────────────────────────────────
    sources = sorted(set(r["metadata"]["source"] for r in results))
    scores  = [f"{r['score']:.0%}" for r in results]

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer.strip())
    print()
    print(f"Sources:   {', '.join(sources)}")
    print(f"Relevance: {', '.join(scores)}")
    print("=" * 70)
    print()

    return answer
