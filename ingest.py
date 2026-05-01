"""
ingest.py — Load, chunk, embed via Gemini API, save to local JSON index.
Pure stdlib only.
"""

import sys
import time
from pathlib import Path

from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_PATH

SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}


def _p(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            _p(f"  ! pdfplumber not installed — skipping {path.name}")
            _p(f"    install with: pip3 install pdfplumber")
            return ""
        try:
            parts = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n\n".join(parts)
        except Exception as e:
            _p(f"  ! PDF read error: {e}")
            return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            return fp.read()
    except Exception as e:
        _p(f"  ! read error: {e}")
        return ""


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    chunks = []
    step = max(1, size - overlap)
    i = 0
    while i < len(text):
        chunks.append(text[i: i + size])
        i += step
    return chunks


def ingest_documents(docs_dir="./docs"):
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        _p(f"Error: directory not found: {docs_dir}")
        sys.exit(1)

    files = sorted([f for f in docs_path.rglob("*")
                    if f.is_file() and f.suffix.lower() in SUPPORTED])
    if not files:
        _p(f"Error: no supported documents found in '{docs_dir}'")
        _p(f"Supported: {', '.join(sorted(SUPPORTED))}")
        sys.exit(1)

    _p(f"\nFound {len(files)} document(s):")
    for f in files:
        _p(f"  - {f.name} ({f.stat().st_size:,} bytes)")

    # ── Chunk ────────────────────────────────────────────────────────────────
    _p("\nChunking documents...")
    all_chunks, all_meta = [], []
    for f in files:
        raw = load_document(f)
        if not raw.strip():
            _p(f"  - {f.name}: skipped (empty)")
            continue
        chunks = chunk_text(raw)
        for j, c in enumerate(chunks):
            all_chunks.append(c)
            all_meta.append({
                "source":       f.name,
                "chunk_index":  j,
                "total_chunks": len(chunks),
            })
        _p(f"  + {f.name}: {len(chunks)} chunks")

    if not all_chunks:
        _p("Error: no text could be extracted")
        sys.exit(1)

    # ── Embed ────────────────────────────────────────────────────────────────
    import gemini
    import vector_store

    _p(f"\nEmbedding {len(all_chunks)} chunks via Gemini API ({EMBEDDING_MODEL})...")
    BATCH = 20
    all_embeddings = []
    n_batches = (len(all_chunks) + BATCH - 1) // BATCH

    for i in range(0, len(all_chunks), BATCH):
        b     = i // BATCH + 1
        batch = all_chunks[i: i + BATCH]
        _p(f"  batch {b}/{n_batches} ({len(batch)} chunks)... ", end="")
        try:
            vecs = gemini.embed_batch(batch, model=EMBEDDING_MODEL,
                                      task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.extend(vecs)
            _p("ok")
        except Exception as e:
            _p(f"failed: {e}")
            sys.exit(1)
        if i + BATCH < len(all_chunks):
            time.sleep(0.3)

    # ── Save ─────────────────────────────────────────────────────────────────
    vector_store.save(all_chunks, all_embeddings, all_meta, INDEX_PATH)
    _p(f"\nDone. {len(all_chunks)} chunks from {len(files)} file(s) saved to {INDEX_PATH}/\n")
