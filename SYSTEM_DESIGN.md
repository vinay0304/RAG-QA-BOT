# System Design — RAG Q&A Bot
> Lo-Fi Architecture Diagram

A Retrieval-Augmented Generation (RAG) Q&A bot built with **only Python stdlib + the Gemini REST API**. Zero ML dependencies. Runs on any machine.

---

## Big Picture (Two Pipelines)

```
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                         │
│                   (run once, or on update)                      │
└─────────────────────────────────────────────────────────────────┘

  Your Files (docs/)
  ┌──────────────┐
  │  .txt / .md  │
  │  .pdf        │──────►  Text Extraction
  │  .markdown   │         (pdfplumber for PDFs, open() for text)
  └──────────────┘                │
                                  ▼
                         ┌────────────────┐
                         │  Raw Full Text │
                         └────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Chunking               │
                    │  Size:    300 chars     │
                    │  Overlap: 40 chars      │
                    │  Pure Python slicing    │
                    └─────────────────────────┘
                                  │
                     [chunk 1] [chunk 2] [chunk 3] ...
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  Gemini Embedding API       │
                    │  gemini-embedding-001       │
                    │  Output: 768-dim vectors    │
                    │  HTTPS POST via urllib      │
                    └─────────────────────────────┘
                                  │
                         [0.12, -0.43, ...]
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  Local JSON Index           │
                    │  rag_index/index.json       │
                    │  Stores:                    │
                    │    • chunk text             │
                    │    • embedding vector       │
                    │    • metadata (filename,    │
                    │      chunk index)           │
                    └─────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                            │
│                   (runs every time you ask)                     │
└─────────────────────────────────────────────────────────────────┘

  User types question
         │
         ▼
  "What is RAG?"
         │
         ▼
  ┌─────────────────────────┐
  │  Gemini Embedding API   │
  │  task=RETRIEVAL_QUERY   │
  └─────────────────────────┘
         │
         ▼
  [0.08, -0.51, ...]  ← query vector
         │
         ▼
  ┌──────────────────────────────────────┐
  │  Cosine Similarity Search            │
  │  Pure Python (math.sqrt + zip + sum) │
  │  Compare query vector vs all stored  │
  │  vectors → return Top-K (default 3) │
  └──────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  Retrieved Chunks                                   │
  │                                                     │
  │  [1] sample.txt (chunk 2) — score 75%              │
  │      "An embedding is a dense numerical..."        │
  │                                                     │
  │  [2] sample.txt (chunk 5) — score 73%              │
  │      "Texts with similar meanings are placed..."   │
  │                                                     │
  │  [3] sample.txt (chunk 1) — score 72%              │
  │      "Documents are collected and split..."        │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Prompt Assembly                                         │
  │                                                          │
  │  "Answer ONLY using the context below.                   │
  │   Cite your sources using [1], [2], etc.                │
  │                                                          │
  │   Context:                                               │
  │   [1] sample.txt ... <chunk text> ...                    │
  │   [2] sample.txt ... <chunk text> ...                    │
  │   ...                                                    │
  │                                                          │
  │   Question: What is RAG?"                               │
  └──────────────────────────────────────────────────────────┘
         │
         ▼  (HTTPS POST via urllib)
  ┌──────────────────────────────┐
  │  Gemini 2.5 Flash (Google)   │
  │  Free tier                   │
  │  Generation API              │
  └──────────────────────────────┘
         │
         ▼
  ┌────────────────────────────────────────────────────────────┐
  │  Answer with Citations                                     │
  │                                                            │
  │  "An embedding is a dense numerical representation        │
  │   of text [1]. Texts with similar meanings are placed     │
  │   closer together [1]. Queries are also converted into    │
  │   embedding vectors [3]..."                                │
  │                                                            │
  │  Sources: sample.txt  ·  Scores: 75%, 73%, 72%           │
  └────────────────────────────────────────────────────────────┘
         │
         ▼
  User sees answer in terminal
```

---

## Component Map

```
┌─────────────────────────────────────────────────────────┐
│                       rag-qa-bot/                       │
│                                                         │
│   main.py          CLI router + interactive session     │
│      │                                                  │
│      ├──► ingest.py    load → chunk → embed → store     │
│      │       │                                          │
│      │       ├── pdfplumber (optional, PDF only)        │
│      │       ├── gemini.py  (HTTPS POST → embed API)    │
│      │       └── vector_store.py (JSON write)           │
│      │                                                  │
│      └──► query.py     embed → search → LLM → answer   │
│              │                                          │
│              ├── gemini.py  (embed query + generate)    │
│              └── vector_store.py (cosine similarity)    │
│                                                         │
│   gemini.py        urllib-based REST client (no SDK)    │
│   vector_store.py  pure-Python JSON store + cosine sim  │
│   config.py        all tuneable settings in one place   │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
INGEST:  File ──► Text ──► Chunks ──► Vectors ──► JSON file (disk)
QUERY:   Question ──► Vector ──► Top-K Chunks ──► Gemini ──► Answer
```

---

## Technology Choices & Why

| Layer | Choice | Why |
|---|---|---|
| HTTP | `urllib.request` (stdlib) | No `requests`, no `httpx`, no `grpc`. Just stdlib. |
| Embeddings | `gemini-embedding-001` | Free, 768-dim, no local model needed |
| LLM | `gemini-2.5-flash` | Free tier, fast, accurate citations |
| Vector store | JSON file | No database, no daemon, no native deps |
| Similarity | Pure Python cosine | `sum(x*y) / (sqrt(sum(x²)) * sqrt(sum(y²)))` |
| Chunking | Fixed-size character slicing | Deterministic and fast |

**Total external Python dependencies: ZERO** (pdfplumber only if you need PDF support).

---

## Similarity Search

```
        Query Vector                Document Vectors
        ─────────────               ───────────────────
        [0.08, -0.51, ...]          chunk_1: [0.12, -0.43, ...]  ← closest (cos=0.94)
                │                   chunk_2: [0.31,  0.12, ...]
                │                   chunk_3: [0.07, -0.48, ...]  ← close (cos=0.87)
                │                   chunk_N: [0.80,  0.20, ...]  ← far (cos=0.12)
                │
                └── cosine similarity = dot(q,d) / (|q| × |d|)
                    Returns 0.0 (opposite) → 1.0 (identical meaning)
                    Threshold: 0.25 (chunks below this are filtered out)
```

---

## What Stays Local vs Cloud

```
LOCAL (your machine):                    CLOUD (Google API):
─────────────────────                    ───────────────────
• All your documents                     • Your question
• All embeddings (cached)                • Top-3 retrieved chunks
• JSON index file                        • Generated answer
• Chunking + cosine math                 
                                         (HTTPS, encrypted in transit)
```

---

## Cost

| Component | Cost |
|---|---|
| Embedding (Gemini) | $0 — free tier |
| Generation (Gemini Flash) | $0 — free tier |
| Vector storage | $0 — local JSON |
| Compute | $0 — stdlib only |
| **Total** | **$0** |
