# ─── Model Settings ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"   # Gemini embedding API (current model)
GEMINI_MODEL    = "gemini-2.5-flash"     # Free tier, fast, current model

# ─── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 300    # reduced from 600
CHUNK_OVERLAP = 40     # reduced from 80

# ─── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 3              # reduced from 5

# ─── Paths ─────────────────────────────────────────────────────────────────────
INDEX_PATH = "./rag_index"   # folder where embeddings + chunks are stored
DOCS_PATH  = "./docs"
