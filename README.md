# RAG Q&A Bot

A Retrieval-Augmented Generation Q&A bot. Drop your documents into `docs/`, ask questions, get answers with citations.

**Pure Python stdlib + Gemini REST API. Zero ML dependencies. $0 to run.**

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for architecture details.

---

## How It Works (TL;DR)

1. **Ingest:** chunk your documents → embed each chunk via Gemini API → save vectors to a local JSON file
2. **Query:** embed your question → cosine-similarity search over stored vectors → top-K chunks + question → Gemini generates a cited answer

No PyTorch, no ChromaDB, no heavy ML libraries. Works on any Python 3.7+.

---

## Step 0 — Prerequisites

Just Python.
```bash
python3 --version
```

> **Note:** Use `python3` (the system one). Avoid bleeding-edge versions like 3.14 — they can have regressions on macOS.

---

## Step 1 — Get Your Free Gemini API Key

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click **"Create API key"**
4. Copy it (looks like `AIzaSy...`)

Free tier: enough requests/day for personal use across embeddings + generation.

---

## Step 2 — Set Your API Key

Each new terminal:
```bash
export GEMINI_API_KEY="paste-your-key-here"
```

To set permanently:
```bash
echo 'export GEMINI_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

Verify it's set:
```bash
echo $GEMINI_API_KEY    # should print your key
```

---

## Step 3 — Navigate to Project

```bash
cd ~/Desktop/rag-qa-bot
```

---

## Step 4 — Install Dependencies

For `.txt` and `.md` files: **nothing to install** — pure stdlib.

For PDF support (optional):
```bash
pip3 install pdfplumber
```

---

## Step 5 — Add Your Documents

Drop files into `docs/`:
- `.txt` — plain text
- `.md`, `.markdown` — markdown
- `.pdf` — PDFs (requires pdfplumber)

A `sample.txt` is included for testing.

---

## Step 6 — Ingest (Index) Your Documents

```bash
python3 main.py ingest
```

What it does:
1. Reads every file in `docs/`
2. Splits each into ~300-character chunks
3. Sends each chunk to Gemini's embedding API → gets a vector
4. Saves everything to `rag_index/index.json`

Re-run only when you add or change documents.

Custom folder:
```bash
python3 main.py ingest /path/to/my/docs
```

---

## Step 7 — Ask Questions

Interactive mode:
```bash
python3 main.py query
```
Type a question, press Enter. Type `exit` to quit.

One-shot:
```bash
python3 main.py query "What is RAG?"
```

---

## Bonus Commands

```bash
python3 main.py status      # show what's indexed
```

---

## Typical Workflow

```bash
cd ~/Desktop/rag-qa-bot
# (drop new docs into docs/ if needed)
python3 main.py ingest      # only re-run if docs changed
python3 main.py query       # ask questions
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY not set` | Run `export GEMINI_API_KEY="your-key"` in the same terminal |
| `API key not valid` (400) | Generate a fresh key at https://aistudio.google.com/app/apikey |
| `Model not found` (404) | Model name changed. List available models (see below) and update `config.py` |
| `Quota exceeded` (429) | Either you hit the daily limit, or that model isn't free for your account. Try a different model. |
| `No index found` | Run `python3 main.py ingest` first |
| `No relevant content found` | Rephrase your question or your docs may not have the answer |
| Process gets killed | Use `python3` (system Python), not `python3.14`. Avoid heavy ML libs. |

### List models your API key can access

```bash
# Models that support text generation
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(m['name']) for m in d['models'] if 'generateContent' in m.get('supportedGenerationMethods',[])]"

# Models that support embeddings
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(m['name']) for m in d['models'] if 'embedContent' in m.get('supportedGenerationMethods',[])]"
```

Pick a model name and put it in [config.py](config.py) (without the `models/` prefix).

---

## File Structure

```
rag-qa-bot/
├── main.py            ← entry point
├── ingest.py          ← chunking + embedding
├── query.py           ← retrieval + answer
├── gemini.py          ← REST API wrapper (urllib only)
├── vector_store.py    ← JSON store + cosine similarity
├── config.py          ← all tunable settings
├── requirements.txt   ← (no required packages)
├── docs/              ← put your documents here
│   └── sample.txt
├── rag_index/         ← auto-created index (gitignored)
├── README.md          ← you are here
└── SYSTEM_DESIGN.md   ← architecture diagram
```

---

## Tuning (edit `config.py`)

| Setting | Default | When to change |
|---|---|---|
| `EMBEDDING_MODEL` | `gemini-embedding-001` | If your account doesn't support it |
| `GEMINI_MODEL` | `gemini-2.5-flash` | If quota is 0 — try `gemini-2.5-flash-lite` etc. |
| `CHUNK_SIZE` | 300 | Bigger for dense docs, smaller for Q&A-style content |
| `CHUNK_OVERLAP` | 40 | More overlap if answers feel cut off |
| `TOP_K` | 3 | More chunks = more context but more tokens |

After changing chunk settings, re-run `python3 main.py ingest`.

