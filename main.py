#!/usr/bin/env python3
"""
main.py — RAG Q&A Bot. Pure Python stdlib + Gemini REST API.

Commands:
  python3 main.py ingest              Index all docs in docs/
  python3 main.py ingest <path>       Index a custom directory
  python3 main.py query               Interactive Q&A session
  python3 main.py query "question"    Single question and exit
  python3 main.py status              Show index stats
"""

import sys

BANNER = r"""
================================================
       RAG Q&A Bot
       Pure Python + Gemini API (free)
================================================
"""


def show_help():
    print(BANNER)
    print("Usage:")
    print("  python3 main.py ingest             # index docs/ folder")
    print("  python3 main.py ingest <path>      # index a custom folder")
    print("  python3 main.py query              # interactive Q&A")
    print('  python3 main.py query "..."        # one-shot question')
    print("  python3 main.py status             # show what's indexed")
    print()


def cmd_ingest(docs_dir="./docs"):
    from ingest import ingest_documents
    ingest_documents(docs_dir)


def cmd_status():
    import vector_store
    from config import INDEX_PATH

    print(BANNER)
    chunks, _, metadata = vector_store.load(INDEX_PATH)
    if chunks is None:
        print("No index found. Run: python3 main.py ingest\n")
        return

    sources = sorted(set(m["source"] for m in metadata))
    print(f"Total chunks: {len(chunks)}")
    print(f"Documents:    {len(sources)}")
    print(f"Index path:   {INDEX_PATH}/")
    print(f"Files:")
    for s in sources:
        print(f"  - {s}")
    print()


def cmd_query(question=None):
    from query import query_rag

    if question:
        try:
            query_rag(question)
        except Exception as e:
            print(f"\nError: {e}\n")
        return

    print(BANNER)
    print("Type your question. Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q", ":q"}:
            print("Goodbye!")
            break

        try:
            query_rag(q)
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    if len(sys.argv) < 2:
        show_help()
        return

    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        path = sys.argv[2] if len(sys.argv) > 2 else "./docs"
        cmd_ingest(path)
    elif cmd == "query":
        q = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_query(q)
    elif cmd == "status":
        cmd_status()
    else:
        print(f"Unknown command: '{cmd}'")
        show_help()


if __name__ == "__main__":
    main()
