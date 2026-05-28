from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from rag_engine import (
    OpenAIEmbeddingVectorStore,
    build_context_block,
    answer_from_context,
    late_orders_summary,
)


def main():
    index_path = ROOT / "data" / "vector_store" / "openai_embeddings_index.json"
    db_path = ROOT / "data" / "retail_bi.db"

    print("=== QUICK START: RAG DENGAN OPENAI EMBEDDINGS ===")

    if not index_path.exists():
        print("Index embedding belum ditemukan.")
        print("Jalankan dulu: python scripts/build_index.py")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY belum ditemukan.")
        print("Copy .env.example menjadi .env, lalu isi OPENAI_API_KEY.")
        return

    question = "Kalau order internal masuk saat weekend, kapan batas SLA-nya?"

    store = OpenAIEmbeddingVectorStore(index_path=index_path).load()
    contexts = store.search(question, k=4)

    print("\nPERTANYAAN")
    print(question)

    print("\nTOP CONTEXT")
    print(build_context_block(contexts))

    print("\nJAWABAN")
    print(answer_from_context(question, contexts))

    print("\nSUMMARY SQL - LATE ORDER PER CHANNEL")
    print(late_orders_summary(db_path).to_string(index=False))


if __name__ == "__main__":
    main()
