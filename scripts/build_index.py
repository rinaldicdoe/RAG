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

from rag_engine import load_all_documents, make_chunks, OpenAIEmbeddingVectorStore


def main():
    raw_folder = ROOT / "data" / "raw_docs"
    index_path = ROOT / "data" / "vector_store" / "openai_embeddings_index.json"

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY belum ditemukan. Copy .env.example menjadi .env, "
            "lalu isi OPENAI_API_KEY."
        )

    docs = load_all_documents(raw_folder)
    chunks = make_chunks(docs, chunk_size=900, overlap=150)

    print("=== BUILD INDEX OPENAI EMBEDDINGS ===")
    print("Total dokumen:", len(docs))
    print("Total chunk:", len(chunks))
    print("Model embedding:", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    store = OpenAIEmbeddingVectorStore(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        index_path=index_path,
        batch_size=32,
    )
    store.build(chunks)
    store.save()

    print("\nIndex berhasil dibuat:", index_path)
    print("Setelah ini jalankan: python scripts/quick_start.py")


if __name__ == "__main__":
    main()
