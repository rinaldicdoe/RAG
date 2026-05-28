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

from rag_engine import OpenAIEmbeddingVectorStore, build_context_block, answer_from_context


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Tulis pertanyaan: ").strip()

    index_path = ROOT / "data" / "vector_store" / "openai_embeddings_index.json"

    if not question:
        print("Pertanyaan tidak boleh kosong.")
        return
    if not index_path.exists():
        print("Index belum ada. Jalankan: python scripts/build_index.py")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY belum ditemukan. Isi .env terlebih dahulu.")
        return

    store = OpenAIEmbeddingVectorStore(index_path=index_path).load()
    contexts = store.search(question, k=4)

    print("\nKONTEKS TERAMBIL")
    print(build_context_block(contexts))

    print("\nJAWABAN")
    print(answer_from_context(question, contexts))


if __name__ == "__main__":
    main()
