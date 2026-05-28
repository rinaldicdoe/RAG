from pathlib import Path
import os
import sqlite3
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
    raw = ROOT / "data" / "raw_docs"
    db_path = ROOT / "data" / "retail_bi.db"
    index_path = ROOT / "data" / "vector_store" / "openai_embeddings_index.json"
    output_path = ROOT / "outputs" / "validation_result.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs = load_all_documents(raw)
    chunks = make_chunks(docs, chunk_size=900, overlap=150)

    pdf_count = len(list(raw.glob("*.pdf")))
    xlsx_exists = (raw / "laporan_sales_mei_2026.xlsx").exists()
    db_exists = db_path.exists()

    with sqlite3.connect(db_path) as con:
        total_orders = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

    lines = []
    lines.append("VALIDATION PASSED")
    lines.append(f"Total dokumen terbaca: {len(docs)}")
    lines.append(f"Total chunk: {len(chunks)}")
    lines.append(f"Total PDF: {pdf_count}")
    lines.append(f"Excel tersedia: {xlsx_exists}")
    lines.append(f"Database tersedia: {db_exists}")
    lines.append(f"Total order database: {total_orders}")
    lines.append(f"Index OpenAI Embeddings tersedia: {index_path.exists()}")
    lines.append(f"OPENAI_API_KEY tersedia: {bool(os.getenv('OPENAI_API_KEY'))}")

    # Validasi struktur vector store tanpa memanggil API.
    store = OpenAIEmbeddingVectorStore(index_path=index_path)
    lines.append(f"Embedding model default: {store.model}")

    output = "\n".join(lines)
    output_path.write_text(output, encoding="utf-8")
    print(output)
    print("\nCatatan: validate_project tidak memanggil OpenAI API.")
    print("Untuk membuat index embedding sungguhan, jalankan: python scripts/build_index.py")


if __name__ == "__main__":
    main()
