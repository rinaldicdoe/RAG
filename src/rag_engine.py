from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import json
import math
import os
import re
import sqlite3

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

WORD_RE = re.compile(r"[a-zA-Z0-9_\-]+")


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]


def tokenize(text: str) -> List[str]:
    """Tokenisasi ringan untuk keyword metadata/debug, bukan untuk embedding utama."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Memotong teks panjang menjadi beberapa chunk dengan overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size harus > 0")
    if overlap < 0:
        raise ValueError("overlap tidak boleh negatif")
    if overlap >= chunk_size:
        raise ValueError("overlap harus lebih kecil dari chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap

    return chunks


def load_text_docs(folder: str | Path = "data/raw_docs") -> List[Dict[str, Any]]:
    folder = Path(folder)
    docs: List[Dict[str, Any]] = []

    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = path.read_text(encoding="utf-8")
        docs.append({
            "text": text,
            "metadata": {
                "source": path.name,
                "doc_type": path.suffix.lower().replace(".", ""),
                "path": str(path),
            },
        })

    return docs


def load_pdf_docs(folder: str | Path = "data/raw_docs") -> List[Dict[str, Any]]:
    from pypdf import PdfReader

    folder = Path(folder)
    docs: List[Dict[str, Any]] = []

    for path in sorted(folder.glob("*.pdf")):
        reader = PdfReader(str(path))

        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            docs.append({
                "text": text,
                "metadata": {
                    "source": path.name,
                    "doc_type": "pdf",
                    "page": page_no,
                    "path": str(path),
                },
            })

    return docs


def load_excel_rows(path: str | Path, sheet_name: str) -> List[Dict[str, Any]]:
    import pandas as pd

    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name)
    docs: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns]
        docs.append({
            "text": "; ".join(parts),
            "metadata": {
                "source": path.name,
                "doc_type": "excel",
                "sheet": sheet_name,
                "row": int(idx) + 2,
                "path": str(path),
            },
        })

    return docs


def load_all_documents(folder: str | Path = "data/raw_docs", include_excel: bool = True) -> List[Dict[str, Any]]:
    folder = Path(folder)
    docs = load_text_docs(folder)
    docs.extend(load_pdf_docs(folder))

    if include_excel:
        excel_path = folder / "laporan_sales_mei_2026.xlsx"
        if excel_path.exists():
            for sheet in ["Ringkasan_Channel", "Glossary_KPI", "Action_Plan"]:
                docs.extend(load_excel_rows(excel_path, sheet_name=sheet))

    return docs


def make_chunks(
    docs: Iterable[Dict[str, Any]],
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []

    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)):
            meta = dict(doc.get("metadata", {}))
            meta["chunk_id"] = i
            chunks.append(DocumentChunk(text=chunk, metadata=meta))

    return chunks


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity untuk dense embedding vector."""
    if not a or not b:
        return 0.0

    # Jika panjang vector berbeda, zip akan memakai panjang terpendek.
    # Pada penggunaan normal OpenAI model yang sama, panjang vector selalu sama.
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_openai_client():
    """OpenAI client; supports custom gateway via OPENAI_BASE_URL (e.g. Sumopod)."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY belum ditemukan. Copy .env.example menjadi .env, "
            "lalu isi OPENAI_API_KEY terlebih dahulu."
        )

    from openai import OpenAI

    kwargs: Dict[str, Any] = {"api_key": os.getenv("OPENAI_API_KEY")}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")

    return OpenAI(**kwargs)


class OpenAIEmbeddingVectorStore:
    """
    Vector store sederhana berbasis OpenAI Embeddings.

    Alur:
    1. build(chunks): setiap chunk dikirim ke OpenAI Embeddings API.
    2. save(): embedding chunk disimpan ke file JSON agar tidak embedding ulang.
    3. load(): index dimuat ulang dari JSON.
    4. search(question): pertanyaan di-embed, lalu dibandingkan dengan semua chunk.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        index_path: str | Path = "data/vector_store/openai_embeddings_index.json",
        batch_size: int = 32,
        dimensions: Optional[int] = None,
    ):
        self.model = model
        self.index_path = Path(index_path)
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.items: List[Dict[str, Any]] = []

    def _client(self):
        return make_openai_client()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Membuat embedding untuk list teks menggunakan OpenAI Embeddings API."""
        if not texts:
            return []

        client = self._client()
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        response = client.embeddings.create(**payload)
        return [item.embedding for item in response.data]

    def build(self, chunks: List[DocumentChunk]) -> "OpenAIEmbeddingVectorStore":
        """Membuat index embedding dari semua chunk."""
        self.items = []
        total = len(chunks)

        for start in range(0, total, self.batch_size):
            batch = chunks[start:start + self.batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embed_texts(texts)

            for chunk, embedding in zip(batch, embeddings):
                self.items.append({
                    "id": _hash_text(chunk.text + json.dumps(chunk.metadata, sort_keys=True)),
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                })

            print(f"Embedded chunk {min(start + self.batch_size, total)}/{total}")

        return self

    def save(self) -> None:
        """Menyimpan index embedding ke JSON."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "dimensions": self.dimensions,
            "total_items": len(self.items),
            "items": self.items,
        }

        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> "OpenAIEmbeddingVectorStore":
        """Memuat index embedding dari JSON."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index belum ditemukan: {self.index_path}. "
                "Jalankan: python scripts/build_index.py"
            )

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.model = payload.get("model", self.model)
        self.dimensions = payload.get("dimensions", self.dimensions)
        self.items = payload.get("items", [])
        return self

    def search(self, question: str, k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Mencari chunk paling relevan berdasarkan cosine similarity embedding."""
        if not self.items:
            return []

        query_embedding = self.embed_texts([question])[0]

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in self.items:
            score = cosine_similarity(query_embedding, item["embedding"])
            if score >= min_score:
                scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])

        results: List[Dict[str, Any]] = []
        for score, item in scored[:k]:
            results.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "score": round(float(score), 4),
            })

        return results


def build_context_block(contexts: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for i, context in enumerate(contexts, start=1):
        meta = context.get("metadata", {})
        source = meta.get("source", "unknown")
        chunk_id = meta.get("chunk_id", "-")
        page = meta.get("page")
        sheet = meta.get("sheet")
        row = meta.get("row")
        score = context.get("score")

        location = f"chunk {chunk_id}"
        if page:
            location += f", halaman {page}"
        if sheet:
            location += f", sheet {sheet}"
        if row:
            location += f", row {row}"

        score_text = f" | score {score}" if score is not None else ""
        lines.append(f"[Sumber {i}: {source} | {location}{score_text}]\n{context['text']}")

    return "\n\n".join(lines)


def answer_from_context(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Jawaban sederhana tanpa chat model.
    Untuk training, ini membuat output grounded tanpa panggilan LLM tambahan.
    """
    if not contexts:
        return "Informasi belum ditemukan dalam dokumen."

    best = contexts[0]
    meta = best.get("metadata", {})
    source = meta.get("source", "unknown")
    chunk_id = meta.get("chunk_id", "-")
    page = meta.get("page")
    sheet = meta.get("sheet")
    row = meta.get("row")

    location = f"chunk {chunk_id}"
    if page:
        location += f", halaman {page}"
    if sheet:
        location += f", sheet {sheet}"
    if row:
        location += f", row {row}"

    excerpt = best["text"].strip().replace("\n", " ")
    if len(excerpt) > 700:
        excerpt = excerpt[:700].rstrip() + "..."

    return (
        "Jawaban berdasarkan konteks paling relevan:\n"
        f"{excerpt}\n\n"
        f"Sumber utama: {source} | {location}\n"
        f"Pertanyaan: {question}"
    )


def generate_answer_with_openai(
    question: str,
    contexts: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> str:
    model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    """Opsional: gunakan chat model untuk menyusun jawaban dari konteks retrieval."""
    if not contexts:
        return "Informasi belum ditemukan dalam dokumen."

    client = OpenAIEmbeddingVectorStore()._client()
    context_block = build_context_block(contexts)

    prompt = f"""
Anda adalah AI Knowledge Base internal perusahaan.
Jawab hanya berdasarkan KONTEKS berikut.
Jika jawaban tidak ada dalam konteks, katakan: Informasi belum ditemukan dalam dokumen.
Cantumkan sumber dokumen yang digunakan.

KONTEKS:
{context_block}

PERTANYAAN:
{question}
""".strip()

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text


def validate_select_only(sql: str) -> str:
    cleaned = sql.strip().strip(chr(96)).strip()
    cleaned = re.sub(r"^sql\s*", "", cleaned, flags=re.I).strip()

    if not re.match(r"^select\b", cleaned, flags=re.I):
        raise ValueError("Hanya query SELECT yang boleh dieksekusi.")

    forbidden = [
        "insert", "update", "delete", "drop", "alter", "create",
        "replace", "truncate", "attach", "pragma",
    ]
    lowered = cleaned.lower()
    for word in forbidden:
        if re.search(rf"\b{word}\b", lowered):
            raise ValueError(f"Query mengandung keyword terlarang: {word}")

    return cleaned


def run_sql_query(db_path: str | Path, sql: str):
    import pandas as pd

    sql = validate_select_only(sql)
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con)


def late_orders_summary(db_path: str | Path):
    sql = """
    SELECT
        channel,
        COUNT(*) AS total_order,
        SUM(CASE WHEN packed_at > sla_deadline THEN 1 ELSE 0 END) AS late_order,
        ROUND(
            100.0 * SUM(CASE WHEN packed_at > sla_deadline THEN 1 ELSE 0 END)
            / COUNT(*),
            2
        ) AS late_rate_pct,
        SUM(revenue) AS revenue
    FROM orders
    GROUP BY channel
    ORDER BY late_rate_pct DESC
    """
    return run_sql_query(db_path, sql)
