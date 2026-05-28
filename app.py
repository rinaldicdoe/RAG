from pathlib import Path
import os
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
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
    generate_answer_with_openai,
    late_orders_summary,
)

st.set_page_config(page_title="RAG KB - OpenAI Embeddings", layout="wide")

st.title("Knowledge Base Internal - OpenAI Embeddings")
st.caption("Prototype RAG: PDF + Excel + SQLite dengan OpenAI Embeddings")

index_path = ROOT / "data" / "vector_store" / "openai_embeddings_index.json"
db_path = ROOT / "data" / "retail_bi.db"

with st.sidebar:
    st.header("Status")
    st.write("Index tersedia:", index_path.exists())
    st.write("API key tersedia:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("Embedding model:", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    st.divider()
    st.markdown("**Urutan menjalankan:**")
    st.code("python scripts/build_index.py\nstreamlit run app.py", language="bash")

if not index_path.exists():
    st.warning("Index embedding belum ada. Jalankan dulu: python scripts/build_index.py")
    st.stop()

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY belum ada. Copy .env.example menjadi .env, lalu isi API key.")
    st.stop()

question = st.text_input(
    "Pertanyaan",
    value="Kalau order internal masuk saat weekend, kapan batas SLA-nya?",
)

col1, col2 = st.columns([1, 1])
with col1:
    top_k = st.slider("Top-K context", min_value=1, max_value=8, value=4)
with col2:
    answer_mode = st.selectbox(
        "Mode jawaban",
        ["Extractive sederhana", "LLM grounded opsional"],
    )

if st.button("Tanya Knowledge Base", type="primary"):
    store = OpenAIEmbeddingVectorStore(index_path=index_path).load()
    contexts = store.search(question, k=top_k)

    st.subheader("Jawaban")
    if answer_mode == "LLM grounded opsional":
        st.write(generate_answer_with_openai(question, contexts))
    else:
        st.write(answer_from_context(question, contexts))

    st.subheader("Konteks yang ditemukan")
    for i, context in enumerate(contexts, start=1):
        meta = context["metadata"]
        with st.expander(f"Sumber {i}: {meta.get('source')} | score {context['score']}"):
            st.json(meta)
            st.write(context["text"])

st.divider()
st.subheader("Contoh hasil SQL - late order per channel")
try:
    st.dataframe(late_orders_summary(db_path), use_container_width=True)
except Exception as exc:
    st.error(f"Gagal membaca database: {exc}")
