"""Ingestion Manager page — upload documents and run the ingest pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Ingestion Manager · RAG Studio", layout="wide")

st.title("📥 Ingestion Manager")
st.caption("Upload a document and run it through the ingest pipeline.")

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None
if "ingest_config" not in st.session_state:
    st.session_state.ingest_config = None
if "ingest_error" not in st.session_state:
    st.session_state.ingest_error = None


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    db_path = st.text_input("SQLite DB path", value="data/rag.db")
    index_dir = st.text_input("Index directory", value="data/indexes")
    collection = st.text_input("Collection name", value="default")

    st.divider()
    st.subheader("Embedding")
    embedding_provider = st.selectbox(
        "Provider", ["openai", "none"], index=0
    )
    embedding_model = st.text_input(
        "Model",
        value="text-embedding-3-small",
        disabled=(embedding_provider == "none"),
    )
    vector_dimension = st.number_input(
        "Vector dimension",
        min_value=1,
        max_value=4096,
        value=1536,
        disabled=(embedding_provider == "none"),
    )

    st.divider()
    st.subheader("Index")
    index_type = st.selectbox("Vector index type", ["FAISS (IndexFlatL2)", "none"], index=0)
    token_budget = st.number_input("Chunk token budget", min_value=64, max_value=2048, value=512)


# ---------------------------------------------------------------------------
# Main panel — file upload + run
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a file to ingest",
    type=["txt", "md", "pdf", "html", "htm", "docx", "xlsx"],
    help="Supported formats: plain text, Markdown, PDF, HTML, Word (.docx), Excel (.xlsx)",
)

run_btn = st.button("▶ Run Ingest", type="primary", disabled=(uploaded_file is None))

if run_btn and uploaded_file is not None:
    # Build configuration summary first (before any pipeline work)
    config = {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model if embedding_provider != "none" else "—",
        "vector_dimension": int(vector_dimension) if embedding_provider != "none" else "—",
        "index_type": index_type,
        "token_budget": int(token_budget),
        "collection": collection,
        "db_path": db_path,
        "index_dir": index_dir,
    }
    st.session_state.ingest_config = config
    st.session_state.ingest_result = None
    st.session_state.ingest_error = None

    with st.spinner(f"Ingesting **{uploaded_file.name}** …"):
        try:
            from rag.infra.stores.docstore_sqlite import SQLiteDocStore
            from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
            from rag.pipelines.ingest_pipeline import IngestPipeline

            # Ensure directories exist
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            Path(index_dir).mkdir(parents=True, exist_ok=True)

            doc_store = SQLiteDocStore(db_path)
            trace_store = SQLiteTraceStore(db_path)

            # Optional embedding provider
            embed_provider = None
            vec_index = None
            kw_index = None

            if embedding_provider == "openai":
                from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
                from rag.infra.indexes.bm25_local import BM25LocalIndex
                from rag.infra.indexes.faiss_local import FaissLocalIndex
                from rag.infra.indexes.index_manager import IndexManager

                embed_provider = OpenAIEmbeddingProvider(
                    model=embedding_model,
                    dimensions=int(vector_dimension),
                )

                # Check for dimension mismatch against existing index
                faiss_path = Path(index_dir) / "faiss.index"
                if faiss_path.exists():
                    try:
                        existing = FaissLocalIndex()
                        existing.load(index_dir)
                        if existing.dimension != int(vector_dimension):
                            st.session_state.ingest_error = (
                                f"⚠️ **Dimension mismatch**: existing FAISS index has "
                                f"dimension **{existing.dimension}**, but current "
                                f"configuration uses **{int(vector_dimension)}**. "
                                "Delete or rebuild the index to continue."
                            )
                            st.rerun()
                    except Exception:
                        pass  # Corrupted or incompatible — will be overwritten

                mgr = IndexManager(index_dir=index_dir)
                vec_index = mgr.faiss
                kw_index = mgr.bm25

            # Write upload to a temp file
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            pipeline = IngestPipeline(
                doc_store=doc_store,
                trace_store=trace_store,
                token_budget=int(token_budget),
                embedding_provider=embed_provider,
                vector_index=vec_index,
                keyword_index=kw_index,
            )
            result = pipeline.ingest(tmp_path)

            # Persist indexes if updated
            if vec_index is not None:
                vec_index.save(index_dir)
            if kw_index is not None:
                kw_index.save(index_dir)

            st.session_state.ingest_result = result

        except Exception as exc:
            st.session_state.ingest_error = f"❌ Ingestion failed: {exc}"

    st.rerun()


# ---------------------------------------------------------------------------
# Dimension mismatch error
# ---------------------------------------------------------------------------

if st.session_state.ingest_error:
    st.error(st.session_state.ingest_error)

# ---------------------------------------------------------------------------
# Ingest summary
# ---------------------------------------------------------------------------

result = st.session_state.ingest_result
config = st.session_state.ingest_config

if result is not None and config is not None:
    if result.error:
        st.error(f"❌ Ingestion failed: {result.error}")
    elif result.skipped:
        st.warning(f"⏭️ Document **{Path(result.source_path).name}** was already up-to-date and skipped.")
    else:
        st.success(f"✅ Ingestion complete for **{Path(result.source_path).name}**")

        # ── Summary metrics ───────────────────────────────────────────────
        st.subheader("📋 Ingest Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Blocks", result.block_count)
        col2.metric("Chunks", result.chunk_count)
        col3.metric("Elapsed", f"{result.elapsed_ms:.0f} ms")
        col4.metric("Embed tokens", result.embed_tokens)

        # ── Embedding / index configuration ───────────────────────────────
        st.subheader("🔧 Pipeline Configuration")
        cfg_col1, cfg_col2 = st.columns(2)

        with cfg_col1:
            st.markdown("**Embedding**")
            st.markdown(f"- Provider: `{config['embedding_provider']}`")
            st.markdown(f"- Model: `{config['embedding_model']}`")
            st.markdown(f"- Vector dimension: `{config['vector_dimension']}`")

        with cfg_col2:
            st.markdown("**Index**")
            st.markdown(f"- Vector index type: `{config['index_type']}`")
            st.markdown(f"- Chunk token budget: `{config['token_budget']}`")
            st.markdown(f"- Collection: `{config['collection']}`")

        # ── Run metadata ──────────────────────────────────────────────────
        with st.expander("Run metadata", expanded=False):
            st.json({
                "doc_id": result.doc_id,
                "run_id": result.run_id,
                "source_path": result.source_path,
                "block_count": result.block_count,
                "chunk_count": result.chunk_count,
                "embed_tokens": result.embed_tokens,
                "elapsed_ms": round(result.elapsed_ms, 1),
            })

# ---------------------------------------------------------------------------
# Pipeline overview (always visible at bottom)
# ---------------------------------------------------------------------------

with st.expander("📖 Pipeline overview", expanded=False):
    st.markdown(
        """
        1. **Loader** — read raw bytes from the uploaded file
        2. **Sniffer** — detect MIME type via magic bytes + extension
        3. **Router → Parser** — dispatch to MdParser / HtmlTrafilaturaParser / PdfPyMuPDFParser
        4. **Quality Gate** — filter documents below minimum quality thresholds
        5. **Cleaner** — unicode fix, empty filter, OCR merge, deduplication
        6. **Block Splitter** — map IRBlocks to hashed TextBlocks
        7. **Chunk Packer** — anchor-aware packing with token budget
        8. **Embedding** — embed chunks via configured provider
        9. **DocStore + TraceStore** — persist document, chunks, and run trace
        """
    )
