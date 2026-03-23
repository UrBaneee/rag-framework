"""Ingestion Manager page — upload documents and run the ingest pipeline."""

import streamlit as st

st.set_page_config(page_title="Ingestion Manager · RAG Studio", layout="wide")

st.title("📥 Ingestion Manager")
st.caption("Upload a document and run it through the ingest pipeline.")

st.info(
    "**Coming soon** — full ingestion UI (Task 8.2).\n\n"
    "This page will allow you to upload files, select a collection, "
    "run the ingest pipeline, and view the ingest summary including "
    "embedding provider, model, vector dimension, and index configuration.",
    icon="🚧",
)

with st.expander("Pipeline overview", expanded=False):
    st.markdown(
        """
        The ingest pipeline runs the following stages:

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
