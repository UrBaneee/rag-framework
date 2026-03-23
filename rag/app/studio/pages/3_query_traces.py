"""Query Traces page — run queries and inspect retrieval, rerank, and generation."""

import streamlit as st

st.set_page_config(page_title="Query Traces · RAG Studio", layout="wide")

st.title("💬 Query Traces")
st.caption("Ask questions against your corpus and inspect the full retrieval pipeline.")

st.info(
    "**Coming soon** — full query UI (Task 8.4).\n\n"
    "This page will let you submit queries and inspect: retrieved chunks with "
    "BM25/vector/RRF scores, source attribution, reranking changes, context "
    "packing details, the final answer with inline citations, and generation "
    "token usage.",
    icon="🚧",
)

with st.expander("Query pipeline stages", expanded=False):
    st.markdown(
        """
        A query flows through the following pipeline stages:

        1. **BM25 Retrieval** — keyword search over the indexed chunks
        2. **Vector Retrieval** *(optional)* — approximate nearest-neighbour search via FAISS
        3. **Source Attribution** — merge results and label as BM25-only / vector-only / hybrid
        4. **RRF Fusion** — Reciprocal Rank Fusion across both result lists
        5. **Reranker** *(optional)* — cross-encoder reranking (e.g. Voyage AI)
        6. **Context Packer** — select top-k unique chunks within token budget
        7. **Prompt Builder** — assemble grounded QA prompt with citation map
        8. **LLM Generation** — call the configured LLM and parse the answer
        9. **Answer** — grounded text with inline `[N]` citation markers
        """
    )
