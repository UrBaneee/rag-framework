"""Evaluation Panel page — run and review RAG evaluation metrics."""

import streamlit as st

st.set_page_config(page_title="Evaluation Panel · RAG Studio", layout="wide")

st.title("📊 Evaluation Panel")
st.caption("Run evaluation suites and review RAG quality metrics.")

st.info(
    "**Coming soon** — full evaluation UI (Phase 12).\n\n"
    "This page will let you load an evaluation dataset, run the retrieval and "
    "generation pipeline against it, and review metrics including Recall@k, "
    "MRR, faithfulness, and answer relevance.",
    icon="🚧",
)

with st.expander("Planned evaluation metrics", expanded=False):
    st.markdown(
        """
        **Retrieval metrics**
        - `Recall@k` — fraction of relevant chunks retrieved in top-k
        - `MRR` — Mean Reciprocal Rank of the first relevant result
        - `NDCG@k` — Normalised Discounted Cumulative Gain

        **Generation metrics**
        - `Faithfulness` — are all answer claims supported by cited chunks?
        - `Answer relevance` — does the answer address the question?
        - `Citation precision` — do cited chunks actually support the claim?

        **System metrics**
        - Average retrieval latency
        - Average generation latency
        - Token cost per query
        """
    )
