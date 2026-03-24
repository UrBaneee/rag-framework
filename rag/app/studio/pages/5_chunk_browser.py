"""Chunk Browser — inspect all chunks stored in the DocStore."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Chunk Browser · RAG Studio", layout="wide")

st.title("🧩 Chunk Browser")
st.caption("Inspect every chunk produced by the ingestion pipeline — text, token counts, and metadata.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    db_path = st.text_input("SQLite DB path", value="data/rag.db")

    st.divider()
    st.subheader("Filter")
    sort_by = st.selectbox("Sort by", ["Ingestion order", "Token count ↑", "Token count ↓"])
    search_term = st.text_input("Search text", placeholder="keyword in chunk text…")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def _load_data(db_path: str):
    """Load documents and chunks from SQLite. Cached for 10 seconds."""
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    store = SQLiteDocStore(db_path)
    docs = store.list_documents()
    chunks = store.get_all_chunks()
    return docs, chunks


if not Path(db_path).exists():
    st.info("No database found at the configured path. Ingest a document first.")
    st.stop()

try:
    docs, chunks = _load_data(db_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if not chunks:
    st.info("No chunks found. Ingest a document to populate the index.")
    st.stop()

# ---------------------------------------------------------------------------
# Document selector
# ---------------------------------------------------------------------------

doc_options = {"All documents": None}
for doc in docs:
    label = Path(doc["source_path"]).name or doc["doc_id"][:16]
    doc_options[f"{label}  —  {doc['doc_id'][:12]}…"] = doc["doc_id"]

selected_label = st.selectbox("Document", list(doc_options.keys()))
selected_doc_id = doc_options[selected_label]

# ---------------------------------------------------------------------------
# Filter + sort
# ---------------------------------------------------------------------------

filtered = [c for c in chunks if selected_doc_id is None or c.doc_id == selected_doc_id]

if search_term.strip():
    q = search_term.strip().lower()
    filtered = [c for c in filtered if q in (c.display_text or "").lower()
                or q in (c.stable_text or "").lower()]

if sort_by == "Token count ↑":
    filtered.sort(key=lambda c: c.token_count)
elif sort_by == "Token count ↓":
    filtered.sort(key=lambda c: c.token_count, reverse=True)
# default: ingestion order (already ordered by rowid from DB)

# ---------------------------------------------------------------------------
# Stats bar
# ---------------------------------------------------------------------------

if filtered:
    token_counts = [c.token_count for c in filtered]
    avg_tokens = sum(token_counts) / len(token_counts)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Chunks", len(filtered))
    s2.metric("Documents", len({c.doc_id for c in filtered}))
    s3.metric("Avg tokens", f"{avg_tokens:.0f}")
    s4.metric("Min tokens", min(token_counts))
    s5.metric("Max tokens", max(token_counts))

    # Token distribution mini-bar
    buckets = {"0–128": 0, "129–256": 0, "257–384": 0, "385–512": 0, "512+": 0}
    for t in token_counts:
        if t <= 128:
            buckets["0–128"] += 1
        elif t <= 256:
            buckets["129–256"] += 1
        elif t <= 384:
            buckets["257–384"] += 1
        elif t <= 512:
            buckets["385–512"] += 1
        else:
            buckets["512+"] += 1

    with st.expander("📊 Token distribution", expanded=False):
        import pandas as pd
        df = pd.DataFrame({"Range": list(buckets.keys()), "Count": list(buckets.values())})
        st.bar_chart(df.set_index("Range"))

st.divider()

# ---------------------------------------------------------------------------
# Chunk table
# ---------------------------------------------------------------------------

st.subheader(f"🗂️ Chunks ({len(filtered)})")

if not filtered:
    st.info("No chunks match the current filters.")
else:
    for i, chunk in enumerate(filtered, start=1):
        doc_name = Path(
            next((d["source_path"] for d in docs if d["doc_id"] == chunk.doc_id), chunk.doc_id)
        ).name or chunk.doc_id[:12]

        # Token count badge colour
        t = chunk.token_count
        if t <= 256:
            badge = f"🟢 {t} tok"
        elif t <= 450:
            badge = f"🟡 {t} tok"
        else:
            badge = f"🔴 {t} tok"

        preview = (chunk.display_text or "")[:80].replace("\n", " ")
        label = (
            f"**#{i}** · `{(chunk.chunk_id or '')[:12]}…` · "
            f"📄 {doc_name} · {badge}"
        )

        with st.expander(label, expanded=False):
            col_meta, col_text = st.columns([1, 2])

            with col_meta:
                st.markdown("**Chunk ID**")
                st.code(chunk.chunk_id or "—", language=None)
                st.markdown("**Doc ID**")
                st.code(chunk.doc_id[:32] + "…", language=None)
                st.markdown(f"**Tokens:** `{chunk.token_count}`")
                if chunk.metadata:
                    pages = chunk.metadata.get("start_page") or chunk.metadata.get("page")
                    if pages:
                        st.markdown(f"**Page:** `{pages}`")
                    st.json(chunk.metadata, expanded=False)

            with col_text:
                st.markdown("**Chunk text:**")
                st.markdown(
                    f"""<div style="
                        background:#f8f9fa;
                        border-left:3px solid #dee2e6;
                        padding:10px 14px;
                        border-radius:4px;
                        font-size:0.9em;
                        white-space:pre-wrap;
                        word-break:break-word;
                    ">{chunk.display_text or chunk.stable_text}</div>""",
                    unsafe_allow_html=True,
                )
