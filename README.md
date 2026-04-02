# RAG Framework

A modular, pluggable Retrieval-Augmented Generation (RAG) framework with hybrid search, full pipeline observability, and a Streamlit developer studio.

## Features

- **Hybrid retrieval** — BM25 + FAISS with Reciprocal Rank Fusion (RRF)
- **Collection scoping** — isolate retrieval to a named corpus partition
- **Pluggable architecture** — swap embeddings, LLMs, rerankers, parsers, indexes
- **Streamlit Studio** — ingest, inspect chunks, run queries, evaluate, manage connectors
- **MCP tools** — `rag.query`, `retrieve`, `retrieve_with_metadata`, `list_collections`
- **External connectors** — Email (IMAP), Slack, Notion, Google Docs
- **Evaluation** — Recall@K, MRR, nDCG, RAGAS answer quality
- **Incremental ingestion** — document fingerprinting, block-level diffs

## Quick Start (Docker)

```bash
git clone <repo-url>
cd rag-framework
cp .env.example .env        # fill in your API keys
docker-compose up --build
```

Open **http://localhost:8501** for the Studio UI.

## Quick Start (Local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install ragas langchain langchain-openai  # for answer quality eval

cp .env.example .env        # fill in your API keys
streamlit run rag/app/studio/studio.py
```

## Environment Variables

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (embeddings + LLM) |
| `VOYAGE_API_KEY` | Optional | Voyage AI reranker |
| `RAG_DATA_DIR` | Optional | Data directory (default: `./data`) |

## CLI Usage

```bash
# Ingest a document
python -m rag.cli.ingest --path /path/to/doc.pdf --collection resumes

# Query
python -m rag.cli.query --question "What is Rita Ouyang's most recent job?"

# Run evaluation suite
python -m rag.cli.eval --suite resume_qrels
```

## Evaluation Results

| Suite | Recall@10 | MRR | nDCG@10 |
|-------|----------:|----:|--------:|
| resume_qrels (hybrid + collection scope) | **0.964** | 0.622 | 0.708 |
| BEIR SciFact (BM25 + FAISS) | — | — | **0.703** |

## Project Structure

```
rag/
├── core/          # Contracts, interfaces, registry
├── pipelines/     # IngestPipeline, QueryPipeline, EvalPipeline
├── infra/         # Stores, indexes, parsers, embeddings, connectors, LLM
└── app/
    ├── studio/    # Streamlit UI (4 pages)
    ├── mcp_server/ # MCP tool implementations
    └── cli/       # CLI entry points
tests/             # 657 tests (unit + integration + e2e)
```

## Running Tests

```bash
pytest tests/ -m "not e2e" -q
```
