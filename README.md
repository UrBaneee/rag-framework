# RAG Framework

A modular, pluggable Retrieval-Augmented Generation (RAG) framework for knowledge retrieval, evidence-based answers, and pipeline transparency.

## Features

- Pluggable architecture (embedding, vector index, reranker, LLM, parsers, storage)
- Full pipeline observability via Streamlit interface
- Config-driven behavior
- Incremental document ingestion
- MCP tool integration

## Project Structure

```
rag/
  core/
    contracts/     # Data models (Document, IRBlock, Chunk, etc.)
    interfaces/    # Abstract base classes for all pluggable components
    registry/      # Component factories and registries
    utils/         # Shared utilities
  pipelines/       # Ingestion and query pipeline orchestration
  infra/           # Infrastructure implementations (stores, indexes, parsers)
    connectors/    # External source connectors (V2)
  app/             # Streamlit UI
  cli/             # CLI entry points
configs/
  prompts/         # LLM prompt templates
  connectors/      # Connector-specific configs
tests/             # Test suite
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Ingest documents
python -m rag.cli.ingest --path /path/to/docs

# Query
python -m rag.cli.query --question "What is...?"

# Launch Streamlit UI
streamlit run rag/app/main.py
```
