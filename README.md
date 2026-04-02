# RAG Framework

A modular, pluggable Retrieval-Augmented Generation (RAG) framework with hybrid search, full pipeline observability, and a Streamlit developer studio.

> **Not sure what RAG is?** RAG (Retrieval-Augmented Generation) lets you ask questions about your own documents. You upload files (PDFs, Word docs, web pages), and the system finds the most relevant passages and uses an LLM to write a grounded answer with citations.

---

## Table of Contents

1. [What You'll Need](#1-what-youll-need)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Your First Run — Streamlit Studio](#4-your-first-run--streamlit-studio)
5. [Step-by-Step: Ingest → Query → Evaluate](#5-step-by-step-ingest--query--evaluate)
6. [CLI Usage](#6-cli-usage)
7. [Docker](#7-docker)
8. [MCP Server (Claude Integration)](#8-mcp-server-claude-integration)
9. [External Connectors](#9-external-connectors)
10. [Evaluation Results](#10-evaluation-results)
11. [Project Structure](#11-project-structure)
12. [Running Tests](#12-running-tests)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What You'll Need

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 or 3.12 | [Download](https://www.python.org/downloads/) |
| OpenAI API key | — | [Get one here](https://platform.openai.com/api-keys) — used for embeddings and answer generation |
| Git | Any | To clone this repo |
| Docker (optional) | 24+ | Only needed for the Docker setup path |

You do **not** need a GPU. Everything runs on CPU.

---

## 2. Installation

### Option A — Local (recommended for development)

```bash
# 1. Clone the repo
git clone <repo-url>
cd rag-framework

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
#    On macOS / Linux:
source .venv/bin/activate
#    On Windows:
.venv\Scripts\activate

# 4. Install the framework
pip install -e ".[dev]"

# 5. Install optional extras (needed for RAGAS answer quality evaluation)
pip install ragas langchain langchain-openai

# 6. Install optional document parsers (Word / Excel support)
pip install python-docx openpyxl
```

### Option B — Docker

See [Section 7 — Docker](#7-docker).

---

## 3. Configuration

### 3.1 Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```dotenv
# Required
OPENAI_API_KEY=sk-...          # Your OpenAI API key

# Optional — only needed if you use Voyage reranking
VOYAGE_API_KEY=pa-...

# Optional — storage location (defaults to ./data)
RAG_DATA_DIR=./data
```

> ⚠️ Never commit your `.env` file. It is already in `.gitignore`.

### 3.2 Settings overview

All pipeline settings live in `configs/settings.yaml`. You can edit them directly or create a profile override. The main settings you might want to change:

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `embedding.provider` | `openai` | Which embedding model to use (`openai` or `multilingual`) |
| `retrieval.bm25_top_k` | `20` | How many BM25 candidates to retrieve |
| `generation.llm_model` | `gpt-4o-mini` | Which OpenAI model generates answers |
| `ingestion.chunk_max_tokens` | `400` | Max tokens per chunk |
| `reranking.enabled` | `true` | Whether to rerank results |

---

## 4. Your First Run — Streamlit Studio

The easiest way to use the framework is through the browser UI:

```bash
# Make sure your .env is set up, then:
streamlit run rag/app/studio/studio.py
```

Open **http://localhost:8501** in your browser. You'll see 4 pages in the sidebar:

| Page | What it does |
|------|-------------|
| **Ingest & Inspect** | Upload documents and browse the chunks created from them |
| **Query Traces** | Ask questions and see the full retrieval pipeline step-by-step |
| **Evaluation Panel** | Run retrieval quality benchmarks and RAGAS answer quality scoring |
| **Connector Sync** | Connect Email, Slack, Notion, or Google Docs as document sources |

---

## 5. Step-by-Step: Ingest → Query → Evaluate

### Step 1 — Upload a document

1. Go to **Ingest & Inspect** in the sidebar
2. Click **Upload Files** and select a PDF, Word doc, or text file
3. Choose your embedding provider (`openai` recommended)
4. Click **Ingest** — you'll see chunk count, token stats, and a success message

You can also ingest from a URL (GitHub, web pages) using the **From URL** tab.

### Step 2 — Ask a question

1. Go to **Query Traces**
2. Type your question in the query box
3. Select the same embedding provider you used during ingestion
4. Click **Run Query**
5. You'll see:
   - The generated answer with citations
   - Which chunks were retrieved (BM25 vs vector vs hybrid)
   - Reranking scores
   - Latency and token usage

### Step 3 — Check retrieval quality

1. Go to **Evaluation Panel**
2. Select an evaluation suite:
   - **Example Queries** — quick regression test (note: circular ground truth)
   - **Resume Gold Eval** — 30 human-labeled queries, non-circular
3. Click **▶ Run Evaluation**
4. See Recall@K, MRR, nDCG, and per-query results

### Step 4 — Check answer quality (RAGAS)

1. Still on **Evaluation Panel**, click the **Answer Quality (RAGAS)** tab
2. Set embedding provider and LLM model
3. Click **▶ Run RAGAS Evaluation**
4. See faithfulness, answer relevancy, and context precision scores

---

## 6. CLI Usage

You can also use the framework from the terminal without the UI:

```bash
# Ingest a single file
python -m rag.cli.ingest --path /path/to/document.pdf

# Ingest into a named collection (keeps corpora separate)
python -m rag.cli.ingest --path /path/to/resume.pdf --collection resumes

# Ask a question
python -m rag.cli.query --question "What is retrieval-augmented generation?"

# Run an evaluation suite
python -m rag.cli.eval --suite resume_qrels

# Run RAGAS answer quality evaluation
python -m rag.cli.eval --answer-quality
```

---

## 7. Docker

Docker lets you run the full system without installing Python dependencies manually.

### Prerequisites

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### Start

```bash
# 1. Set up your environment file
cp .env.example .env
# Edit .env with your API keys

# 2. Build and start
docker-compose up --build
```

This starts two services:
- **Studio UI** → http://localhost:8501
- **MCP Server** → http://localhost:8000

Your data (database + indexes) is stored in a Docker volume so it persists across restarts.

### Stop

```bash
docker-compose down
```

### Rebuild after code changes

```bash
docker-compose up --build
```

---

## 8. MCP Server (Claude Integration)

The MCP server exposes the RAG pipeline as tools that Claude (or any MCP-compatible client) can call.

### Available tools

| Tool | What it does |
|------|-------------|
| `rag.ingest` | Ingest a file or URL into the knowledge base |
| `rag.query` | Ask a question and get a grounded answer with citations |
| `retrieve` | Get raw ranked chunks for a query (no LLM generation) |
| `retrieve_with_metadata` | Same as retrieve, but returns source file, page, and chunk ID |
| `list_collections` | List all document collections and their sizes |
| `rag.eval.run` | Run an evaluation suite |
| `rag.sync_source` | Sync documents from an external connector |

### Starting the MCP server

```bash
python -m rag.app.mcp_server.server
```

The server runs on `http://localhost:8000`.

### Using collection scoping

When your corpus has multiple document types, use the `collection` parameter to restrict retrieval to a specific group:

```python
# Only search resumes, not the knowledge base
retrieve(query="What is Rita's GPA?", collection="resumes")
```

---

## 9. External Connectors

The framework can automatically pull documents from external sources. Configure credentials in `.env` and use the **Connector Sync** page in the Studio.

| Connector | Required env vars |
|-----------|------------------|
| Email (IMAP) | `RAG_EMAIL_SERVER`, `RAG_EMAIL_USER`, `RAG_EMAIL_PASSWORD` |
| Slack | `RAG_SLACK_TOKEN`, `RAG_SLACK_CHANNEL_IDS` |
| Notion | `RAG_NOTION_TOKEN`, `RAG_NOTION_DATABASE_IDS` |
| Google Docs | `RAG_GOOGLE_CREDENTIALS_PATH`, `RAG_GOOGLE_FOLDER_IDS` |

---

## 10. Evaluation Results

| Suite | Method | Recall@10 | MRR | nDCG@10 |
|-------|--------|----------:|----:|--------:|
| Resume Gold Eval (BM25 only) | keyword | 0.923 | 0.615 | 0.688 |
| Resume Gold Eval (hybrid + collection scope) | BM25 + FAISS | **0.964** | **0.622** | **0.708** |
| BEIR SciFact | BM25 + FAISS | — | — | **0.703** |

> BEIR SciFact nDCG@10 = 0.703 beats the published ColBERT baseline (0.671).

---

## 11. Project Structure

```
rag-framework/
├── rag/
│   ├── core/
│   │   ├── contracts/      # Data models: Document, Chunk, Answer, Citation
│   │   ├── interfaces/     # Abstract base classes for all pluggable components
│   │   ├── registry/       # Component factories
│   │   └── utils/          # Hashing, batching, token counting
│   ├── pipelines/          # IngestPipeline, QueryPipeline, EvalPipeline
│   ├── infra/
│   │   ├── stores/         # SQLite document + trace store
│   │   ├── parsing/        # PDF, HTML, Markdown, Word, Excel parsers
│   │   ├── chunking/       # Paragraph splitter, chunk packer
│   │   ├── embedding/      # OpenAI, multilingual (local) providers
│   │   ├── indexes/        # FAISS vector index, BM25 keyword index
│   │   ├── rerank/         # Voyage, cross-encoder rerankers
│   │   ├── llm/            # OpenAI LLM client
│   │   ├── connectors/     # Email, Slack, Notion, Google Docs, Web
│   │   └── evaluation/     # RAGAS answer quality evaluator
│   └── app/
│       ├── studio/         # Streamlit UI (4 pages + components)
│       ├── mcp_server/     # MCP tool implementations + schemas
│       └── cli/            # CLI entry points
├── configs/                # YAML settings, prompt templates, profiles
├── tests/                  # 657 tests (unit + integration + e2e)
├── scripts/                # BEIR evaluation runner
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

---

## 12. Running Tests

```bash
# Run all unit and integration tests (fast, no API calls)
pytest tests/ -m "not e2e" -q

# Run everything including end-to-end tests
pytest tests/ -q

# Run a specific test file
pytest tests/test_bm25_local.py -v
```

---

## 13. Troubleshooting

**`OPENAI_API_KEY` not found**
Make sure you've created a `.env` file from `.env.example` and filled in your key. If running via terminal, also check that your shell loaded the file (`source .env` or restart your terminal).

**`KMP_DUPLICATE_LIB_OK` error on macOS**
This is an OpenMP conflict between FAISS and PyTorch. Fix it by adding this to your `.env`:
```
KMP_DUPLICATE_LIB_OK=TRUE
```
Or prefix your command: `KMP_DUPLICATE_LIB_OK=TRUE streamlit run ...`

**RAGAS evaluation fails with "RAGAS is not installed"**
```bash
pip install ragas langchain langchain-openai
```
Then restart Streamlit.

**Chunks not found after re-ingesting documents**
Chunk IDs are content-based hashes. If you change the parser or chunking settings and re-ingest, new chunk IDs are generated. You'll need to regenerate any evaluation fixtures that reference specific chunk IDs (like `resume_qrels.json`).

**Streamlit shows a blank page**
Try a hard refresh (`Cmd+Shift+R` on macOS, `Ctrl+Shift+R` on Windows/Linux). If it persists, restart Streamlit.
