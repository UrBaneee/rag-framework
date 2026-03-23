# Project: Pluggable RAG Knowledge Retrieval Framework

**Author:** (Your Name)
**Status:** Architecture Finalized — Implementation Phase Ready

---

## 1. Project Overview

This project implements a modular, pluggable Retrieval-Augmented Generation (RAG) framework designed for:

- knowledge retrieval
- evidence-based answers
- pipeline transparency
- incremental document ingestion
- configurable infrastructure

The system receives user questions via MCP tools, retrieves relevant document fragments using hybrid search, optionally reranks them, and generates grounded answers with citations.

Unlike many demo-style RAG implementations, this framework is designed as an engineering-grade system emphasizing:

- **traceability** — every pipeline step is observable
- **incremental ingestion** — large documents update efficiently
- **pluggability** — infrastructure components can be swapped
- **config-driven behavior** — runtime behavior controlled via configuration
- **clear system boundaries** between ingestion, retrieval, and generation

The framework is intended to support both:

- local experimentation
- future enterprise deployment

---

## 2. System Positioning

This project should be understood as a:

> **Traceable, modular RAG system with conditional orchestration capabilities**

rather than a minimal retrieval demo.

The architecture focuses on:

- deterministic pipelines
- configurable retrieval strategies
- explicit pipeline tracing
- grounded answer synthesis

The system includes agent-like orchestration features in specific stages:

- ingestion routing with quality-gated fallback
- conditional reranking strategies
- configurable generation policies

However, it does not yet implement full Agentic RAG capabilities, such as:

- query planning or decomposition
- multi-hop retrieval loops
- autonomous tool chaining
- self-reflection or answer verification

These capabilities can be added later without major architectural changes.

---

## 3. Core Design Goals

### 3.1 Pluggable Architecture

Every major system component can be replaced through interfaces.

Pluggable modules include:

| Component | Example Providers |
|-----------|------------------|
| Embedding | OpenAI / Voyage / Cohere / Ollama |
| Vector Index | FAISS / Milvus / Qdrant |
| Keyword Index | BM25 / OpenSearch |
| Reranker | Voyage / Cohere / Cross-Encoder |
| LLM Client | OpenAI / Azure / Ollama |
| Parsers | PyMuPDF / Trafilatura / OCR |
| Storage | SQLite / Postgres |

This enables future deployment flexibility without rewriting pipelines.

---

### 3.2 Local-First Development

The system is optimized for development on a local machine.

Initial environment:

- Apple M2
- 16GB RAM
- No dedicated GPU required

Future deployment targets include:

- Docker containers
- Cloud environments
- enterprise internal infrastructure

---

### 3.3 Full Pipeline Observability

The framework exposes complete pipeline traces through a Streamlit interface.

Observable stages include:

- ingestion routing decisions
- parser selection and fallback
- chunk creation
- retrieval candidates
- reranking changes
- answer generation

This allows developers to inspect how answers were produced, not just what the answer is.

---

### 3.4 Config-Driven Behavior

System behavior is controlled through layered configuration files.

Configuration controls:

- retrieval parameters
- reranking policies
- generation rules
- parser routing strategies
- evaluation metrics
- tenant-specific providers

This avoids hard-coding pipeline behavior.

---

## 4. Document Ingestion System

The ingestion pipeline processes heterogeneous document formats and converts them into a unified intermediate representation.

Pipeline stages:

```
Loader
 ↓
Sniffer
 ↓
Router
 ↓
Parser Plugins
 ↓
Unified IR
 ↓
Cleaner
 ↓
Chunking
 ↓
Embedding
 ↓
Indexing
```

Supported formats (scope-aligned) include:

| Format | Parser | V1 | V2 |
|--------|--------|----|----|
| PDF | PyMuPDF | ✅ | |
| HTML | Trafilatura | ✅ | |
| TXT / Markdown | md_parser | ✅ | |
| DOCX | python-docx | | ✅ |
| PPTX | python-pptx | | ✅ |
| Scanned PDF | PaddleOCR | | ✅ |
| Images | OCR | | ✅ |

For V1 main path, parser routing should only guarantee PDF/HTML/TXT/Markdown end-to-end support, and return explicit `unsupported_format` results for formats whose parsers are not yet implemented.

### 4.1 External Source Connectors (V2)

In addition to local files, V2 adds connector-based ingestion sources:

- Email (IMAP/Gmail API)
- Slack (channels/threads)
- Notion (pages/databases)
- Google Docs (Docs/Drive APIs)

Connector sync model:

- pull remote content into raw artifacts on a schedule
- normalize all sources through the same parser → cleaner → chunking pipeline
- maintain stable `source_id` + cursor/checkpoint for incremental sync
- propagate source metadata and ACL hints for downstream filtering

For V1 main path, these connectors are out of scope.

Parser routing uses:

- magic byte detection
- MIME detection
- heuristic classification
- parse quality gates
- automatic fallback

---

## 5. Unified Intermediate Representation

All parsers output a block-based intermediate representation.

```
Document
  ├── metadata
  ├── blocks
  └── parse_report
```

Each block contains:

- `block_type`
- `text`
- `page`
- bounding box
- confidence
- section path

Benefits:

- format-independent processing
- consistent cleaning
- structure-aware chunking

---

## 6. Cleaning Pipeline

Noise removal occurs before chunking.

Cleaning steps:

1. Unicode normalization
2. Empty paragraph removal
3. Duplicate paragraph removal
4. PDF header/footer filtering
5. HTML navigation/footer filtering
6. OCR line reconstruction

These steps reduce noise and improve embedding quality.

---

## 7. Smart Chunking

Chunking is structure-aware rather than naive text splitting.

Enhancements include:

- anchor detection (headings, lists, section numbers)
- semantic chunk refinement
- pronoun ambiguity detection

**Example improvement:**

Original chunk:
```
3. Cloud Deployment:
All services are deployed on AWS/Azure.
```

Refined chunk:
```
Deployment Mode 3 — Cloud Deployment:
All services are deployed on AWS/Azure.
```

This improves chunk readability and retrieval relevance.

---

## 8. Optional Semantic Metadata Enrichment (V2)

Semantic metadata enrichment is optional in V1 and planned as a V2 capability.

When enabled, each chunk receives additional metadata:

- `title`
- `summary`
- `tags`

These are generated automatically to support:

- hybrid retrieval
- metadata filtering
- improved UI presentation

---

## 9. Incremental Document Ingestion

Large documents are ingested incrementally to avoid unnecessary re-embedding.

The system uses a layered identity strategy.

### 9.1 Document Fingerprint

A document fingerprint identifies whether a file version has changed.

Sources may include:

- source path
- file metadata
- raw content hash

This enables fast detection of unchanged files.

---

### 9.2 Block-Level Hashing

The minimal stable unit for incremental updates is a cleaned paragraph block.

Hash computation (deterministic across runs):

```python
# canonicalize: preserve case + collapse whitespace + strip
# defined in rag/core/utils/hashing.py and reused everywhere
def canonicalize(text: str) -> str:
    return " ".join(text.split())

block_hash = sha256(
    canonicalize(normalized_clean_text).encode("utf-8")
).hexdigest()
```

`canonicalize()` must be the **single shared implementation** in `rag/core/utils/hashing.py`. Do not inline or re-implement it elsewhere — any deviation will cause the same block to produce a different hash across runs, silently breaking incremental update.  
Case normalization (for example `.lower()`) should be opt-in via config, not default behavior.

Hashing occurs:

- after cleaning
- before refinement
- before metadata enrichment

This ensures cosmetic changes do not trigger unnecessary updates.
Do not use Python built-in `hash()` for persistent identities.

---

### 9.3 Chunk Signatures

Chunks are defined as ordered combinations of blocks.

Chunk identity is computed using:

```python
chunk_signature = sha256(
    "|".join(block_hash_sequence).encode("utf-8")
).hexdigest()
```

This allows the system to detect when chunk composition changes.

---

### 9.4 Handling Location Changes

If chunk content remains unchanged but its document location shifts (for example due to page insertion):

- embeddings are reused
- chunk identity remains unchanged
- only location metadata (citations) is updated

This separates **content identity** from **document location**, and avoids unnecessary embedding recomputation.

---

### 9.5 Resynchronization

When block hashes diverge between document versions, a resynchronization algorithm attempts to realign chunk boundaries.

The resync algorithm:

- scans forward to find matching block sequences
- reuses embeddings when possible
- limits recomputation to changed regions

This minimizes embedding costs for large documents.

---

## 10. Retrieval Pipeline

The query pipeline combines multiple retrieval strategies.

```
User Query
   ↙                 ↘
BM25 Retrieval    Vector Retrieval
 (top_k=20)         (top_k=20)
   ↘                 ↙
Reciprocal Rank Fusion (RRF, k=60, pool=40)
   ↓
Cross-Encoder Rerank (top_k=8)
   ↓
Optional LLM Rerank (V2)
   ↓
Context Packing (top_k=3)
   ↓
Answer Generation
   ↓
Answer + Citations
```

**Retrieval configuration:**

| Stage | Default |
|-------|---------|
| BM25 candidates | 20 |
| Vector candidates | 20 |
| Fusion pool | 40 |
| Cross-encoder output | 8 |
| Generation context | 3 |

**Design rationale:**

- RRF k=60 follows academic recommendations as a stable default across diverse datasets
- BM25 + vector separation ensures complementary coverage (keyword precision vs. semantic recall)
- Cross-encoder reranking is applied after fusion, not before, to avoid scoring candidates in isolation

---

## 11. Answer Generation

Answer generation is strictly grounded in retrieved context.

Rules:

- answers must rely only on retrieved evidence
- the system abstains when evidence is insufficient
- citations are mandatory

Answer generation components:

- `ContextPacker`
- `PromptBuilder`
- `LLMClient`
- `AnswerComposer`

**Citation style:**

Inline citations:
```
Cloud deployment requires container orchestration.[1]
```

Sources section:
```
Sources:
[1] architecture.pdf — page 12
```

---

## 12. Evaluation Framework

Evaluation is divided into four layers (the 4th is future-facing).

### 12.1 Retrieval Quality

Metrics:

- Recall@K
- MRR
- nDCG

These evaluate retrieval and reranking effectiveness.

---

### 12.2 System Efficiency

Metrics include:

- `skipped_chunks`
- `changed_chunks`
- `token_saved_est`
- `ingest_latency`
- `query_latency`

These measure incremental ingestion performance.

---

### 12.3 Retrieval Diagnostics

Hybrid retrieval diagnostics record:

- BM25-only hits
- vector-only hits
- hybrid hits

This helps evaluate retrieval complementarity.

---

### 12.4 Answer Quality (V2 — Phase 14)

Answer quality evaluation is implemented in Phase 14 using RAGAS metrics:

- faithfulness
- answer relevance
- context precision
- context recall

### 12.5 Evaluation Assets (Required)

To make evaluation reproducible and understandable, maintain two curated query sets:

- `example_queries`:
  - representative "normal" user questions for demo and regression checks
  - expected to produce grounded answers with citations
- `failure_cases`:
  - intentionally difficult or invalid queries (insufficient evidence, ambiguous intent, conflicting evidence, noisy/oversized input)
  - expected behavior is explicit abstain/warning/fallback, not hallucinated answers

Both sets should be versioned and used in CLI + Streamlit evaluation flows.

---

## 13. System Architecture

Streamlit and MCP are **two independent entry points** into the same pipeline. Streamlit calls the pipeline directly — it does not go through MCP.

```
  +---------------------+        +---------------------+
  |      Streamlit UI   |        |     MCP Interface   |
  | (Studio + Debug)    |        |  (AI Agent / Tools) |
  +----------+----------+        +----------+----------+
             |                              |
             +---------------+--------------+
                             |
                  +----------+----------+
                  |   Query / Ingest    |
                  |      Pipeline       |
                  +----------+----------+
                             |
          +------------------+------------------+
          |                                     |
     Retrieval Layer                        Generation
 (BM25 + Vector + RRF + Rerank)           Answer Composer
          |                                     |
          +------------------+------------------+
                             |
                           Chunk
                             |
                  +----------+----------+
                  |     DocStore        |
                  |      SQLite         |
                  +----------+----------+
                             |
                  +----------+----------+
                  |     Vector Index    |
                  |       FAISS         |
                  +----------+----------+
                             |
                      Ingestion Pipeline
                             |
          Loader → Sniffer → Router → Parser
                             |
                          Cleaner
                             |
                         Chunking
                             |
                         Embedding
                             |
                          Indexing
```

---

## 14. Storage Design

SQLite stores:

- documents
- text blocks
- chunks
- pipeline runs
- trace events

Core tables:

```
documents
text_blocks
chunks
runs
trace_events
```

Vector embeddings are stored in FAISS.

---

## 15. Configuration System

The framework uses layered configuration:

```
configs/
  settings.yaml
  profiles/
  routers/
  chunking/
  evaluation/
  connectors/
  tenants/
```

Configuration hierarchy:

```
global settings
 ↓
runtime profiles
 ↓
router rules
 ↓
tenant overrides
```

Key fields in `settings.yaml` include:

```yaml
retrieval:
  bm25_top_k: 20
  vector_top_k: 20
  fusion_pool_size: 40
  rerank_top_k: 8

generation:
  context_top_k: 3
  token_budget: 3000        # max tokens to pack into context window
  abstain_if_empty: true

reranking:
  enabled: true
  provider: voyage

connectors:
  enabled: false
  providers:
    email:
      enabled: false
    slack:
      enabled: false
    notion:
      enabled: false
    google_docs:
      enabled: false
```

---

## 16. Streamlit Studio

The UI includes four panels.

**Ingestion Manager**
- document overview
- ingestion statistics

**Ingestion Traces**
- parser routing
- fallback events

**Query Traces**
- retrieval candidates
- reranking changes
- generation traces

**Evaluation Panel**
- retrieval metrics
- efficiency metrics
- experiment comparisons
- hover tooltips on metric names (plain-language explanations for non-technical users)

---

## 17. Implementation Strategy

### V1 Providers

| Component | Provider |
|-----------|---------|
| Embedding | OpenAI |
| Vector DB | FAISS |
| Keyword Search | BM25 |
| Reranker | Voyage |
| LLM | OpenAI |
| DocStore | SQLite |

### V1 / V2 Scope

| Feature | V1 | V2 |
|---------|----|----|
| PDF ingestion | ✅ | |
| HTML ingestion | ✅ | |
| Scanned PDF / OCR | | ✅ |
| Hybrid Search (BM25 + Vector) | ✅ | |
| RRF Fusion | ✅ | |
| Cross-Encoder Reranking | ✅ | |
| Optional LLM Reranking | | ✅ |
| Incremental Identity Hashing (block hash + chunk_signature) | ✅ | |
| Document Fingerprint Early-Skip | | ✅ |
| Incremental Index Consistency (remove stale chunks on update) | | ✅ |
| External Connectors (Email / Slack / Notion / Google Docs) | | ✅ |
| Connector Incremental Sync (cursor/checkpoint) | | ✅ |
| Semantic Metadata Enrichment | | ✅ |
| Resync Engine | | ✅ |
| RAGAS Answer Quality Evaluation | | ✅ |
| Full Agentic Loop | | Future |

Additional providers can be added later through the plugin interface.

---

## 18. Summary

This project implements a transparent, modular RAG system designed for:

- reliable knowledge retrieval
- evidence-based answers
- incremental document updates
- configurable infrastructure
- full pipeline observability

The architecture supports both local experimentation and future production deployment.

---
# RAG Framework Development Task Breakdown

This task plan is designed to get the **end-to-end main path working first**, then add observability, quality improvements, and optional features.

## Planning Principles

- Each task is scoped to roughly **1 hour**
- Each task includes:
  - files to modify
  - acceptance criteria
  - test method
- Dependencies are explicit
- The order prioritizes:
  1. core contracts
  2. minimal storage
  3. minimal ingestion
  4. minimal retrieval
  5. minimal generation
  6. MCP + Streamlit
  7. traces and evaluation
  8. advanced routing/chunking improvements

---

# Phase 0 — Project Skeleton

## Task 0.1 — Create project folder structure
**Depends on:** none

**Files to create/modify**
- `rag/`
- `rag/core/contracts/`
- `rag/core/interfaces/`
- `rag/core/registry/`
- `rag/core/utils/`
- `rag/pipelines/`
- `rag/infra/`
- `rag/infra/connectors/`
- `rag/app/`
- `rag/cli/`
- `configs/`
- `configs/prompts/`
- `configs/prompts/grounded_answer.txt` (placeholder content, to be filled in Phase 7)
- `configs/prompts/llm_rerank.txt` (placeholder content, V2)
- `configs/connectors/`
- `tests/`
- `README.md`

**Acceptance criteria**
- All top-level folders from the architecture exist
- Empty `__init__.py` files exist where needed
- `configs/prompts/` directory exists with placeholder prompt files
- Project can be opened cleanly in editor without missing import roots

**Test method**
- Run a directory listing and verify expected folders exist
- Open the project in your IDE and confirm Python package imports resolve

---

## Task 0.2 — Add Python project config
**Depends on:** 0.1

**Files to modify**
- `pyproject.toml`
- `requirements.txt`
- `.env.example`

**Acceptance criteria**
- Project dependencies are listed
- Main libraries are declared (streamlit, sqlite support, pymupdf, faiss, etc.)
- `.env.example` includes placeholder keys

**Test method**
- Create a fresh virtual environment
- Install dependencies
- Confirm import of key packages works

---

## Task 0.3 — Set up pytest infrastructure
**Depends on:** 0.2

**Files to modify**
- `tests/conftest.py`
- `pyproject.toml` (add `[tool.pytest.ini_options]` section)

**Acceptance criteria**
- pytest can discover and run test files
- `conftest.py` includes basic shared fixtures (e.g. `tmp_path`, `sample_config`)
- Test markers are defined: `unit`, `integration`, `e2e`
- `pytest --collect-only` reports zero errors

**Test method**
- Add a dummy `tests/test_smoke.py` with one passing test
- Run `pytest -v` and confirm it passes

---

# Phase 1 — Core Contracts and Config

## Task 1.1 — Define core document contracts
**Depends on:** 0.2

**Files to modify**
- `rag/core/contracts/document.py`
- `rag/core/contracts/ir_block.py`
- `rag/core/contracts/parse_report.py`

**Acceptance criteria**
- `Document`, `IRBlock`, and `ParseReport` are defined
- `ParseReport` includes the following fields:
  - `char_count: int`
  - `block_count: int`
  - `non_printable_ratio: float`
  - `repetition_score: float`
  - `parser_used: str`
  - `fallback_triggered: bool`
- Fields match the architecture spec
- Data models are importable without circular dependencies

**Test method**
- Create a tiny unit test in `tests/test_contracts_document.py`
- Instantiate each object with sample values

---

## Task 1.2 — Define chunking and retrieval contracts
**Depends on:** 1.1

**Files to modify**
- `rag/core/contracts/text_block.py`
- `rag/core/contracts/chunk.py`
- `rag/core/contracts/candidate.py`

**Acceptance criteria**
- `TextBlock`, `Chunk`, and `Candidate` are defined
- Candidate includes score fields and source attribution
- Chunk includes both `stable_text` and `display_text`

**Test method**
- Add `tests/test_contracts_chunk.py`
- Instantiate objects and validate required fields

---

## Task 1.3 — Define citation and answer contracts
**Depends on:** 1.2

**Files to modify**
- `rag/core/contracts/citation.py`
- `rag/core/contracts/answer.py`
- `rag/core/contracts/trace.py`

**Acceptance criteria**
- `Span`, `Citation`, `Answer`, and `AnswerTrace` are defined
- Span supports multiple `span_type` values
- AnswerTrace includes tokens and latency fields

**Test method**
- Add `tests/test_contracts_answer.py`
- Instantiate answer + citation objects for PDF and chunk-only cases

---

## Task 1.4a — Define ingestion-side interface contracts
**Depends on:** 1.3

**Files to modify**
- `rag/core/interfaces/parser.py`
- `rag/core/interfaces/cleaner.py`
- `rag/core/interfaces/block_splitter.py`
- `rag/core/interfaces/chunk_packer.py`

**Acceptance criteria**
- All ingestion interfaces exist with method signatures matching the architecture spec
- No pipeline code depends on concrete implementations

**Test method**
- Import all interfaces in a single test file
- Confirm no import errors

---

## Task 1.4b — Define retrieval-side interface contracts
**Depends on:** 1.3

**Files to modify**
- `rag/core/interfaces/embedding.py`
- `rag/core/interfaces/vector_index.py`
- `rag/core/interfaces/keyword_index.py`
- `rag/core/interfaces/reranker.py`

**Acceptance criteria**
- All retrieval interfaces exist with method signatures matching the architecture spec
- No pipeline code depends on concrete implementations

**Test method**
- Import all interfaces in a single test file
- Confirm no import errors

---

## Task 1.4c — Define storage and generation interface contracts
**Depends on:** 1.3

**Files to modify**
- `rag/core/interfaces/doc_store.py`
- `rag/core/interfaces/trace_store.py`
- `rag/core/interfaces/llm_client.py`
- `rag/core/interfaces/context_packer.py`
- `rag/core/interfaces/answer_composer.py`

**Acceptance criteria**
- All storage and generation interfaces exist with method signatures matching the architecture spec
- No pipeline code depends on concrete implementations

**Test method**
- Import all interfaces in a single test file
- Confirm no import errors

---

## Task 1.5 — Add base settings config
**Depends on:** 1.4c

**Files to modify**
- `configs/settings.yaml`
- `configs/profiles/local_fast.yaml`
- `configs/profiles/local_quality.yaml`

**Acceptance criteria**
- Global retrieval/rerank/generation/evaluation defaults exist
- `local_fast` and `local_quality` override the expected values

**Test method**
- Manually inspect YAML
- Load files with a small Python script and verify merged values

---

## Task 1.6 — Build config loader
**Depends on:** 1.5

**Files to modify**
- `rag/core/registry/config_loader.py`

**Acceptance criteria**
- Can load `settings.yaml`
- Can apply a profile override
- Returns one merged config object/dict

**Test method**
- Add `tests/test_config_loader.py`
- Load `local_fast` and assert top-k values match expectations

---

# Phase 2 — SQLite Foundation

## Task 2.1 — Implement SQLite DocStore schema creation
**Depends on:** 1.6

**Files to modify**
- `rag/infra/stores/docstore_sqlite.py`

**Acceptance criteria**
- Creates tables:
  - `documents`
  - `text_blocks`
  - `chunks`
- Adds required indexes
- Initializes DB on first run

**Test method**
- Create a temporary SQLite file
- Run schema init
- Inspect tables using sqlite CLI or Python query

---

## Task 2.2 — Implement SQLite TraceStore schema creation
**Depends on:** 1.6

**Files to modify**
- `rag/infra/stores/tracestore_sqlite.py`

**Acceptance criteria**
- Creates tables:
  - `runs`
  - `trace_events`
- Adds indexes on `run_id` and `event_type`

**Test method**
- Create a temporary SQLite file
- Run schema init
- Query sqlite master to confirm tables/indexes

---

## Task 2.3 — Implement basic DocStore write/read methods
**Depends on:** 2.1, 1.3

**Files to modify**
- `rag/infra/stores/docstore_sqlite.py`

**Acceptance criteria**
- Can insert one document
- Can insert blocks
- Can insert chunks
- Can fetch chunk by `chunk_id`
- Can fetch all chunks for a given `doc_id` (required by block diff in Phase 11)

**Test method**
- Add `tests/test_docstore_sqlite.py`
- Insert sample records and query them back

---

## Task 2.4 — Implement basic TraceStore write/read methods
**Depends on:** 2.2, 1.3

**Files to modify**
- `rag/infra/stores/tracestore_sqlite.py`

**Acceptance criteria**
- Can create a run record
- Can append trace events
- Can fetch trace events by `run_id`

**Test method**
- Add `tests/test_tracestore_sqlite.py`
- Insert a run and two events, then verify retrieval order

---

# Phase 3 — Minimal Ingestion Main Path

## Task 3.1 — Implement local file loader
**Depends on:** 1.4a

**Files to modify**
- `rag/infra/loading/local_file_loader.py`

**Acceptance criteria**
- Loads a local file into a raw artifact object
- Returns source path and basic metadata

**Test method**
- Load a sample `.txt` file
- Verify bytes/text and metadata are returned

---

## Task 3.2 — Implement minimal sniffer
**Depends on:** 3.1

**Files to modify**
- `rag/infra/sniffing/sniffer_magic.py`
- `rag/infra/sniffing/sniffer_mime.py`
- `rag/infra/sniffing/composite_sniffer.py`

**Acceptance criteria**
- Detects at least:
  - PDF
  - HTML
  - TXT/Markdown
- Optionally detects DOCX/PPTX/images as unsupported types until corresponding parsers are implemented
- Returns a stable detected type

**Test method**
- Add `tests/test_sniffer.py`
- Run sniffer on supported sample files and one unsupported file, and verify unsupported output is explicit

---

## Task 3.3 — Implement minimal parser router
**Depends on:** 3.2, 1.4a

**Files to modify**
- `configs/routers/parser_candidates.yaml`
- `rag/pipelines/parsing/plans.py`
- `rag/pipelines/parsing/orchestrator.py`

**Acceptance criteria**
- Router can map detected type to a parser candidate list
- Router attempts parser in order
- Router returns first successful parse result

**Test method**
- Mock two parsers: first fails, second succeeds
- Verify fallback happens

---

## Task 3.4 — Implement TXT/Markdown parser
**Depends on:** 3.3

**Files to modify**
- `rag/infra/parsing/md_parser.py`

**Acceptance criteria**
- Converts plain text/markdown file into `Document + IRBlocks + ParseReport`
- Produces paragraph/title blocks

**Test method**
- Parse a small markdown file
- Verify blocks and parse report fields

---

## Task 3.5 — Implement HTML parser
**Depends on:** 3.3

**Files to modify**
- `rag/infra/parsing/html_trafilatura.py`

**Acceptance criteria**
- Extracts main text from HTML
- Produces `Document` with `IRBlocks`

**Test method**
- Parse a saved HTML sample
- Verify extracted text excludes obvious boilerplate

---

## Task 3.6 — Implement PDF text parser
**Depends on:** 3.3

**Files to modify**
- `rag/infra/parsing/pdf_pymupdf.py`

**Acceptance criteria**
- Extracts text from digital PDFs
- Produces blocks with page numbers
- Produces parse report

**Test method**
- Parse a sample digital PDF
- Verify page numbers and non-empty text

---

## Task 3.7 — Implement parse quality gates
**Depends on:** 3.4, 3.5, 3.6

**Files to modify**
- `configs/routers/quality_gates.yaml`
- `rag/pipelines/parsing/quality_gates.py`

**Acceptance criteria**
- Computes basic pass/fail on:
  - char_count
  - non_printable_ratio
  - repetition_score
- Router can reject bad parse output

**Test method**
- Feed a fake parse report with low char_count
- Verify gate fails

---

## Task 3.8a — Implement unicode_fix and empty_filter cleaners
**Depends on:** 3.4, 3.5, 3.6

**Files to modify**
- `rag/infra/cleaning/unicode_fix.py`
- `rag/infra/cleaning/empty_filter.py`

**Acceptance criteria**
- Unicode normalization handles common encoding issues (smart quotes, ligatures, etc.)
- Empty paragraphs are removed
- Both cleaners implement the cleaner interface

**Test method**
- Add `tests/test_cleaner_unicode.py` and `tests/test_cleaner_empty.py`
- Feed text with known encoding issues and verify output
- Feed text with empty paragraphs and verify they are removed

---

## Task 3.8b — Implement dedupe and PDF header/footer cleaners
**Depends on:** 3.4, 3.5, 3.6

**Files to modify**
- `rag/infra/cleaning/dedupe_paragraphs.py`
- `rag/infra/cleaning/pdf_header_footer_dedupe.py`

**Acceptance criteria**
- Duplicate paragraphs are detected and removed
- Repeating PDF headers/footers across pages are identified and removed

**Test method**
- Add `tests/test_cleaner_dedupe.py` and `tests/test_cleaner_pdf_headers.py`
- Feed text with duplicated paragraphs and verify deduplication
- Feed multi-page PDF output with repeated headers and verify removal

---

## Task 3.8c — Implement HTML nav/footer and OCR line merge cleaners
**Depends on:** 3.4, 3.5, 3.6

**Files to modify**
- `rag/infra/cleaning/html_nav_footer_remove.py`
- `rag/infra/cleaning/ocr_line_merge.py`

**Acceptance criteria**
- HTML navigation and footer elements are removed
- OCR broken lines are merged into proper paragraphs (when OCR is enabled)

**Test method**
- Add `tests/test_cleaner_html.py` and `tests/test_cleaner_ocr.py`
- Feed HTML with nav/footer elements and verify removal
- Feed OCR-style broken lines and verify merge

---

## Task 3.8d — Wire cleaner pipeline (sequential execution of all 6 steps)
**Depends on:** 3.8a, 3.8b, 3.8c

**Files to modify**
- `rag/infra/cleaning/cleaner_pipeline.py`
- `configs/routers/cleaner_router.yaml`

**Acceptance criteria**
- Cleaner pipeline runs all 6 steps in sequence
- Each step can be enabled/disabled via config
- Pipeline accepts parsed text and returns normalized output

**Test method**
- Run one small document through the full pipeline
- Verify output is clean and all steps were executed

---

## Task 3.9 — Implement paragraph block splitter
**Depends on:** 3.8d

**Files to modify**
- `rag/infra/chunking/block_splitter_paragraph.py`

**Acceptance criteria**
- Converts cleaned text into `TextBlock[]`
- Computes `block_hash`
- Preserves order

**Test method**
- Split a 5-paragraph text
- Verify 5 blocks and stable hashes

---

## Task 3.10 — Implement anchor annotator
**Depends on:** 3.9

**Files to modify**
- `configs/chunking/anchors.yaml`
- `rag/infra/chunking/anchor_annotator_rules.py`

**Acceptance criteria**
- Detects heading/list/section anchors from block text
- Adds anchor fields to TextBlocks

**Test method**
- Feed blocks with headings and numbered list items
- Verify anchor type and level

---

## Task 3.11 — Implement chunk packer
**Depends on:** 3.10

**Files to modify**
- `rag/infra/chunking/chunk_packer_anchor_aware.py`

**Acceptance criteria**
- Packs blocks into chunks
- Computes `chunk_signature`
- Produces `stable_text`
- Preserves `loc_span`

**Test method**
- Pack a small block list
- Verify chunk count, chunk_signature, block_hash order

---

## Task 3.12 — Implement minimal ingest pipeline
**Depends on:** 2.3, 2.4, 3.3–3.11

**Files to modify**
- `rag/pipelines/ingest_pipeline.py`

**Acceptance criteria**
- End-to-end ingest works for one TXT/MD/HTML/PDF file
- Stores document, text blocks, chunks
- Writes a run and trace events

**Test method**
- Run ingestion on one sample file
- Verify database has rows in all expected tables

---

## Task 3.13 — Implement CLI ingest entry point
**Depends on:** 3.12

**Files to modify**
- `rag/cli/ingest.py`

**Acceptance criteria**
- `python -m rag.cli.ingest --path sample.pdf` runs the ingest pipeline end-to-end
- Supports `--collection` flag for specifying target collection
- Prints summary on success (document name, chunk count, time elapsed)
- Returns non-zero exit code on failure

**Test method**
- Run CLI on a sample TXT file
- Verify output summary and database rows

---

# Phase 4 — Embedding and Indexes

## Task 4.1 — Implement base embedding provider + factory
**Depends on:** 1.4b

**Files to modify**
- `rag/infra/embedding/base_embedding.py`
- `rag/infra/embedding/embedding_factory.py`

**Acceptance criteria**
- Factory can return a provider instance from config
- Base provider exposes required methods

**Test method**
- Create a fake embedding provider and instantiate via factory

---

## Task 4.2 — Implement OpenAI embedding provider
**Depends on:** 4.1

**Files to modify**
- `rag/infra/embedding/openai_embedding.py`

**Acceptance criteria**
- Can embed text batches
- Returns embeddings and token usage
- Supports tenant-configured model

**Test method**
- Run on 2 short strings
- Verify output count and vector dimensions

---

## Task 4.3 — Implement embedding batcher
**Depends on:** 4.2

**Files to modify**
- `rag/core/utils/batching.py`
- `rag/infra/embedding/openai_embedding.py`

**Acceptance criteria**
- Large text list can be split into batches
- Batch usage is aggregated

**Test method**
- Simulate 100 short chunks
- Verify multiple batch calls occur and counts match

---

## Task 4.4 — Implement BM25 index
**Depends on:** 1.4b

**Files to modify**
- `rag/infra/indexes/bm25_local.py`

**Acceptance criteria**
- Can build index from chunks using `stable_text` (not `display_text`) as the indexed field
- Can search query and return `Candidate` objects
- Can save index to disk (`bm25.pkl`)
- Can load index from disk on startup
- Supports removing a chunk by `chunk_id` (for incremental update)

**Test method**
- Build on 5 chunks
- Search one keyword and verify expected chunk ranks high
- Save, reload, and verify search still works
- Remove one chunk and verify it no longer appears in results

---

## Task 4.5 — Implement FAISS vector index
**Depends on:** 4.2, 1.4

**Files to modify**
- `rag/infra/indexes/faiss_local.py`

**Acceptance criteria**
- Can insert chunk embeddings with associated `chunk_id`
- Can search query embedding and return `chunk_id` + scores
- Can save index to disk (`faiss.index`) along with a `chunk_id` mapping file
- Can load index and mapping from disk on startup
- Supports removing a vector by `chunk_id` (rebuild index from remaining vectors)

**Note:** FAISS does not natively support deletion. Implement deletion by maintaining a `chunk_id → vector` mapping in memory, filtering out removed IDs, and rebuilding the index. This is acceptable for V1 given expected corpus size.

**Test method**
- Insert 3 vectors manually with chunk IDs
- Query nearest one and verify correct chunk ID is returned
- Save, reload, and verify query still works
- Remove one chunk ID and verify it no longer appears in results

---

## Task 4.6 — Extend ingest pipeline to embed and index
**Depends on:** 4.3, 4.4, 4.5, 3.12

**Files to modify**
- `rag/pipelines/ingest_pipeline.py`

**Acceptance criteria**
- New chunks are embedded
- BM25 and FAISS are updated
- Token usage is written to run stats/trace

**Test method**
- Run ingestion on a sample file
- Verify FAISS index exists and BM25 search works

---

## Task 4.7 — Implement index startup loader
**Depends on:** 4.4, 4.5

**Files to modify**
- `rag/core/registry/workspace.py` (or a new `rag/infra/indexes/index_manager.py`)

**Acceptance criteria**
- On pipeline initialization, check if `faiss.index` and `bm25.pkl` exist on disk
- If they exist: load both into memory before serving any query
- If they do not exist: initialize empty indexes (first-run case)
- This logic must run before `query_pipeline.py` is callable

**Note:** Without this task, every restart loses all indexed data. The bug only surfaces after a restart — ingest tests will pass, but query will return empty results or raise an error on a fresh process.

**Test method**
- Ingest a file, stop the process, restart, then run a query
- Verify results are returned without re-ingesting

---

# Phase 5 — Retrieval Main Path

## Task 5.1 — Implement retrieval source attribution
**Depends on:** 4.4, 4.5, 1.2

**Files to modify**
- `rag/core/contracts/candidate.py`
- `rag/pipelines/query_pipeline.py`

**Acceptance criteria**
- Each candidate is marked:
  - `bm25_only`
  - `vector_only`
  - `both`

**Test method**
- Search a query with overlapping and non-overlapping results
- Verify attribution labels

---

## Task 5.2 — Implement RRF fusion
**Depends on:** 5.1

**Files to modify**
- `rag/infra/indexes/rrf_fusion.py`
- `rag/core/interfaces/fusion.py` (if not already created)

**Acceptance criteria**
- BM25 and vector candidate lists can be fused
- Produces ranked candidates with fusion score

**Test method**
- Feed two small ranked lists
- Verify fused order is deterministic

---

## Task 5.3 — Implement query pipeline without rerank
**Depends on:** 4.6, 4.7, 5.2, 2.3, 2.4

**Files to modify**
- `rag/pipelines/query_pipeline.py`

**Acceptance criteria**
- Query returns fused top chunks with citations
- Query run and trace events are recorded

**Test method**
- Ingest one file, then query it
- Verify non-empty result list with citations

---

## Task 5.4 — Implement CLI query entry point
**Depends on:** 5.3

**Files to modify**
- `rag/cli/query.py`

**Acceptance criteria**
- `python -m rag.cli.query "test question"` runs the query pipeline end-to-end
- Supports `--top-k` flag
- Supports `--verbose` flag for showing retrieval details (scores, source attribution)
- Prints formatted results with citations

**Test method**
- Ingest a sample file, then run CLI query
- Verify results are printed with citations

---

# Phase 6 — Reranking

## Task 6.1 — Implement reranker interface and factory
**Depends on:** 1.4b

**Files to modify**
- `rag/infra/rerank/noop.py`
- `rag/core/interfaces/reranker.py`
- `rag/core/registry/plugin_registry.py` (or rerank factory)

**Acceptance criteria**
- Query pipeline can call reranker via interface
- No-op reranker works as fallback

**Test method**
- Run query pipeline with noop reranker
- Verify result order unchanged

---

## Task 6.2 — Implement API cross-encoder reranker
**Depends on:** 6.1

**Files to modify**
- `rag/infra/rerank/voyage_rerank.py`

**Acceptance criteria**
- Can rerank candidate chunks
- Returns rerank scores
- Supports top-8 output

**Test method**
- Mock or call provider with a small candidate set
- Verify reordered results and score field

---

## Task 6.3 — Add rerank stage to query pipeline
**Depends on:** 6.2, 5.3

**Files to modify**
- `rag/pipelines/query_pipeline.py`

**Acceptance criteria**
- Fused results are reranked
- Query trace records pre/post rerank order
- Final output uses reranked top K

**Test method**
- Query a sample corpus
- Verify trace contains both fused and reranked results

---

## Task 6.4 — Add optional LLM rerank stage (V2)
**Depends on:** 6.3, 7.2

**Files to modify**
- `rag/pipelines/query_pipeline.py`
- `configs/settings.yaml`

**Acceptance criteria**
- LLM rerank can be toggled by config (default off)
- When enabled, it reranks cross-encoder top-K candidates only
- Query trace records whether LLM rerank was enabled and pre/post order

**Test method**
- Run one query with LLM rerank off and one with it on
- Verify outputs differ only when enabled, and trace fields are present

---

# Phase 7 — LLM Abstraction and Generation

## Task 7.1 — Implement LLM client interface
**Depends on:** 1.4c

**Files to modify**
- `rag/core/interfaces/llm_client.py`

**Acceptance criteria**
- Interface supports:
  - `generate`
  - optional structured output
  - token/latency reporting

**Test method**
- Import interface and verify no pipeline imports concrete SDKs

---

## Task 7.2 — Implement OpenAI LLM client
**Depends on:** 7.1

**Files to modify**
- `rag/infra/llm/openai_llm_client.py`

**Acceptance criteria**
- Can generate grounded answer text
- Returns token usage and latency

**Test method**
- Run one short prompt
- Verify text and usage fields

---

## Task 7.3 — Implement light context packer
**Depends on:** 6.3, 7.1

**Files to modify**
- `rag/infra/generation/context_packer_light.py`

**Acceptance criteria**
- Packs top 3 chunks
- Deduplicates repeated chunk text
- Produces chunk→citation mapping
- Respects token budget

**Test method**
- Feed 5 chunks with some duplicates
- Verify output contains 3 useful chunks and mapping

---

## Task 7.4 — Implement grounded prompt builder
**Depends on:** 7.3

**Files to modify**
- `rag/infra/generation/prompt_builder_grounded.py`

**Acceptance criteria**
- Prompt clearly instructs grounded answering
- Prompt includes selected context and citation mapping
- Prompt includes abstain behavior for insufficient evidence

**Test method**
- Generate one prompt from sample chunks
- Verify instructions and context layout

---

## Task 7.5 — Implement answer composer
**Depends on:** 7.2, 7.4, 1.3

**Files to modify**
- `rag/infra/generation/answer_composer_basic.py`

**Acceptance criteria**
- Produces `Answer`
- Produces `AnswerTrace`
- Includes citations in final answer output

**Test method**
- Use mocked chunks and a real or mocked LLM response
- Verify answer object fields

---

## Task 7.6 — Add generation stage to query pipeline
**Depends on:** 7.5, 6.3

**Files to modify**
- `rag/pipelines/query_pipeline.py`

**Acceptance criteria**
- Query pipeline returns:
  - answer
  - citations
  - answer trace
- Query trace stores generation stats

**Test method**
- Run end-to-end query
- Verify final answer and citation list exist

---

# Phase 8 — Streamlit Main Path

## Task 8.1 — Create Streamlit app shell
**Depends on:** 0.1

**Files to modify**
- `rag/app/studio/studio.py`
- `rag/app/studio/pages/1_ingestion_manager.py`
- `rag/app/studio/pages/2_ingestion_traces.py`
- `rag/app/studio/pages/3_query_traces.py`
- `rag/app/studio/pages/4_evaluation_panel.py`

**Acceptance criteria**
- Streamlit launches
- Four pages appear

**Test method**
- Run `streamlit run ...`
- Verify pages load without crashing

---

## Task 8.2 — Build ingestion manager page
**Depends on:** 3.12, 4.6, 8.1

**Files to modify**
- `rag/app/studio/pages/1_ingestion_manager.py`

**Acceptance criteria**
- User can choose a file and run ingestion
- Ingest summary is displayed
- Ingest summary includes embedding/index configuration used for this run:
  - `embedding_provider`
  - `embedding_model`
  - `vector_dimension`
  - vector index type (for example FAISS index type)
- If vector dimension mismatches index expectation, UI shows a clear warning/error state

**Test method**
- Upload a file
- Verify successful ingest summary on screen
- Verify summary shows provider/model/dimension/index fields
- Simulate a dimension mismatch and verify warning/error is displayed

---

## Task 8.3 — Build ingestion trace page
**Depends on:** 2.4, 8.1

**Files to modify**
- `rag/app/studio/pages/2_ingestion_traces.py`
- `rag/app/studio/components/trace_viewer.py`

**Acceptance criteria**
- User can inspect parse/router/chunking events by run_id

**Test method**
- Run one ingestion
- Open trace page and verify stages appear

---

## Task 8.4 — Build query trace page
**Depends on:** 7.6, 8.1

**Files to modify**
- `rag/app/studio/pages/3_query_traces.py`
- `rag/app/studio/components/candidate_table.py`

**Acceptance criteria**
- Page shows:
  - retrieved chunks
  - scores
  - source attribution
  - rerank changes
  - context packing details:
    - selected chunks (with chunk IDs)
    - dropped chunks (if any) and drop reason (for example token budget limit or dedup)
    - packing summary (`context_top_k`, `token_budget`, packed token count)
  - final answer
  - citations
  - generation usage stats:
    - prompt_tokens
    - completion_tokens
    - total_tokens
    - generation latency
- Query input must have an explicit submit control:
  - `Run Query` button (primary)
  - keyboard submit shortcut (Enter for single-line input, or documented shortcut for multiline)
- Query execution is triggered only on submit (not on every keystroke)
- Empty query is rejected with a clear validation message
- While query is running, submit control is disabled and a loading indicator is shown

**Test method**
- Run a query
- Verify all sections render
- Verify context packing section shows selected/dropped chunks and token-budget-related summary
- Verify token usage and generation latency fields are displayed for the run
- Type text without submit and verify query is not executed
- Click `Run Query` and verify query executes once
- Use keyboard submit and verify behavior matches button submit

---

# Phase 9 — MCP Main Path

## Task 9.1 — Add MCP tool schemas
**Depends on:** 1.3

**Files to modify**
- `rag/app/mcp_server/schemas.py`

**Acceptance criteria**
- Schemas exist for:
  - `rag.ingest`
  - `rag.query`
  - `rag.eval.run`

**Test method**
- Validate a sample payload against each schema

---

## Task 9.2 — Implement MCP server wiring
**Depends on:** 9.1, 3.12, 7.6

**Files to modify**
- `rag/app/mcp_server/server.py`
- `rag/app/mcp_server/wiring.py`

**Acceptance criteria**
- MCP server can invoke ingest/query/eval
- Query returns answer + citations

**Test method**
- Run a local MCP call
- Verify structured response

---

## Task 9.3 — End-to-end main path validation
**Depends on:** 9.2, 8.4

**Files to modify**
- `tests/e2e/test_main_path.py`

**Acceptance criteria**
- Ingest a PDF/HTML/TXT file via CLI — succeeds
- Query via CLI — returns results with citations
- Streamlit four pages load without errors
- MCP tool call returns structured response with answer + citations
- All unit tests pass (`pytest -q`)

**Test method**
- Run the full sequence manually: ingest → query → open Streamlit → MCP call
- Run `pytest -q` for full regression

---

# Phase 10 — Evaluation

## Task 10.1 — Implement retrieval metrics
**Depends on:** 5.3

**Files to modify**
- `rag/pipelines/scoring/metrics.py`

**Acceptance criteria**
- Computes:
  - Recall@K
  - MRR
  - nDCG

**Test method**
- Add `tests/test_metrics.py`
- Run metrics on a tiny synthetic eval set

---

## Task 10.2 — Implement source attribution diagnostics
**Depends on:** 5.1, 10.1

**Files to modify**
- `rag/pipelines/eval_pipeline.py`
- `rag/core/contracts/eval_report.py`

**Acceptance criteria**
- Eval report includes ratios of:
  - bm25_only
  - vector_only
  - both

**Test method**
- Run eval on a small dataset
- Verify ratios are present in report

---

## Task 10.3 — Implement system efficiency metrics
**Depends on:** 4.6, 2.4

**Files to modify**
- `rag/pipelines/eval_pipeline.py`

**Acceptance criteria**
- Eval/run summary reports:
  - token_saved_est
  - ingest_latency
  - query_latency
- After Task 11.2 is completed, summary additionally reports:
  - skipped_chunks
  - changed_chunks

**Note:** `changed_chunks` and `skipped_chunks` must come from block diff results in Task 11.2.  
Before Task 11.2 is complete, these two fields should be emitted as `null` (or omitted) rather than inferred.

**Test method**
- Run an ingest twice with one small change
- Verify efficiency metrics are reported

---

## Task 10.4 — Implement CLI eval entry point
**Depends on:** 10.1

**Files to modify**
- `rag/cli/eval.py`

**Acceptance criteria**
- `python -m rag.cli.eval --suite example_queries` runs evaluation on the example query suite
- `python -m rag.cli.eval --suite failure_cases` runs evaluation on the failure case suite
- `python -m rag.cli.eval --answer-quality` runs RAGAS evaluation (when Phase 14 is complete; before that, prints "RAGAS not available")
- Prints per-case results and aggregate metrics to stdout
- Returns non-zero exit code if any unexpected failures occur

**Test method**
- Run CLI eval with a minimal fixture
- Verify output format and exit code

---

## Task 10.5 — Build evaluation query suites (example_queries + failure_cases)
**Depends on:** 10.1, 10.2, 10.4

**Files to modify**
- `tests/fixtures/example_queries.json`
- `tests/fixtures/failure_cases.json`
- `rag/pipelines/eval_pipeline.py`

**Acceptance criteria**
- `example_queries.json` contains at least 10 representative user questions with expected grounded behavior
- `failure_cases.json` contains at least 8 challenging cases, covering:
  - insufficient evidence
  - ambiguous query
  - conflicting evidence
  - noisy or oversized input
- Each case defines:
  - `query`
  - `expected_behavior` (`answer` / `abstain` / `warn`)
  - optional `expected_sources`
- Eval pipeline can run both suites and emit per-case outcomes

**Test method**
- Run `python -m rag.cli.eval --suite example_queries`
- Run `python -m rag.cli.eval --suite failure_cases`
- Verify per-case pass/fail and expected vs actual behavior are printed/stored

---

## Task 10.6 — Build evaluation page
**Depends on:** 10.1, 10.2, 10.3, 10.5, 8.1

**Files to modify**
- `rag/app/studio/pages/4_evaluation_panel.py`
- `rag/app/studio/components/metrics_table.py`
- `rag/app/studio/components/metric_glossary.py`

**Acceptance criteria**
- User can run evaluation and see:
  - retrieval metrics
  - source attribution
  - efficiency metrics
- Page supports running:
  - `example_queries` suite
  - `failure_cases` suite
- Result view clearly labels each case as:
  - pass/fail
  - expected behavior
  - actual behavior
- Hovering over each metric name shows a tooltip explaining:
  - what the metric means
  - whether higher/lower is better
  - common interpretation pitfalls (short plain language)
- Tooltip text comes from one centralized glossary mapping to avoid page/component drift
- If block-diff metrics are unavailable (before Task 11.2), UI shows `N/A` for `skipped_chunks` and `changed_chunks`

**Test method**
- Run evaluation on a small dataset
- Verify metrics display
- Run both suites (`example_queries` and `failure_cases`) and verify per-case result rows are shown
- Hover over at least 5 metric names and verify tooltips render with correct definitions

---

# Phase 11 — Incremental Ingestion Refinement

## Task 11.1 — Implement document fingerprint tracking
**Depends on:** 2.3, 3.12

**Files to modify**
- `rag/pipelines/ingest_pipeline.py`
- `rag/core/utils/hashing.py`

**Acceptance criteria**
- Pipeline can detect unchanged documents before deep processing

**Test method**
- Run ingest twice on same file
- Verify second run is skipped early

---

## Task 11.2 — Implement block diff logic
**Depends on:** 3.9, 2.3

**Files to modify**
- `rag/pipelines/ingest_pipeline.py`

**Acceptance criteria**
- Pipeline can classify blocks as:
  - unchanged
  - added
  - removed

**Test method**
- Modify one paragraph in sample document
- Verify diff identifies only affected blocks

---

## Task 11.3 — Implement resync engine
**Depends on:** 3.11, 11.2

**Files to modify**
- `rag/infra/chunking/resync_window.py`
- `rag/infra/chunking/resync_hybrid.py`

**Acceptance criteria**
- Inserting one new paragraph does not force all downstream chunks to re-embed
- Resync stats are traceable

**Test method**
- Create old/new block sequences with one insertion
- Verify only nearby chunks are marked changed

---

## Task 11.4 — Add threshold guardrails
**Depends on:** 11.3

**Files to modify**
- `configs/chunking/resync.yaml`
- `rag/pipelines/ingest_pipeline.py`

**Acceptance criteria**
- Pipeline warns or records when changed ratio exceeds threshold

**Test method**
- Simulate large rewrite
- Verify threshold event is emitted

---

## Task 11.5 — Remove stale chunks from indexes after incremental update
**Depends on:** 11.2, 4.4, 4.5

**Files to modify**
- `rag/pipelines/ingest_pipeline.py`
- `rag/infra/indexes/bm25_local.py`
- `rag/infra/indexes/faiss_local.py`

**Acceptance criteria**
- After block diff, chunks marked `removed` or `changed` are deleted from both BM25 and FAISS indexes
- New/changed chunks are re-embedded and re-inserted
- DocStore chunk records are updated to reflect the new version
- Index is saved to disk after update

**Note:** This task closes the incremental update loop. Without it, old chunk versions remain in the indexes and will appear in retrieval results even after the source document is updated.

**Test method**
- Ingest a document, then modify one section and re-ingest
- Verify the old chunk text no longer appears in search results
- Verify the updated chunk text is retrievable

---

# Phase 12 — Nice-to-Have but Still Useful

## Task 12.1 — Add pronoun risk detection
**Depends on:** 3.11

**Files to modify**
- `rag/infra/chunking/pronoun_risk_rules.py`

**Acceptance criteria**
- Chunks can get a pronoun risk score

**Test method**
- Feed chunks with “it/this/they”
- Verify risk score is non-zero

---

## Task 12.2 — Add metadata enrichment (rules first)
**Depends on:** 3.11

**Files to modify**
- `rag/infra/chunking/metadata_enricher_rules.py`

**Acceptance criteria**
- Chunks receive title/summary/tags using rules fallback

**Test method**
- Enrich a few chunks and verify metadata fields exist

---

## Task 12.3 — Add LLM metadata enrichment
**Depends on:** 7.2, 12.2

**Files to modify**
- `rag/infra/chunking/metadata_enricher_llm_batch.py`

**Acceptance criteria**
- Chunks can be batch-enriched using LLM
- Usage appears in trace

**Test method**
- Run enrichment on 3 chunks
- Verify metadata plus token counts

---

# Phase 13 — OCR Support (Scanned PDF)

## Task 13.1 — Implement OCR provider interface and PaddleOCR provider
**Depends on:** 1.4a

**Files to modify**
- `rag/core/interfaces/ocr_provider.py`
- `rag/infra/ocr/paddleocr_provider.py`

**Acceptance criteria**
- OCR provider interface defines `ocr(image) -> list[TextBlock]`
- PaddleOCR implementation can extract text from a page image
- Returns text blocks with bounding box and confidence score
- Graceful error when PaddleOCR is not installed (clear ImportError message)

**Test method**
- Add `tests/test_paddleocr_provider.py`
- Run OCR on a sample image with known text
- Verify extracted text matches expected content

---

## Task 13.2 — Implement page renderer (PDF page → image)
**Depends on:** 13.1

**Files to modify**
- `rag/core/interfaces/page_renderer.py`
- `rag/infra/ocr/renderer_pymupdf.py`

**Acceptance criteria**
- Page renderer interface defines `render(pdf_path, page_num) -> Image`
- PyMuPDF implementation renders a PDF page to a PIL Image at configurable DPI
- Supports rendering a range of pages

**Test method**
- Add `tests/test_page_renderer.py`
- Render page 1 of a sample PDF
- Verify output is a valid image with expected dimensions

---

## Task 13.3 — Implement scanned PDF parser (renderer + OCR)
**Depends on:** 13.1, 13.2, 3.3

**Files to modify**
- `rag/infra/parsing/pdf_ocr_parser.py`

**Acceptance criteria**
- Parser takes a PDF path, renders each page to image, runs OCR, produces `Document + IRBlocks + ParseReport`
- ParseReport includes `parser_used: "pdf_ocr"` and per-page confidence scores
- Blocks include page number and bounding box metadata

**Test method**
- Add `tests/test_pdf_ocr_parser.py`
- Parse a sample scanned PDF (image-only, no selectable text)
- Verify blocks contain expected text content

---

## Task 13.4 — Integrate OCR parser into parser router
**Depends on:** 13.3, 3.7

**Files to modify**
- `configs/routers/parser_candidates.yaml`
- `rag/pipelines/parsing/orchestrator.py`

**Acceptance criteria**
- Parser router detects scanned PDFs (text extraction yields very low char count)
- Router falls back to OCR parser when text parser quality gate fails
- Parse report records fallback event
- OCR can be disabled via config (`ocr.enabled: false`)

**Test method**
- Feed a scanned PDF through the router
- Verify OCR parser is selected after text parser fails quality gate
- Feed a normal text PDF and verify OCR is not triggered

---

# Phase 14 — RAGAS Answer Quality Evaluation

## Task 14.1 — Implement RAGAS evaluator interface
**Depends on:** 10.1, 7.6

**Files to modify**
- `rag/core/interfaces/answer_evaluator.py`
- `rag/infra/evaluation/ragas_evaluator.py`

**Acceptance criteria**
- Answer evaluator interface defines `evaluate(query, answer, contexts, ground_truth) -> dict`
- RAGAS implementation computes:
  - `faithfulness` — is the answer supported by retrieved context?
  - `answer_relevancy` — does the answer address the question?
  - `context_precision` — are the retrieved contexts relevant?
- Graceful degradation: if `ragas` package is not installed, raises clear ImportError with install instructions
- Returns metrics as a dictionary

**Test method**
- Add `tests/test_ragas_evaluator.py`
- Run evaluator with mocked LLM on a sample query/answer/context triplet
- Verify all three metric fields are present and within [0, 1]

---

## Task 14.2 — Create golden test set for answer evaluation
**Depends on:** 14.1

**Files to modify**
- `tests/fixtures/golden_answer_set.json`
- `rag/pipelines/eval_pipeline.py`

**Acceptance criteria**
- Golden test set contains at least 5 entries, each with:
  - `query`
  - `expected_answer` (reference answer)
  - `expected_sources` (expected source documents)
- Eval pipeline can load the golden set and run RAGAS metrics against it
- Results are written to trace store

**Test method**
- Run `python -m rag.cli.eval --answer-quality`
- Verify output includes faithfulness, answer_relevancy, context_precision per query

---

## Task 14.3 — Add answer quality metrics to evaluation panel
**Depends on:** 14.2, 10.6

**Files to modify**
- `rag/app/studio/pages/4_evaluation_panel.py`
- `rag/app/studio/components/metrics_table.py`
- `rag/app/studio/components/metric_glossary.py`

**Acceptance criteria**
- Evaluation panel shows a separate "Answer Quality" section
- Displays faithfulness, answer_relevancy, context_precision per query
- Shows aggregate averages across the golden test set
- Answer quality metric names also provide hover tooltips via the same glossary mechanism
- If RAGAS is not installed, section shows "RAGAS not available — install with `pip install ragas`"

**Test method**
- Run evaluation, open Streamlit panel
- Verify answer quality metrics display correctly
- Hover over answer-quality metric names and verify tooltips appear

---

# Phase 15 — External Connectors (Email / Slack / Notion / Google Docs)

## Task 15.1 — Define connector interface and sync contract
**Depends on:** 1.4a, 1.6

**Files to modify**
- `rag/core/interfaces/source_connector.py`
- `rag/core/contracts/source_artifact.py`
- `configs/connectors/sources.yaml`

**Acceptance criteria**
- Defines a common connector interface:
  - `list_items(since_cursor) -> list[SourceArtifact]`
  - `next_cursor() -> str`
  - `healthcheck() -> dict`
- `SourceArtifact` includes:
  - `source_type`
  - `source_id`
  - `external_url`
  - `content_bytes` or `content_text`
  - `metadata`
- Cursor/checkpoint format is provider-agnostic
- Cursor storage: persisted in DocStore via a `connector_state` table (`connector_name TEXT PRIMARY KEY, cursor TEXT, last_sync_at TIMESTAMP`). Schema creation is added to `docstore_sqlite.py`.

**Test method**
- Add `tests/test_source_connector_contract.py`
- Implement a fake connector and verify interface compatibility

---

## Task 15.2 — Implement Email connector
**Depends on:** 15.1

**Files to modify**
- `rag/infra/connectors/email_connector.py`

**Acceptance criteria**
- Can pull email messages since cursor
- Converts message body + selected attachments into `SourceArtifact`
- Emits stable `source_id` per message
- Stores provider cursor for next sync run

**Test method**
- Run connector on mocked mailbox payloads
- Verify deterministic `source_id`, artifact count, and cursor advancement

---

## Task 15.3 — Implement Slack connector
**Depends on:** 15.1

**Files to modify**
- `rag/infra/connectors/slack_connector.py`

**Acceptance criteria**
- Can pull channel/thread messages since cursor
- Flattens thread content into artifacts suitable for parsing/chunking
- Preserves channel/thread/message IDs in metadata
- Stores provider cursor for next sync run

**Test method**
- Run connector on mocked Slack API responses
- Verify artifact text ordering and metadata fields

---

## Task 15.4 — Implement Notion connector
**Depends on:** 15.1

**Files to modify**
- `rag/infra/connectors/notion_connector.py`

**Acceptance criteria**
- Can pull pages (and database rows rendered to text) since cursor
- Converts block-rich content into plain text artifacts with structural hints
- Preserves page/database IDs and workspace metadata
- Stores provider cursor for next sync run

**Test method**
- Run connector on mocked Notion API responses
- Verify page content extraction and stable source identifiers

---

## Task 15.5 — Implement Google Docs connector
**Depends on:** 15.1

**Files to modify**
- `rag/infra/connectors/google_docs_connector.py`

**Acceptance criteria**
- Can pull Google Docs files since cursor
- Converts document structure into ingestible text artifacts
- Preserves file/document IDs and drive links
- Stores provider cursor for next sync run

**Test method**
- Run connector on mocked Google Docs/Drive API responses
- Verify extracted text, source metadata, and cursor advancement

---

## Task 15.6 — Wire connector sync pipeline and MCP entrypoint
**Depends on:** 15.1, 3.12, 9.2

**Note:** This task only requires the connector interface (15.1) and at least one implemented connector to test with. Remaining connectors (15.2–15.5) can be completed before or after this task.

**Files to modify**
- `rag/pipelines/connector_sync_pipeline.py`
- `rag/app/mcp_server/schemas.py` (add `rag.sync_source` schema)
- `rag/app/mcp_server/server.py` (register `rag.sync_source` tool)

**Acceptance criteria**
- Adds a unified sync pipeline that:
  - runs selected connector
  - converts `SourceArtifact` to ingestable artifacts
  - invokes the existing ingest pipeline
- MCP adds `rag.sync_source` tool for triggering connector sync
- Sync run writes trace events and summary stats (fetched/ingested/skipped)

**Test method**
- Call `rag.sync_source` for a mocked connector (using fake connector from 15.1 tests)
- Verify chunks are written and query can retrieve synced content
- Verify run trace includes sync and ingest stages

---


# Recommended Build Order Summary

## Main path first
1. 0.1 → 0.2 → 0.3
2. 1.1 → 1.4c → 1.5 → 1.6
3. 2.1 → 2.4
4. 3.1 → 3.8d → 3.9 → 3.12 → 3.13
5. 4.1 → 4.7
6. 5.1 → 5.4
7. 6.1 → 6.3
8. 7.1 → 7.6
9. 8.1 → 8.4
10. 9.1 → 9.3
11. 10.1 → 10.2 → 10.3 → 10.4 → 10.5 → 10.6

## Then refine / release hardening
12. 11.1 → 11.5  (required before enabling incremental update in production)
13. 12.1 → 12.3
14. 6.4 (optional LLM rerank, V2)
15. 13.1 → 13.4 (OCR support for scanned PDFs)
16. 14.1 → 14.3 (RAGAS answer quality evaluation)
17. 15.1 → 15.6 (external connectors: Email / Slack / Notion / Google Docs)

---

# Definition of "End-to-End Main Path Done" (Demo Scope)

The main path is considered done when you can:

1. ingest a PDF/HTML/TXT file
2. chunk and embed it
3. search it with BM25 + vector
4. rerank results
5. generate a grounded answer
6. return citations
7. inspect the whole process in Streamlit
8. call the same flow through MCP

If production incremental update is enabled, Phase 11.5 must also be completed to prevent stale chunk versions from remaining in retrieval indexes.

---

## 19. Progress Tracking

Status legend: `[ ]` Not Started | `[~]` In Progress | `[x]` Completed

Update rule: After each task is completed, update the corresponding status, completion date, and notes.

### Phase 0 — Project Skeleton

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 0.1 | Create project folder structure | [x] | 2026-03-22 | Created full package tree and placeholders |
| 0.2 | Add Python project config | [x] | 2026-03-22 | pyproject.toml, requirements.txt, .env.example |
| 0.3 | Set up pytest infrastructure | [x] | 2026-03-22 | conftest.py, smoke tests, markers registered |

### Phase 1 — Core Contracts and Config

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 1.1 | Define core document contracts | [x] | 2026-03-22 | Document, IRBlock, ParseReport with Pydantic v2 |
| 1.2 | Define chunking and retrieval contracts | [x] | 2026-03-22 | TextBlock, Chunk, Candidate with Pydantic v2 |
| 1.3 | Define citation and answer contracts | [x] | 2026-03-22 | Span, Citation, Answer, AnswerTrace, PipelineStep |
| 1.4a | Define ingestion-side interface contracts | [x] | 2026-03-22 | BaseParser, BaseCleaner, BaseBlockSplitter, BaseChunkPacker |
| 1.4b | Define retrieval-side interface contracts | [x] | 2026-03-22 | BaseEmbeddingProvider, BaseVectorIndex, BaseKeywordIndex, BaseReranker |
| 1.4c | Define storage and generation interface contracts | [x] | 2026-03-22 | BaseDocStore, BaseTraceStore, BaseLLMClient, BaseContextPacker, BaseAnswerComposer |
| 1.5 | Add base settings config | [x] | 2026-03-22 | settings.yaml + local_fast/local_quality profiles |
| 1.6 | Build config loader | [x] | 2026-03-22 | load_config with deep merge and profile support |

### Phase 2 — SQLite Foundation

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 2.1 | Implement SQLite DocStore schema creation | [x] | 2026-03-22 | documents, text_blocks, chunks tables + indexes |
| 2.2 | Implement SQLite TraceStore schema creation | [x] | 2026-03-22 | runs, trace_events tables + indexes |
| 2.3 | Implement basic DocStore write/read methods | [x] | 2026-03-22 | SQLiteDocStore with full CRUD for docs/blocks/chunks |
| 2.4 | Implement basic TraceStore write/read methods | [x] | 2026-03-22 | SQLiteTraceStore with runs + answer trace CRUD |

### Phase 3 — Minimal Ingestion Main Path

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 3.1 | Implement local file loader | [x] | 2026-03-22 | RawArtifact + LocalFileLoader with metadata |
| 3.2 | Implement minimal sniffer | [x] | 2026-03-22 | MagicSniffer, MimeSniffer, CompositeSniffer |
| 3.3 | Implement minimal parser router | [x] | 2026-03-22 | ParsePlan + ParserOrchestrator with fallback |
| 3.4 | Implement TXT/Markdown parser | [x] | 2026-03-22 | MdParser: heading/paragraph/code blocks |
| 3.5 | Implement HTML parser | [x] | 2026-03-22 | HtmlTrafilaturaParser strips boilerplate |
| 3.6 | Implement PDF text parser | [x] | 2026-03-22 | PdfPyMuPDFParser with page numbers |
| 3.7 | Implement parse quality gates | [x] | 2026-03-22 | QualityGateChecker with YAML thresholds |
| 3.8a | Implement unicode_fix + empty_filter cleaners | [x] | 2026-03-22 | UnicodeFixer + EmptyBlockFilter cleaners |
| 3.8b | Implement dedupe + PDF header/footer cleaners | [x] | 2026-03-22 | DedupeParagraphs + PdfHeaderFooterDedupe |
| 3.8c | Implement HTML nav/footer + OCR line merge cleaners | [x] | 2026-03-22 | HtmlNavFooterRemover + OcrLineMerger |
| 3.8d | Wire cleaner pipeline | [x] | 2026-03-22 | CleanerPipeline wires all 6 steps via YAML |
| 3.9 | Implement paragraph block splitter | [x] | 2026-03-22 | ParagraphBlockSplitter with stable block_hash |
| 3.10 | Implement anchor annotator | [x] | 2026-03-22 | AnchorAnnotator with YAML rules |
| 3.11 | Implement chunk packer | [x] | 2026-03-22 | AnchorAwareChunkPacker with token budget |
| 3.12 | Implement minimal ingest pipeline | [x] | 2026-03-22 | IngestPipeline end-to-end for TXT/MD/HTML/PDF |
| 3.13 | Implement CLI ingest entry point | [x] | 2026-03-22 | CLI with --path/--collection, prints summary, exits 1 on error |

### Phase 4 — Embedding and Indexes

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 4.1 | Implement base embedding provider + factory | [x] | 2026-03-22 | EmbeddingResult + factory with lazy-import registry |
| 4.2 | Implement OpenAI embedding provider | [ ] | - | |
| 4.3 | Implement embedding batcher | [ ] | - | |
| 4.4 | Implement BM25 index | [ ] | - | |
| 4.5 | Implement FAISS vector index | [ ] | - | |
| 4.6 | Extend ingest pipeline to embed and index | [ ] | - | |
| 4.7 | Implement index startup loader | [ ] | - | |

### Phase 5 — Retrieval Main Path

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 5.1 | Implement retrieval source attribution | [ ] | - | |
| 5.2 | Implement RRF fusion | [ ] | - | |
| 5.3 | Implement query pipeline without rerank | [ ] | - | |
| 5.4 | Implement CLI query entry point | [ ] | - | |

### Phase 6 — Reranking

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 6.1 | Implement reranker interface and factory | [ ] | - | |
| 6.2 | Implement API cross-encoder reranker | [ ] | - | |
| 6.3 | Add rerank stage to query pipeline | [ ] | - | |
| 6.4 | Add optional LLM rerank stage (V2) | [ ] | - | |

### Phase 7 — LLM Abstraction and Generation

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 7.1 | Implement LLM client interface | [ ] | - | |
| 7.2 | Implement OpenAI LLM client | [ ] | - | |
| 7.3 | Implement light context packer | [ ] | - | |
| 7.4 | Implement grounded prompt builder | [ ] | - | |
| 7.5 | Implement answer composer | [ ] | - | |
| 7.6 | Add generation stage to query pipeline | [ ] | - | |

### Phase 8 — Streamlit Main Path

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 8.1 | Create Streamlit app shell | [ ] | - | |
| 8.2 | Build ingestion manager page | [ ] | - | |
| 8.3 | Build ingestion trace page | [ ] | - | |
| 8.4 | Build query trace page | [ ] | - | |

### Phase 9 — MCP Main Path

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 9.1 | Add MCP tool schemas | [ ] | - | |
| 9.2 | Implement MCP server wiring | [ ] | - | |
| 9.3 | End-to-end main path validation | [ ] | - | |

### Phase 10 — Evaluation

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 10.1 | Implement retrieval metrics | [ ] | - | |
| 10.2 | Implement source attribution diagnostics | [ ] | - | |
| 10.3 | Implement system efficiency metrics | [ ] | - | |
| 10.4 | Implement CLI eval entry point | [ ] | - | |
| 10.5 | Build evaluation query suites (example_queries + failure_cases) | [ ] | - | |
| 10.6 | Build evaluation page | [ ] | - | |

### Phase 11 — Incremental Ingestion Refinement

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 11.1 | Implement document fingerprint tracking | [ ] | - | |
| 11.2 | Implement block diff logic | [ ] | - | |
| 11.3 | Implement resync engine | [ ] | - | |
| 11.4 | Add threshold guardrails | [ ] | - | |
| 11.5 | Remove stale chunks from indexes after incremental update | [ ] | - | |

### Phase 12 — Nice-to-Have but Still Useful

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 12.1 | Add pronoun risk detection | [ ] | - | |
| 12.2 | Add metadata enrichment (rules first) | [ ] | - | |
| 12.3 | Add LLM metadata enrichment | [ ] | - | |

### Phase 13 — OCR Support (Scanned PDF)

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 13.1 | Implement OCR provider interface + PaddleOCR | [ ] | - | |
| 13.2 | Implement page renderer (PDF → image) | [ ] | - | |
| 13.3 | Implement scanned PDF parser | [ ] | - | |
| 13.4 | Integrate OCR parser into router | [ ] | - | |

### Phase 14 — RAGAS Answer Quality Evaluation

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 14.1 | Implement RAGAS evaluator interface | [ ] | - | |
| 14.2 | Create golden test set for answer evaluation | [ ] | - | |
| 14.3 | Add answer quality metrics to evaluation panel | [ ] | - | |

### Phase 15 — External Connectors (Email / Slack / Notion / Google Docs)

| Task ID | Task Name | Status | Completed | Notes |
|---|---|---|---|---|
| 15.1 | Define connector interface and sync contract | [ ] | - | |
| 15.2 | Implement Email connector | [ ] | - | |
| 15.3 | Implement Slack connector | [ ] | - | |
| 15.4 | Implement Notion connector | [ ] | - | |
| 15.5 | Implement Google Docs connector | [ ] | - | |
| 15.6 | Wire connector sync pipeline and MCP entrypoint | [ ] | - | |

## 📈 Overall Progress

| Phase | Total Tasks | Completed | Progress |
|---|---:|---:|---:|
| Phase 0 | 3 | 3 | 100% |
| Phase 1 | 8 | 8 | 100% |
| Phase 2 | 4 | 4 | 100% |
| Phase 3 | 16 | 16 | 100% |
| Phase 4 | 7 | 1 | 14% |
| Phase 5 | 4 | 0 | 0% |
| Phase 6 | 4 | 0 | 0% |
| Phase 7 | 6 | 0 | 0% |
| Phase 8 | 4 | 0 | 0% |
| Phase 9 | 3 | 0 | 0% |
| Phase 10 | 6 | 0 | 0% |
| Phase 11 | 5 | 0 | 0% |
| Phase 12 | 3 | 0 | 0% |
| Phase 13 | 4 | 0 | 0% |
| Phase 14 | 3 | 0 | 0% |
| Phase 15 | 6 | 0 | 0% |
| **Total** | **86** | **32** | **37%** |
