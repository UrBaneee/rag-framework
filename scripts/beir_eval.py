#!/usr/bin/env python3
"""BEIR benchmark evaluation runner for the RAG framework.

Downloads a BEIR dataset, ingests the corpus as single-doc chunks into
*isolated* BM25+FAISS indexes, runs all queries through QueryPipeline,
and reports NDCG@10, Recall@100, MRR against the official human-labeled
qrels.

Your production data in data/ is NEVER touched.  All BEIR data lives in:
    data/beir/<dataset>/

Usage:
    # BM25 only (fast, no API key needed):
    python scripts/beir_eval.py --dataset fiqa

    # Hybrid BM25 + OpenAI embeddings:
    python scripts/beir_eval.py --dataset fiqa --embedding openai

    # Hybrid BM25 + local multilingual embeddings (no API key):
    python scripts/beir_eval.py --dataset scifact --embedding multilingual

    # Tune retrieval depth and metric cut-off:
    python scripts/beir_eval.py --dataset nfcorpus --top-k 100 --k 10

    # Save results to JSON:
    python scripts/beir_eval.py --dataset fiqa --output results/fiqa_eval.json

Prerequisites:
    pip install beir

Recommended datasets by size (smallest → largest):
    scifact   ~5K docs,  300 queries   (science fact-checking)
    fiqa      ~57K docs, 648 queries   (financial Q&A)          ← start here
    nfcorpus  ~3.6K docs, 323 queries  (medical, hard)
    trec-covid ~171K docs, 50 queries  (COVID-19 research)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Check beir is installed before importing anything else
# ---------------------------------------------------------------------------
try:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader
except ImportError:
    print(
        "\nERROR: 'beir' package not found.\n"
        "Install it with:  pip install beir\n"
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("beir_eval")

# ---------------------------------------------------------------------------
# BEIR published baselines (NDCG@10) for reference comparison
# ---------------------------------------------------------------------------
_BASELINES: dict[str, dict[str, float]] = {
    "fiqa":      {"BM25": 0.236, "DPR": 0.295, "ColBERT": 0.317},
    "scifact":   {"BM25": 0.665, "DPR": 0.631, "ColBERT": 0.671},
    "nfcorpus":  {"BM25": 0.325, "DPR": 0.189, "ColBERT": 0.305},
    "trec-covid":{"BM25": 0.656, "DPR": 0.332, "ColBERT": 0.677},
    "hotpotqa":  {"BM25": 0.603, "DPR": 0.391, "ColBERT": 0.593},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_dataset(dataset: str, data_dir: Path) -> Path:
    """Download and extract a BEIR dataset if not already present.

    Args:
        dataset: BEIR dataset name (e.g. "fiqa", "scifact").
        data_dir: Root directory where BEIR datasets are stored.

    Returns:
        Path to the dataset directory.
    """
    dataset_dir = data_dir / dataset
    if dataset_dir.exists():
        print(f"[cache] Using existing dataset at {dataset_dir}")
        return dataset_dir

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    print(f"[download] Downloading {dataset} from BEIR...")
    zip_path = beir_util.download_and_unzip(url, str(data_dir))
    print(f"[download] Extracted to {dataset_dir}")
    return dataset_dir


def _make_chunk(beir_doc_id: str, title: str, text: str) -> "Chunk":
    """Create a Chunk from a BEIR corpus document.

    Each BEIR document becomes exactly one chunk:
    - chunk_id  = beir_doc_id  (enables direct comparison to qrels)
    - doc_id    = beir_doc_id  (1-to-1 document/chunk relationship)
    - stable_text = "<title> <text>" (standard BEIR concatenation)

    No chunking or splitting is applied — the whole document is one unit.

    Args:
        beir_doc_id: Original BEIR corpus document ID.
        title: Document title (may be empty).
        text: Document body text.

    Returns:
        A Chunk ready for BM25 / FAISS indexing.
    """
    from rag.core.contracts.chunk import Chunk

    full_text = f"{title} {text}".strip() if title else text
    # Use beir_doc_id as both the chunk_id and the chunk_signature.
    # This makes QueryPipeline return beir_doc_ids directly in candidates,
    # so no mapping layer is needed when comparing against qrels.
    return Chunk(
        chunk_id=beir_doc_id,
        doc_id=beir_doc_id,
        stable_text=full_text,
        display_text=full_text,
        chunk_signature=beir_doc_id,
        block_hashes=[],
        token_count=len(full_text.split()),
        metadata={"beir_doc_id": beir_doc_id, "title": title},
    )


def _build_indexes(
    corpus: dict,
    embedding_provider,
    beir_data_dir: Path,
    batch_size: int = 64,
) -> tuple:
    """Ingest BEIR corpus into fresh BM25 and FAISS indexes.

    Skips re-building if index files already exist on disk (cache).

    Args:
        corpus: BEIR corpus dict {doc_id: {"title": ..., "text": ...}}.
        embedding_provider: Embedding provider instance, or None for BM25-only.
        beir_data_dir: Dataset-specific data directory for storing indexes.
        batch_size: Number of docs to embed per API call.

    Returns:
        (bm25_index, faiss_index_or_None) tuple ready for QueryPipeline.
    """
    from rag.infra.indexes.bm25_local import BM25LocalIndex
    from rag.infra.indexes.faiss_local import FaissLocalIndex

    index_dir = beir_data_dir / "indexes"
    bm25_path = index_dir / "bm25.pkl"
    faiss_path = index_dir / "faiss.index"
    has_embedding = embedding_provider is not None

    bm25 = BM25LocalIndex()
    faiss_idx = FaissLocalIndex() if has_embedding else None

    # Load from cache if both index files exist
    if bm25_path.exists():
        print(f"[cache] Loading BM25 index from {bm25_path}")
        bm25.load(str(index_dir))
        if has_embedding and faiss_path.exists():
            print(f"[cache] Loading FAISS index from {faiss_path}")
            faiss_idx.load(str(index_dir))
            return bm25, faiss_idx
        elif not has_embedding:
            return bm25, None

    # Build from scratch
    print(f"[index] Building indexes for {len(corpus):,} documents...")
    index_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = list(corpus.keys())
    chunks = [
        _make_chunk(did, corpus[did].get("title", ""), corpus[did].get("text", ""))
        for did in doc_ids
    ]

    # BM25 — add all chunks
    print(f"[index] Adding {len(chunks):,} docs to BM25 index...")
    bm25.add(chunks)

    # FAISS — embed in batches and add
    if has_embedding and faiss_idx is not None:
        total = len(chunks)
        print(f"[index] Embedding {total:,} docs in batches of {batch_size}...")
        embedded_chunks = []
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.stable_text for c in batch]
            vectors = embedding_provider.embed(texts)
            for chunk, vec in zip(batch, vectors):
                embedded_chunks.append(
                    chunk.model_copy(update={"embedding": vec})
                )
            done = min(start + batch_size, total)
            print(f"  {done:,}/{total:,} docs embedded", end="\r")
        print()
        faiss_idx.add(embedded_chunks)

    # Persist to disk
    bm25.save(str(index_dir))
    if faiss_idx is not None:
        faiss_idx.save(str(index_dir))
    print(f"[index] Indexes saved to {index_dir}")

    return bm25, faiss_idx


def _build_pipeline(bm25, faiss_idx, embedding_provider, trace_store, top_k: int):
    """Build a QueryPipeline wired to the BEIR indexes.

    Args:
        bm25: BM25LocalIndex populated with BEIR corpus.
        faiss_idx: FaissLocalIndex populated with BEIR corpus, or None.
        embedding_provider: Embedding provider for query encoding, or None.
        trace_store: TraceStore instance (minimal SQLite in BEIR data dir).
        top_k: Number of candidates to retrieve per index.

    Returns:
        Configured QueryPipeline instance.
    """
    from rag.pipelines.query_pipeline import QueryPipeline

    return QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss_idx,
        embedding_provider=embedding_provider,
        trace_store=trace_store,
        top_k=top_k,
        answer_composer=None,
    )


def _run_queries(
    pipeline,
    queries: dict,
    qrels: dict,
    k: int,
) -> tuple[list[dict], dict]:
    """Run all BEIR queries and collect results.

    Args:
        pipeline: Configured QueryPipeline.
        queries: BEIR queries dict {query_id: query_text}.
        qrels: BEIR relevance judgments {query_id: {doc_id: score}}.
        k: Metric cut-off depth.

    Returns:
        (raw_results, timing_info) where raw_results is the list expected
        by run_eval(), and timing_info has latency stats.
    """
    raw_results = []
    latencies = []
    query_ids = list(queries.keys())
    total = len(query_ids)

    print(f"[eval] Running {total:,} queries (k={k})...")

    for i, qid in enumerate(query_ids, 1):
        query_text = queries[qid]
        # Relevant docs for this query (any relevance score > 0)
        relevant_docs = [
            doc_id
            for doc_id, score in qrels.get(qid, {}).items()
            if score > 0
        ]

        t0 = time.perf_counter()
        result = pipeline.query(query_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # chunk_id == beir_doc_id (set in _make_chunk), so no mapping needed
        retrieved_doc_ids = [c.chunk_id for c in result.candidates]

        raw_results.append({
            "query_id": qid,
            "query": query_text,
            "retrieved": retrieved_doc_ids,
            "relevant": relevant_docs,
            "candidates": result.candidates,
            "query_latency_ms": elapsed_ms,
            "error": result.error,
        })

        if i % 50 == 0 or i == total:
            avg_ms = sum(latencies[-50:]) / len(latencies[-50:])
            print(f"  {i:,}/{total:,} queries done (avg {avg_ms:.0f} ms/query)", end="\r")

    print()
    timing = {
        "total_queries": total,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
    }
    return raw_results, timing


def _print_results(
    dataset: str,
    split: str,
    embedding_mode: str,
    report,
    timing: dict,
    k: int,
) -> None:
    """Print a formatted results table to stdout.

    Args:
        dataset: BEIR dataset name.
        split: Dataset split used (e.g. "test").
        embedding_mode: Embedding mode label ("BM25-only", "Hybrid/openai", etc.)
        report: EvalReport from run_eval().
        timing: Timing info dict from _run_queries().
        k: Metric cut-off used.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  BEIR Evaluation Results")
    print(f"  Dataset  : {dataset} ({split} split)")
    print(f"  Mode     : {embedding_mode}")
    print(f"  Queries  : {timing['total_queries']:,}")
    print(sep)
    print(f"\n  Retrieval Metrics (@ k={k})")
    print(f"  {'Recall@K':<20} {report.mean_recall_at_k:.4f}")
    print(f"  {'MRR':<20} {report.mrr:.4f}")
    print(f"  {'nDCG@K':<20} {report.mean_ndcg_at_k:.4f}")

    # Source attribution
    attr = report.source_attribution
    if attr and attr.total_candidates > 0:
        print(f"\n  Source Attribution ({attr.total_candidates:,} total candidates)")
        print(f"  {'BM25-only':<20} {attr.bm25_only * 100:.1f}%")
        print(f"  {'Vector-only':<20} {attr.vector_only * 100:.1f}%")
        print(f"  {'Hybrid (both)':<20} {attr.both * 100:.1f}%")

    # Latency
    print(f"\n  Efficiency")
    print(f"  {'Mean latency':<20} {timing['mean_latency_ms']:.1f} ms/query")
    print(f"  {'P95 latency':<20} {timing['p95_latency_ms']:.1f} ms/query")

    # Baseline comparison
    baselines = _BASELINES.get(dataset)
    if baselines:
        print(f"\n  BEIR Baselines (NDCG@10, published)")
        for system, score in baselines.items():
            marker = " ◄ your system" if abs(report.mean_ndcg_at_k - score) < 0.01 else ""
            print(f"  {'  ' + system:<20} {score:.3f}{marker}")
        your_ndcg = report.mean_ndcg_at_k
        best_baseline = max(baselines.values())
        if your_ndcg >= best_baseline:
            print(f"\n  ✅ Beats all published baselines! ({your_ndcg:.4f} vs {best_baseline:.3f})")
        elif your_ndcg >= baselines.get("BM25", 0):
            print(f"\n  ✅ Beats BM25 baseline ({your_ndcg:.4f} vs {baselines['BM25']:.3f})")
        else:
            gap = baselines.get("BM25", 0) - your_ndcg
            print(f"\n  ⚠️  Below BM25 baseline by {gap:.3f}. Check embedding alignment.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for BEIR evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run BEIR benchmark evaluation against your RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="fiqa",
        help="BEIR dataset name (default: fiqa). See README for full list.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--embedding",
        default="none",
        choices=["none", "openai", "multilingual"],
        help="Embedding mode. 'none' = BM25-only (default).",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help=(
            "Embedding model name. Defaults: "
            "openai='text-embedding-3-small', "
            "multilingual='paraphrase-multilingual-mpnet-base-v2'."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Candidates to retrieve per index (default: 100).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Metric cut-off depth for Recall/MRR/nDCG (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/beir",
        help="Root directory for BEIR datasets (default: data/beir).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save full results as JSON.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild indexes even if cached versions exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load .env for API keys
    try:
        from dotenv import load_dotenv
        env_file = _PROJECT_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass

    data_dir = _PROJECT_ROOT / args.data_dir
    beir_data_dir = data_dir / args.dataset

    # 1. Download dataset
    dataset_dir = _download_dataset(args.dataset, data_dir)

    # 2. Load corpus, queries, qrels
    print(f"[load] Loading {args.dataset} ({args.split} split)...")
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(
        split=args.split
    )
    print(
        f"[load] Corpus: {len(corpus):,} docs | "
        f"Queries: {len(queries):,} | "
        f"Qrels: {sum(len(v) for v in qrels.values()):,} judgments"
    )

    # 3. Build embedding provider
    embed_provider = None
    embedding_mode = "BM25-only"

    if args.embedding == "openai":
        from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
        model = args.embedding_model or "text-embedding-3-small"
        embed_provider = OpenAIEmbeddingProvider(model=model, dimensions=1536)
        embedding_mode = f"Hybrid/OpenAI ({model})"
        print(f"[embed] Using OpenAI embeddings: {model}")

    elif args.embedding == "multilingual":
        from rag.infra.embedding.multilingual_embedding import MultilingualEmbeddingProvider
        model = args.embedding_model or "paraphrase-multilingual-mpnet-base-v2"
        embed_provider = MultilingualEmbeddingProvider(model=model, dim=768)
        embedding_mode = f"Hybrid/Multilingual ({model})"
        print(f"[embed] Using multilingual embeddings: {model}")

    # 4. Clear cached indexes if requested
    if args.rebuild_index:
        import shutil
        index_dir = beir_data_dir / "indexes"
        if index_dir.exists():
            shutil.rmtree(index_dir)
            print(f"[index] Cleared cached indexes at {index_dir}")

    # 5. Build / load indexes
    bm25, faiss_idx = _build_indexes(
        corpus=corpus,
        embedding_provider=embed_provider,
        beir_data_dir=beir_data_dir,
        batch_size=args.batch_size,
    )

    # 6. Build trace store (isolated, in BEIR data dir)
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    trace_db = beir_data_dir / "beir_traces.db"
    trace_store = SQLiteTraceStore(str(trace_db))

    # 7. Build pipeline
    pipeline = _build_pipeline(
        bm25=bm25,
        faiss_idx=faiss_idx,
        embedding_provider=embed_provider,
        trace_store=trace_store,
        top_k=args.top_k,
    )

    # 8. Run queries
    raw_results, timing = _run_queries(
        pipeline=pipeline,
        queries=queries,
        qrels=qrels,
        k=args.k,
    )

    # 9. Compute metrics
    from rag.pipelines.eval_pipeline import run_eval
    report = run_eval(raw_results, k=args.k)

    # 10. Print results
    _print_results(
        dataset=args.dataset,
        split=args.split,
        embedding_mode=embedding_mode,
        report=report,
        timing=timing,
        k=args.k,
    )

    # 11. Optionally save full results to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "dataset": args.dataset,
            "split": args.split,
            "embedding_mode": embedding_mode,
            "top_k": args.top_k,
            "k": args.k,
            "metrics": {
                "recall_at_k": report.mean_recall_at_k,
                "mrr": report.mrr,
                "ndcg_at_k": report.mean_ndcg_at_k,
            },
            "source_attribution": {
                "bm25_only": report.source_attribution.bm25_only if report.source_attribution else None,
                "vector_only": report.source_attribution.vector_only if report.source_attribution else None,
                "both": report.source_attribution.both if report.source_attribution else None,
                "total_candidates": report.source_attribution.total_candidates if report.source_attribution else 0,
            },
            "timing": timing,
            "per_query": [
                {
                    "query_id": r["query_id"],
                    "query": r["query"],
                    "retrieved_count": len(r["retrieved"]),
                    "relevant_count": len(r["relevant"]),
                }
                for r in raw_results
            ],
        }
        output_path.write_text(json.dumps(output, indent=2))
        print(f"[output] Results saved to {output_path}")


if __name__ == "__main__":
    main()
