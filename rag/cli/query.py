"""CLI entry point for querying the RAG index.

Usage::

    python -m rag.cli.query "What is retrieval augmented generation?"
    python -m rag.cli.query "What is BM25?" --top-k 5
    python -m rag.cli.query "How does fusion work?" --verbose
"""

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.cli.query",
        description="Query the RAG index and print ranked results with citations.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Natural-language query string.",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name used to locate the database and index files. "
             "Defaults to 'default'.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Explicit path to the SQLite database file. "
             "If omitted, defaults to data/<collection>.db.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        dest="index_dir",
        help="Directory containing saved BM25 and FAISS index files. "
             "If omitted, defaults to data/<collection>_indexes/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results to return. Defaults to 5.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show retrieval scores and source attribution for each result.",
    )
    return parser


def _print_results(query: str, result, verbose: bool) -> None:
    """Print query results in a human-readable format.

    Args:
        query: Original query string.
        result: QueryResult from the pipeline.
        verbose: Whether to print score/attribution details.
    """
    elapsed_s = result.elapsed_ms / 1000
    print(f'\nResults for: "{query}"')
    print(f"Found {len(result.candidates)} result(s) in {elapsed_s:.2f}s\n")

    if not result.candidates:
        print("  (no results — try ingesting documents first)")
        return

    for i, (cand, cit) in enumerate(
        zip(result.candidates, result.citations), start=1
    ):
        print(f"[{i}] {cit.source_label}")
        # Show a short excerpt (first 120 chars)
        excerpt = cand.display_text.strip().replace("\n", " ")
        if len(excerpt) > 120:
            excerpt = excerpt[:117] + "..."
        print(f"    {excerpt}")
        if verbose:
            print(
                f"    rrf={cand.rrf_score:.4f}  "
                f"bm25={cand.bm25_score if cand.bm25_score is not None else 'n/a'}  "
                f"vec={cand.vector_score if cand.vector_score is not None else 'n/a'}  "
                f"source={cand.source_label}"
            )
        print()


def main(argv: list[str] | None = None) -> int:
    """Run the query CLI.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    from rag.infra.indexes.index_manager import IndexManager
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.query_pipeline import QueryPipeline

    args = _build_parser().parse_args(argv)

    db_path: Path = args.db if args.db else Path("data") / f"{args.collection}.db"
    index_dir: Path = (
        args.index_dir
        if args.index_dir
        else Path("data") / f"{args.collection}_indexes"
    )

    if not db_path.exists():
        print(
            f"Error: database not found at '{db_path}'. "
            "Run rag.cli.ingest first.",
            file=sys.stderr,
        )
        return 1

    trace_store = SQLiteTraceStore(db_path)

    # Load indexes — starts empty if no saved files yet (graceful first-run)
    manager = IndexManager(index_dir)

    pipeline = QueryPipeline(
        keyword_index=manager.bm25,
        vector_index=manager.faiss,
        trace_store=trace_store,
        top_k=args.top_k,
    )

    result = pipeline.query(args.query)

    if result.error:
        print(f"Error: query failed — {result.error}", file=sys.stderr)
        return 1

    _print_results(args.query, result, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
