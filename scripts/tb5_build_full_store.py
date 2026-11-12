"""Build the reveal retrieval store ``tb5_full.db`` (附录 HP). Reveal-time only.

Corpus = the public TB5 corpus (``eval/fixtures/tb5_retrieval/corpus/``, the
same 16 event directories as ``tb5_dev.db``) plus the sealed v2 test corpus
(``eval/sealed/corpus/``). Output ``data/tb5_full.db`` and
``data/tb5_full_indexes/`` (both git-ignored). Embeddings: OpenAI
``text-embedding-3-small`` through ``build_ingest_pipeline``, the path that built
``tb5_dev.db``. PAID (a few thousand tokens, well under one cent).

Nothing about sealed files is printed: no file or event names, only counts.

    python scripts/tb5_build_full_store.py --dry-run          # counts, no API, no writes
    python scripts/tb5_build_full_store.py --reveal-confirmed # builds; needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PUBLIC_CORPUS = _ROOT / "eval" / "fixtures" / "tb5_retrieval" / "corpus"
SEALED_CORPUS = _ROOT / "eval" / "sealed" / "corpus"
DB_PATH = _ROOT / "data" / "tb5_full.db"
INDEX_DIR = _ROOT / "data" / "tb5_full_indexes"


def _counts(root: Path) -> tuple[int, int, int]:
    """(event directories, files, characters) under ``root``."""
    if not root.is_dir():
        return 0, 0, 0
    files = [p for p in root.rglob("*") if p.is_file()]
    dirs = [p for p in root.iterdir() if p.is_dir()]
    return len(dirs), len(files), sum(len(p.read_text(encoding="utf-8", errors="replace")) for p in files)


def _ignored(path: Path) -> bool:
    return subprocess.run(["git", "-C", str(_ROOT), "check-ignore", "-q", str(path)]).returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reveal-confirmed", action="store_true",
                    help="required to build: this is the user-authorised reveal run")
    args = ap.parse_args(argv)

    pd, pf, pc = _counts(PUBLIC_CORPUS)
    sd, sf, sc = _counts(SEALED_CORPUS)
    print(f"public corpus: {pd} event dirs, {pf} files, {pc} chars")
    print(f"sealed corpus: {sd} event dirs, {sf} files, {sc} chars")
    if args.dry_run:
        print("dry run: nothing built, no API call.")
        return 0
    if not args.reveal_confirmed:
        print("refused: pass --reveal-confirmed (or --dry-run).", file=sys.stderr)
        return 2
    if sf == 0:
        print("refused: sealed corpus absent; a store without it is not the reveal store.", file=sys.stderr)
        return 2
    if not (_ignored(DB_PATH) and _ignored(INDEX_DIR / "x")):
        print("refused: output paths are not git-ignored.", file=sys.stderr)
        return 2
    if DB_PATH.exists() or INDEX_DIR.exists():
        print("refused: output already exists; remove it deliberately first.", file=sys.stderr)
        return 2

    from dotenv import load_dotenv

    load_dotenv(override=False)
    from rag.app.mcp_server.wiring import build_ingest_pipeline
    from rag.pipelines.batch_ingest import BatchIngestRunner

    pipeline, mgr = build_ingest_pipeline(
        db_path=str(DB_PATH), index_dir=str(INDEX_DIR), embedding_provider="openai"
    )
    runner = BatchIngestRunner(pipeline, pipeline._trace_store)
    reports = []
    pipeline.begin_batch()
    for root in (PUBLIC_CORPUS, SEALED_CORPUS):
        reports.append(runner.run(root, recursive=True, on_progress=lambda line: None))
    pipeline.end_batch()
    mgr.save()

    problems = 0
    for name, rep in zip(("public", "sealed"), reports):
        print(f"{name}: {rep.total} files, statuses {rep.status_counts()}, "
              f"embed tokens {rep.total_embed_tokens()}, cost ${rep.total_cost_usd():.6f}")
        problems += len(rep.problem_results())
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
