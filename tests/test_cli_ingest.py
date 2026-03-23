"""Tests for rag.cli.ingest — CLI entry point."""

import textwrap
from pathlib import Path

import pytest

from rag.cli.ingest import main


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    doc = tmp_path / "sample.txt"
    doc.write_text(
        textwrap.dedent("""\
            # Introduction

            This is the first paragraph of the document.

            It has enough content to pass the quality gate.

            ## Section Two

            Second section with more interesting content here.
        """)
    )
    return doc


def test_main_success_prints_summary(sample_txt: Path, tmp_path: Path, capsys):
    db = tmp_path / "rag.db"
    exit_code = main(["--path", str(sample_txt), "--db", str(db)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "sample.txt" in captured.out
    assert "chunks" in captured.out
    assert "blocks" in captured.out


def test_main_creates_database_rows(sample_txt: Path, tmp_path: Path):
    import sqlite3

    db = tmp_path / "rag.db"
    main(["--path", str(sample_txt), "--db", str(db)])

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM text_blocks").fetchone()[0] > 0
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] > 0
    conn.close()


def test_main_missing_file_returns_exit_code_1(tmp_path: Path):
    db = tmp_path / "rag.db"
    exit_code = main(["--path", str(tmp_path / "no_such_file.txt"), "--db", str(db)])
    assert exit_code == 1


def test_main_default_collection_creates_db_in_data_dir(sample_txt: Path, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exit_code = main(["--path", str(sample_txt), "--collection", "mycollection"])
    assert exit_code == 0
    assert (tmp_path / "data" / "mycollection.db").exists()
