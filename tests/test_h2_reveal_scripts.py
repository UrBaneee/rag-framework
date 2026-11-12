"""Tests for the reveal-time scripts (附录 HP). Synthetic events only."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp = _load("h2_reveal_export")
full = _load("tb5_build_full_store")


def _ev(eid, split="test", etype="normal", batch="v2", probes=("p1", "p2")):
    return {"event_id": eid, "split": split, "event_type": etype, "batch": batch,
            "business_description": "synthetic cause", "valid_from": "2024-01-01", "valid_to": "2024-01-31",
            "data_signal": {"metric": "revenue", "direction": "down", "magnitude": 0.25,
                            "segment": {"region": "North"}},
            "h2_probes": list(probes)}


def test_pool_selects_sealed_test_normal_v2_only():
    sealed = {"events": [_ev("EVT-A"), _ev("EVT-B", etype="distractor"), _ev("EVT-C", batch="v1"),
                         _ev("EVT-D", split="dev")]}
    assert [e["event_id"] for e in exp.pool_events(sealed)] == ["EVT-A"]


def test_cases_one_per_probe_sorted_with_dev_shape():
    cases = exp.build_cases([_ev("EVT-B"), _ev("EVT-A")], "test")
    assert [c["id"] for c in cases] == ["H2-A-Q1", "H2-A-Q2", "H2-B-Q1", "H2-B-Q2"]
    c = cases[0]
    assert c["split"] == "test" and c["event_cause"] == "synthetic cause"
    assert c["event_anomaly"] == "revenue down by about 25% in region=North, 2024-01-01 to 2024-01-31"
    assert c["data_source"]["path"] == "data/demo/sales_data.csv"


def test_anomaly_text_matches_maeda_dev_cases():
    """Pin the copied wording to MAEDA's committed dev cases (public dev events only)."""
    maeda = Path("/Users/yintianyi/MAEDA/tests/eval/h2_dev_attribution_cases.json")
    if not maeda.is_file():
        pytest.skip("MAEDA checkout not present")
    from rag.eval.split_guard import load_event_script

    # load_event_script(dev) reads the working public yaml; it holds v1 + v2 dev events.
    dev = [e for e in load_event_script("dev") if e.get("h2_probes")]
    mine = {c["id"]: c for c in exp.build_cases(dev, "dev")}
    theirs = {c["id"]: c for c in json.loads(maeda.read_text())}
    assert mine and set(mine) == set(theirs)
    assert all(mine[k] == theirs[k] for k in theirs)


def test_export_refuses_without_confirmation_and_on_tracked_path(tmp_path, capsys):
    assert exp.main(["--out", str(tmp_path / "o.json")]) == 2
    assert exp.main(["--out", str(_ROOT / "tests" / "leak.json"), "--reveal-confirmed"]) == 2
    assert not (_ROOT / "tests" / "leak.json").exists()


def test_path_safety(tmp_path):
    assert exp.path_is_safe(tmp_path / "x.json")
    assert not exp.path_is_safe(_ROOT / "tests" / "x.json")
    assert exp.path_is_safe(_ROOT / "data" / "x.json")


def test_full_store_dry_run_builds_nothing(capsys):
    before = full.DB_PATH.exists()
    assert full.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and full.DB_PATH.exists() == before
    assert full.main([]) == 2
