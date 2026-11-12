"""Export the H2 test-pool attribution cases from the sealed overlay.

Reveal-time only (附录 HP). Reads ``eval/sealed/event_script_v2_test.yaml`` via
``rag.eval.split_guard.load_sealed_source`` and writes one case per
``h2_probes`` entry of every sealed ``split == test``, ``event_type == normal``,
``batch == v2`` event, in the same shape as MAEDA's
``tests/eval/h2_dev_attribution_cases.json`` (built by
``MAEDA/scripts/build_h2_dev_cases.py``; ``anomaly_text`` is a copy of that
function, and a test pins the two together on dev events).

The output contains test-split text. It is refused unless the path is
git-ignored (or outside any git repository), so it cannot be committed by
accident. Stdout carries counts only.

    python scripts/h2_reveal_export.py --reveal-confirmed --out <ignored path>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def anomaly_text(event: dict) -> str:
    """Same wording as MAEDA ``build_h2_dev_cases.anomaly_text``."""
    signal = event.get("data_signal")
    if not signal:
        return ("None in the data: the event below left no measurable effect. "
                f"Event: {event['business_description']}")
    segment = ", ".join(f"{k}={v}" for k, v in (signal.get("segment") or {}).items()) or "all regions"
    return (f"{signal['metric']} {signal['direction']} by about {signal['magnitude']:.0%} "
            f"in {segment}, {event['valid_from']} to {event['valid_to']}")


def build_cases(events: list[dict], split: str) -> list[dict]:
    """One case per ``h2_probes`` entry of each ``split`` event, sorted by event id."""
    cases = []
    for event in sorted(events, key=lambda e: e["event_id"]):
        if event["split"] != split:
            continue
        cause = event["business_description"] if event["event_type"] == "normal" else None
        metric = (event.get("data_signal") or {}).get("metric", "revenue")
        for n, query in enumerate(event["h2_probes"], 1):
            cases.append({
                "id": f"H2-{event['event_id'].removeprefix('EVT-')}-Q{n}",
                "query": query,
                "query_type": "diagnostic",
                "expected_metrics": [metric],
                "expected_dimensions": [],
                "ground_truth": {},
                "tags": ["h2_attribution", event["event_type"]],
                "data_source": {"type": "csv", "path": "data/demo/sales_data.csv"},
                "split": split,
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "event_anomaly": anomaly_text(event),
                "event_cause": cause,
            })
    return cases


def pool_events(sealed: dict[str, Any]) -> list[dict]:
    """The H2 test pool (预注册 §10.2): sealed test, normal, v2 events."""
    return [e for e in sealed.get("events", [])
            if e.get("split") == "test" and e.get("event_type") == "normal"
            and e.get("batch", "v2") == "v2"]


def path_is_safe(out: Path) -> bool:
    """True if ``out`` is git-ignored, or not inside any git work tree."""
    parent = out.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    inside = subprocess.run(["git", "-C", str(parent), "rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True)
    if inside.returncode != 0:
        return True
    ignored = subprocess.run(["git", "-C", str(parent), "check-ignore", "-q", str(out.resolve())],
                             capture_output=True)
    return ignored.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reveal-confirmed", action="store_true",
                    help="required: this is the user-authorised reveal run")
    args = ap.parse_args(argv)
    if not args.reveal_confirmed:
        print("refused: pass --reveal-confirmed (reveal-time only).", file=sys.stderr)
        return 2
    if not path_is_safe(args.out):
        print("refused: --out is inside a git work tree and not ignored.", file=sys.stderr)
        return 2
    from rag.eval.split_guard import load_sealed_source

    _, sealed = load_sealed_source(reason="H2 reveal export of the test-pool attribution cases")
    events = pool_events(sealed)
    if not events:
        print("refused: sealed file absent or holds no pool events.", file=sys.stderr)
        return 2
    cases = build_cases(events, "test")
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases from {len(events)} pool events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
