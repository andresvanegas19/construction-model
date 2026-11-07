#!/usr/bin/env python3
"""Run frozen generation cases in rules mode and, with --live, against the configured model."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ROOT, intake_for, load_cases, render2d_module
from score import grade, markdown, summarize


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def output_path(argument: str | None) -> Path:
    if argument:
        path = Path(argument)
        return path if path.suffix == ".json" else path / f"generation-{timestamp()}.json"
    return ROOT / "benchmarks" / "results" / f"generation-{timestamp()}.json"


def run_case(case: dict[str, Any], mode: str, environ: dict[str, str]) -> dict[str, Any]:
    render2d = render2d_module()
    started = time.monotonic()
    try:
        intake = intake_for(case, environ)
        reply = render2d.tool("generate", {"intake": intake}, env=environ, quiet="warnings")
        elapsed = round(time.monotonic() - started, 3)
        result, calls = reply["result"], reply.get("calls", [])
        return {
            "case_id": case["id"],
            "family": case["family"],
            "mode": mode,
            "status": "ok",
            "latency_s": elapsed,
            "metrics": grade(result, calls, mode),
        }
    except Exception as error:  # Kept as a failed receipt; it must not look like a valid benchmark result.
        return {
            "case_id": case["id"],
            "family": case["family"],
            "mode": mode,
            "status": "failed",
            "latency_s": round(time.monotonic() - started, 3),
            "error": repr(error),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Run both rules and configured live-model cohorts.")
    parser.add_argument("--only", default="", help="Comma-separated frozen case ids.")
    parser.add_argument("--out", help="Report JSON path or output directory.")
    args = parser.parse_args()

    selected = {item for item in args.only.split(",") if item} or None
    cases = load_cases(selected)
    render2d = render2d_module()
    modes = ["rules", "live"] if args.live else ["rules"]
    rows = []
    model_info: dict[str, Any] = {}

    for mode in modes:
        environ = render2d.env(MOCK_LLM="0" if mode == "live" else "1")
        if mode == "live":
            model_info[mode] = render2d.pin_models(environ)
            if model_info[mode]["mock"]:
                raise RuntimeError("The live cohort requires configured model credentials; the backend resolved to rules mode.")
        else:
            model_info[mode] = {"models": {"generation": "rules"}, "mock": True}
        for case in cases:
            row = run_case(case, mode, environ)
            rows.append(row)
            print(f"{mode:5s} {case['id']}: {row['status']} ({row['latency_s']}s)", flush=True)

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": [case["id"] for case in cases],
        "modes": modes,
        "model_info": model_info,
        "rows": rows,
        "summary": summarize(rows),
    }
    path = output_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".md").write_text(markdown(report))
    print(f"wrote {path} and {path.with_suffix('.md')}")
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
