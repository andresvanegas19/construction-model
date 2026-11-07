"""Shared frozen-case and backend-tool helpers for product benchmarks."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "benchmarks" / "cases" / "plan-generation.json"
TOOLS_PATH = ROOT / "backend" / "ai" / "tools"


def load_cases(only: set[str] | None = None) -> list[dict[str, Any]]:
    data = json.loads(CASES_PATH.read_text())
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{CASES_PATH} must contain a non-empty cases list")
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)) or any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise ValueError(f"{CASES_PATH} must contain unique, non-empty case ids")
    selected = [case for case in cases if only is None or case["id"] in only]
    missing = (only or set()) - {case["id"] for case in selected}
    if missing:
        raise ValueError(f"Unknown benchmark case ids: {', '.join(sorted(missing))}")
    return selected


def render2d_module():
    if str(TOOLS_PATH) not in sys.path:
        sys.path.insert(0, str(TOOLS_PATH))
    import render2d

    return render2d


def intake_for(case: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    render2d = render2d_module()
    footprint = render2d.tool(
        "intake-defaults",
        {"shape": case["shape"], "sqft": case["sqft"]},
        env=environ,
        quiet=True,
    )["footprint"]
    intake = {
        "step": "ready",
        "floors": case["floors"],
        "sqft": case["sqft"],
        "footprint": footprint,
        "rooms": case["rooms"],
    }
    for key in ("open_concept", "extras"):
        if key in case:
            intake[key] = case[key]
    return intake
