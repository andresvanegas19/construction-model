#!/usr/bin/env python3
"""Measure the HTTP generation flow and retain only complete, contract-checked receipts."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generation.common import ROOT, intake_for, load_cases, render2d_module


def stream_generation(base: str, intake: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base}/api/plans/generate",
        data=json.dumps({"intake": intake, "think": "simple"}).encode(),
        method="POST",
        headers={"content-type": "application/json"},
    )
    started = time.monotonic()
    first_event_s: float | None = None
    events: list[str] = []
    record: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                if first_event_s is None:
                    first_event_s = round(time.monotonic() - started, 3)
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError as error:
                    return {"status": status, "error": f"invalid_sse_json: {error}", "events": events}
                if isinstance(event, dict):
                    events.append(str(event.get("event", "unnamed")))
                    if isinstance(event.get("record"), dict):
                        record = event["record"]
    except urllib.error.HTTPError as error:
        return {"status": error.code, "error": error.read().decode("utf-8", "replace"), "events": events}
    except OSError as error:
        return {"status": None, "error": repr(error), "events": events}
    return {
        "status": status,
        "first_event_s": first_event_s,
        "completion_s": round(time.monotonic() - started, 3),
        "events": events,
        "record": record,
    }


def run_case(base: str, case: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    render2d = render2d_module()
    intake = intake_for(case, environ)
    streamed = stream_generation(base, intake)
    receipt = {"case_id": case["id"], "family": case["family"], **{k: v for k, v in streamed.items() if k != "record"}}
    record = streamed.get("record")
    plan_id = record.get("plan_id") if isinstance(record, dict) else None
    if streamed.get("status") != 200 or not isinstance(plan_id, str):
        receipt.update(status="failed", error=receipt.get("error", "missing saved record"))
        return receipt
    get_started = time.monotonic()
    status, saved = render2d.request(base, "GET", f"/api/plans/{plan_id}", timeout=60)
    receipt["plan_get_s"] = round(time.monotonic() - get_started, 3)
    if status != 200 or not isinstance(saved, dict) or not isinstance(saved.get("plan"), dict):
        receipt.update(status="failed", error=f"saved plan contract failed: HTTP {status}")
        return receipt
    receipt.update(status="ok", plan_id=plan_id, rooms=len(saved["plan"].get("rooms", [])))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="Comma-separated frozen case ids.")
    parser.add_argument("--out", help="Report JSON path or output directory.")
    args = parser.parse_args()
    selected = {item for item in args.only.split(",") if item} or None
    cases = load_cases(selected)
    render2d = render2d_module()
    out = Path(args.out) if args.out else ROOT / "benchmarks" / "results" / f"api-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if not out.suffix:
        out = out / f"api-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"

    with tempfile.TemporaryDirectory(prefix="render2d-api-benchmark-") as directory:
        environ = render2d.env(
            MOCK_LLM="1",
            STORAGE_BACKEND="files",
            DATA_DIR=str(Path(directory) / "data"),
            LOG_DIR=str(Path(directory) / "logs"),
            USAGE_ENFORCE="0",
        )
        with render2d.server(environ, ready_s=90) as base:
            receipts = [run_case(base, case, environ) for case in cases]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_mode": "rules",
        "receipts": receipts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    return 1 if any(receipt["status"] != "ok" for receipt in receipts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
