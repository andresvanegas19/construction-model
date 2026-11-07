"""Matched cheap-only, larger-only and cascade probes; execute only on Tenki."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def matches(result, expected):
    if expected is None:
        return result is None
    if not isinstance(result, dict):
        return False
    return all(result.get(key) == value for key, value in expected.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--small", required=True)
    parser.add_argument("--large", required=True)
    parser.add_argument("--split", choices=["dev", "holdout"], required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.jsonl").exists():
        parser.error("output already contains a run; choose a fresh evidence directory")
    source = Path(__file__).with_name("cases.json")
    cases = [case for case in json.loads(source.read_text()) if case["split"] == args.split]
    rows = []
    for mode, small, strong in [
        ("cheap", args.small, ""),
        ("large", args.large, ""),
        ("cascade", args.small, args.large),
    ]:
        for case in cases:
            stem = f"{mode}-{case['id']}"
            private = (out / "private" / stem).resolve()
            private.mkdir(parents=True, mode=0o700)
            env = dict(os.environ, INTENT_TOOLS_MODEL=small, INTENT_TOOLS_ESCALATION_MODEL=strong,
                       LLM_DATASET="1", STORAGE_BACKEND="files", DATA_DIR=str(private),
                       APIFY_TOKEN="synthetic-probe-no-lookup",
                       MOCK_LLM="0", JEV_API_KEY="", TYPESAFE_API_KEY="")
            proc = subprocess.run([args.binary], input=json.dumps(case), text=True, capture_output=True, env=env)
            (out / f"{stem}.stdout.json").write_text(proc.stdout)
            (out / f"{stem}.stderr.log").write_text(proc.stderr)
            error = None
            try:
                reply = json.loads(proc.stdout)
            except json.JSONDecodeError:
                reply = {}
                error = "invalid_probe_output"
            calls = [call for call in reply.get("calls", []) if call.get("kind") == "call"]
            event_path = private / "llm_interactions.jsonl"
            events = [json.loads(line) for line in event_path.read_text().splitlines()] if event_path.exists() else []
            raw_calls = [event for event in events if event.get("kind") == "call"]
            failures = [call for call in calls if not call.get("json_ok", False)]
            valid_null = bool(calls) and not failures and "flag unsupported_requirement" in proc.stderr
            row = {
                "mode": mode, "case": case["id"], "split": args.split, "exit": proc.returncode,
                "passed": proc.returncode == 0 and error is None
                          and matches(reply.get("result"), case["expected"])
                          and (case["expected"] is not None or valid_null),
                "result": reply.get("result"), "elapsed_ms": reply.get("elapsed_ms"), "error": error,
                "calls": len(calls), "failed_calls": len(failures),
                "raw_call_events": len(raw_calls),
                "evidence_complete": bool(calls) and len(raw_calls) == len(calls),
                "private_events": str(event_path.relative_to(out.resolve())),
                "served_models": sorted({call.get("model", "unknown") for call in calls}),
                "requested_models": sorted({call.get("requested_model", call.get("model", "unknown")) for call in calls}),
                "flags": re.findall(r"intent tools: flag ([a-z_]+)", proc.stderr),
                "prompt_tokens": sum(call.get("prompt_tokens") or 0 for call in calls),
                "completion_tokens": sum(call.get("completion_tokens") or 0 for call in calls),
                "cost_usd": sum(call["cost_usd"] for call in raw_calls)
                            if raw_calls and all(isinstance(call.get("cost_usd"), (int, float)) for call in raw_calls) else None,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            rows.append(row)
            with (out / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps({key: row[key] for key in ["mode", "case", "passed", "exit", "error"]}), flush=True)
    summary = {
        "split": args.split, "requested": len(cases) * 3, "recorded": len(rows),
        "small_requested": args.small, "large_requested": args.large,
        "cost_status": "Provider-reported cost only; parameter count is not measured expense.",
        "modes": {mode: {"passed": sum(row["passed"] for row in rows if row["mode"] == mode),
                         "total": sum(row["mode"] == mode for row in rows)}
                  for mode in ["cheap", "large", "cascade"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
