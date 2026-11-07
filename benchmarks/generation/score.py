"""Scoring and report formatting for frozen plan-generation benchmark results."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def edge_satisfaction(plan: dict[str, Any], graph: list[dict[str, Any]], report: dict[str, Any]) -> tuple[int, int]:
    room_ids = {room.get("id") for room in plan.get("rooms", [])}
    scoped_edges = [edge for edge in graph if edge.get("a") in room_ids and edge.get("b") in room_ids]
    missed = sum(violation.get("code") == "adjacency" for violation in report.get("violations", []))
    return max(0, len(scoped_edges) - missed), len(scoped_edges)


def grade(result: dict[str, Any], calls: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    plan = result.get("plan", {})
    graph = result.get("graph", [])
    validation = result.get("validation", {})
    satisfied, total_edges = edge_satisfaction(plan, graph, validation)
    chain = [entry for entry in result.get("chain", []) if isinstance(entry, dict)]
    models = sorted(
        {
            call["model"]
            for call in calls
            if call.get("kind") == "call" and isinstance(call.get("model"), str)
        }
    )
    return {
        "validation_score": validation.get("score"),
        "errors": sum(item.get("severity") == "error" for item in validation.get("violations", [])),
        "violations": len(validation.get("violations", [])),
        "graph_edges_satisfied": round(satisfied / total_edges, 3) if total_edges else 1.0,
        "graph_edges_total": total_edges,
        "fallbacks": [
            entry.get("stage")
            for entry in chain
            if mode == "live" and entry.get("model") == "rules"
        ],
        "models": models or ["rules"],
        "tokens_in": sum(call.get("prompt_tokens", 0) for call in calls if call.get("kind") == "call"),
        "tokens_out": sum(call.get("completion_tokens", 0) for call in calls if call.get("kind") == "call"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "ok"]
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_mode[row["mode"]].append(row)

    modes: dict[str, dict[str, Any]] = {}
    for mode, cohort in by_mode.items():
        metrics = [row["metrics"] for row in cohort]
        modes[mode] = {
            "completed": len(cohort),
            "mean_validation_score": round(statistics.mean(m["validation_score"] for m in metrics), 2),
            "mean_latency_s": round(statistics.mean(row["latency_s"] for row in cohort), 2),
            "mean_graph_edges_satisfied": round(statistics.mean(m["graph_edges_satisfied"] for m in metrics), 3),
            "total_errors": sum(m["errors"] for m in metrics),
            "total_fallbacks": sum(len(m["fallbacks"]) for m in metrics),
        }

    deltas: list[dict[str, Any]] = []
    rules = {row["case_id"]: row for row in by_mode.get("rules", [])}
    live = {row["case_id"]: row for row in by_mode.get("live", [])}
    for case_id in sorted(rules.keys() & live.keys()):
        baseline, candidate = rules[case_id], live[case_id]
        deltas.append(
            {
                "case_id": case_id,
                "validation_score": round(
                    candidate["metrics"]["validation_score"] - baseline["metrics"]["validation_score"], 2
                ),
                "latency_s": round(candidate["latency_s"] - baseline["latency_s"], 2),
                "graph_edges_satisfied": round(
                    candidate["metrics"]["graph_edges_satisfied"]
                    - baseline["metrics"]["graph_edges_satisfied"],
                    3,
                ),
            }
        )
    return {"modes": modes, "deltas": deltas}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Plan-generation benchmark",
        "",
        f"Created: `{report['created_at']}`",
        f"Cases: `{len(report['cases'])}`",
        "",
        "## Cohort summary",
        "",
        "| Mode | Completed | Mean score | Mean latency (s) | Mean graph satisfaction | Errors | Fallbacks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in report["summary"]["modes"].items():
        lines.append(
            f"| {mode} | {values['completed']} | {values['mean_validation_score']} | "
            f"{values['mean_latency_s']} | {values['mean_graph_edges_satisfied']} | "
            f"{values['total_errors']} | {values['total_fallbacks']} |"
        )
    if report["summary"]["deltas"]:
        lines += [
            "",
            "## Live minus rules",
            "",
            "| Case | Score delta | Latency delta (s) | Graph-satisfaction delta |",
            "| --- | ---: | ---: | ---: |",
        ]
        for delta in report["summary"]["deltas"]:
            lines.append(
                f"| {delta['case_id']} | {delta['validation_score']} | {delta['latency_s']} | "
                f"{delta['graph_edges_satisfied']} |"
            )
    lines += [
        "",
        "A live comparison is only meaningful when each case completed in both cohorts. "
        "Rules-mode output is deterministic; live-model output can vary between runs.",
        "",
    ]
    return "\n".join(lines)
