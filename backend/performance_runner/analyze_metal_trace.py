#!/usr/bin/env python3
"""Summarize one target process from an xctrace Metal GPU-interval export."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _cell_value(cell: ET.Element, definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ref = cell.get("ref")
    if ref is not None:
        return definitions.get(ref, {"tag": cell.tag, "raw": None, "fmt": None})
    value = {
        "tag": cell.tag,
        "raw": (cell.text or "").strip() or None,
        "fmt": cell.get("fmt"),
    }
    identifier = cell.get("id")
    if identifier is not None:
        definitions[identifier] = value
    return value


def _number(value: dict[str, Any]) -> int | None:
    raw = value.get("raw")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _label_group(label: str) -> str:
    value = re.sub(r"\s+\(\s*python[^)]*\([^)]*\)\s*\)\s+0x[0-9a-f]+$", "", label, flags=re.I)
    value = re.sub(r"\s+0x[0-9a-f]+$", "", value, flags=re.I)
    return value.strip() or "(unlabeled)"


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def analyze_gpu_intervals(path: Path, target_pid: int) -> dict[str, Any]:
    definitions: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag != "row":
            continue
        for descendant in element.iter():
            identifier = descendant.get("id")
            if identifier is not None:
                definitions[identifier] = {
                    "tag": descendant.tag,
                    "raw": (descendant.text or "").strip() or None,
                    "fmt": descendant.get("fmt"),
                }
        cells = [_cell_value(cell, definitions) for cell in element]
        if len(cells) < 18:
            element.clear()
            continue
        process = str(cells[10].get("fmt") or "")
        if not process.endswith(f"({target_pid})"):
            element.clear()
            continue
        start_ns = _number(cells[0])
        duration_ns = _number(cells[1])
        if start_ns is None or duration_ns is None:
            element.clear()
            continue
        rows.append(
            {
                "start_ns": start_ns,
                "duration_ns": duration_ns,
                "channel": cells[2].get("fmt") or cells[2].get("raw") or "unknown",
                "depth": _number(cells[5]),
                "label": cells[6].get("fmt") or cells[6].get("raw") or "",
                "command_buffer_id": cells[15].get("fmt") or cells[15].get("raw"),
                "encoder_id": cells[16].get("fmt") or cells[16].get("raw"),
                "submission_id": cells[17].get("fmt") or cells[17].get("raw"),
            }
        )
        element.clear()

    if not rows:
        raise ValueError(f"no Metal GPU intervals found for PID {target_pid}")

    durations = [row["duration_ns"] for row in rows]
    channel_counts: Counter[str] = Counter()
    channel_duration: defaultdict[str, int] = defaultdict(int)
    label_counts: Counter[str] = Counter()
    label_duration: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        channel_counts[row["channel"]] += 1
        channel_duration[row["channel"]] += row["duration_ns"]
        group = _label_group(row["label"])
        label_counts[group] += 1
        label_duration[group] += row["duration_ns"]

    top_dispatches = sorted(rows, key=lambda row: row["duration_ns"], reverse=True)[:25]
    start_ns = min(row["start_ns"] for row in rows)
    end_ns = max(row["start_ns"] + row["duration_ns"] for row in rows)
    return {
        "schema": "ltx.metal-hotspot-report.v1",
        "source_export": str(path.resolve()),
        "target_pid": target_pid,
        "interval_count": len(rows),
        "trace_window_seconds": (end_ns - start_ns) / 1e9,
        "sum_interval_seconds": sum(durations) / 1e9,
        "duration_ms": {
            "mean": statistics.fmean(durations) / 1e6,
            "median": statistics.median(durations) / 1e6,
            "p95": (_percentile(durations, 0.95) or 0) / 1e6,
            "p99": (_percentile(durations, 0.99) or 0) / 1e6,
            "max": max(durations) / 1e6,
        },
        "channels": [
            {
                "channel": channel,
                "count": channel_counts[channel],
                "sum_interval_seconds": channel_duration[channel] / 1e9,
            }
            for channel in sorted(channel_counts, key=channel_duration.get, reverse=True)
        ],
        "label_groups": [
            {
                "label": label,
                "count": label_counts[label],
                "sum_interval_seconds": label_duration[label] / 1e9,
            }
            for label in sorted(label_counts, key=label_duration.get, reverse=True)[:20]
        ],
        "top_dispatches": [
            {
                **row,
                "start_seconds": row["start_ns"] / 1e9,
                "duration_ms": row["duration_ns"] / 1e6,
            }
            for row in top_dispatches
        ],
        "limitations": [
            "Metal System Trace recorded Shader Timeline: Disabled, so kernel symbols and per-shader counters are unavailable.",
            "GPU intervals can be nested; summed interval duration is not GPU wall time or utilization.",
            "Rows are filtered to the target child PID so unrelated system Metal activity is excluded.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_gpu_intervals(args.export, args.pid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
