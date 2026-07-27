"""Crash-resilient ingestion for per-job MLX JSONL profiles."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

MLXProfileStatus = Literal["running", "success", "error", "cancelled"]
_GIB_TO_MIB = 1024
_PROFILE_IDENTITY_KEYS = {
    "runtime_commit",
    "runtime_dirty",
    "runtime_version",
    "core_version",
    "mlx_version",
    "mlx_metal_version",
    "device_name",
    "device_architecture",
    "device_memory_bytes",
    "device_recommended_working_set_bytes",
    "runtime_family",
    "device_family",
}


@dataclass(frozen=True)
class MLXProfileSnapshot:
    profile_path: str
    status: MLXProfileStatus
    sampled_at: str | None = None
    phase: str | None = None
    active_mib: int | None = None
    cache_mib: int | None = None
    peak_mib: int | None = None
    runtime_identity: dict[str, object] | None = None


_profile_lock = threading.Lock()
_active_profile_path: Path | None = None
_last_profile_snapshot: MLXProfileSnapshot | None = None


def allocate_mlx_profile_path(output_path: str) -> Path:
    """Allocate a durable, prompt-free profile path beside generated outputs."""
    output = Path(output_path).expanduser()
    profile_dir = output.parent / ".mlx-profiles"
    profile_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(profile_dir, 0o700)
    except OSError:
        pass
    return profile_dir / f"{output.stem}-{uuid.uuid4().hex}.jsonl"


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mib(value: object) -> int | None:
    number = _finite_number(value)
    return round(number * _GIB_TO_MIB) if number is not None else None


def _sampled_at(value: object) -> str | None:
    number = _finite_number(value)
    if number is None:
        return None
    try:
        return datetime.fromtimestamp(number, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def read_mlx_profile(path: Path) -> MLXProfileSnapshot | None:
    """Read all complete JSONL records; tolerate a concurrently-written tail."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    status: MLXProfileStatus = "running"
    sampled_at = None
    phase = None
    active_mib = None
    cache_mib = None
    peak_mib = None
    runtime_identity: dict[str, object] | None = None
    records = 0
    for line in lines:
        try:
            raw_record: object = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(raw_record, dict):
            continue
        record = cast(dict[str, object], raw_record)
        records += 1
        sampled_at = _sampled_at(record.get("timestamp_unix_seconds")) or sampled_at
        event = record.get("event")
        if event == "run_start":
            status = "running"
            metadata_value = record.get("metadata")
            if isinstance(metadata_value, dict):
                metadata = cast(dict[str, object], metadata_value)
                runtime_identity = {
                    key: value
                    for key, value in metadata.items()
                    if key in _PROFILE_IDENTITY_KEYS
                    and isinstance(value, (str, int, float, bool))
                }
        elif event == "run_end":
            status = "success"
        elif event == "run_error":
            status = "error"
        phase_value = record.get("phase")
        if isinstance(phase_value, str):
            phase = phase_value

        active_candidate = _mib(record.get("mlx_active_gb"))
        cache_candidate = _mib(record.get("mlx_cache_gb"))
        if active_candidate is not None:
            active_mib = active_candidate
        if cache_candidate is not None:
            cache_mib = cache_candidate
        candidates = (
            _mib(record.get("mlx_peak_gb")),
            _mib(record.get("observed_peak_mlx_gb")),
        )
        for candidate in candidates:
            if candidate is not None:
                peak_mib = max(peak_mib or 0, candidate)

    if records == 0:
        return None
    return MLXProfileSnapshot(
        profile_path=str(path),
        status=status,
        sampled_at=sampled_at,
        phase=phase,
        active_mib=active_mib,
        cache_mib=cache_mib,
        peak_mib=peak_mib,
        runtime_identity=runtime_identity,
    )


def begin_mlx_profile(path: Path) -> None:
    global _active_profile_path, _last_profile_snapshot
    with _profile_lock:
        _active_profile_path = path
        _last_profile_snapshot = MLXProfileSnapshot(
            profile_path=str(path),
            status="running",
        )


def finish_mlx_profile(path: Path, status: MLXProfileStatus) -> None:
    global _active_profile_path, _last_profile_snapshot
    snapshot = read_mlx_profile(path)
    if snapshot is None:
        snapshot = MLXProfileSnapshot(profile_path=str(path), status=status)
    else:
        snapshot = replace(snapshot, status=status)
    with _profile_lock:
        if _active_profile_path == path:
            _active_profile_path = None
        _last_profile_snapshot = snapshot


def get_mlx_profile_snapshot() -> MLXProfileSnapshot | None:
    """Return the live flushed profile, or the last terminal job snapshot."""
    global _last_profile_snapshot
    with _profile_lock:
        active_path = _active_profile_path
        fallback = _last_profile_snapshot
    if active_path is None:
        return fallback
    live = read_mlx_profile(active_path)
    if live is None:
        return fallback
    with _profile_lock:
        if _active_profile_path == active_path:
            _last_profile_snapshot = live
    return live


def reset_mlx_profile_for_tests() -> None:
    global _active_profile_path, _last_profile_snapshot
    with _profile_lock:
        _active_profile_path = None
        _last_profile_snapshot = None
