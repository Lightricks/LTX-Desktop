"""Per-job MLX profile ingestion tests; no model or Metal work."""

from __future__ import annotations

import json
from pathlib import Path

from services.fast_video_pipeline.mlx_profile import (
    begin_mlx_profile,
    finish_mlx_profile,
    get_mlx_profile_snapshot,
    read_mlx_profile,
    reset_mlx_profile_for_tests,
)


def _write_profile(path: Path) -> None:
    records = [
        {
            "schema_version": 2,
            "event": "run_start",
            "timestamp_unix_seconds": 1000.0,
            "mlx_active_gb": 1.0,
            "mlx_peak_gb": 2.0,
            "mlx_cache_gb": 0.5,
            "metadata": {
                "runtime_version": "0.14.20.dev1",
                "runtime_commit": "3171bac4ba901c0237faea2678c34034b37abc2a",
                "device_name": "Apple Test",
                "prompt": "must not be exposed",
            },
        },
        {
            "schema_version": 2,
            "event": "phase_end",
            "phase": "Decoding video + audio + muxing",
            "timestamp_unix_seconds": 1001.0,
            "mlx_active_gb": 0.25,
            "mlx_peak_gb": 5.0,
            "mlx_cache_gb": 0.125,
        },
        {
            "schema_version": 2,
            "event": "run_end",
            "timestamp_unix_seconds": 1002.0,
            "mlx_active_gb": 0.1,
            "mlx_peak_gb": 5.0,
            "mlx_cache_gb": 0.05,
            "observed_peak_mlx_gb": 6.0,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def test_terminal_profile_retains_allocator_peak_phase_and_safe_identity(tmp_path: Path) -> None:
    reset_mlx_profile_for_tests()
    profile = tmp_path / "job.jsonl"
    begin_mlx_profile(profile)
    _write_profile(profile)
    finish_mlx_profile(profile, "success")

    snapshot = get_mlx_profile_snapshot()
    assert snapshot is not None
    assert snapshot.status == "success"
    assert snapshot.active_mib == 102
    assert snapshot.cache_mib == 51
    assert snapshot.peak_mib == 6144
    assert snapshot.phase == "Decoding video + audio + muxing"
    assert snapshot.sampled_at is not None
    assert snapshot.runtime_identity == {
        "runtime_version": "0.14.20.dev1",
        "runtime_commit": "3171bac4ba901c0237faea2678c34034b37abc2a",
        "device_name": "Apple Test",
    }
    reset_mlx_profile_for_tests()


def test_incomplete_tail_and_nonfinite_values_cannot_break_telemetry(tmp_path: Path) -> None:
    profile = tmp_path / "partial.jsonl"
    profile.write_text(
        '{"event":"run_start","timestamp_unix_seconds":Infinity,'
        '"mlx_active_gb":Infinity,"mlx_peak_gb":2.0}\n'
        '{"event":"phase_end","phase":"diffusion",',
        encoding="utf-8",
    )

    snapshot = read_mlx_profile(profile)
    assert snapshot is not None
    assert snapshot.status == "running"
    assert snapshot.sampled_at is None
    assert snapshot.active_mib is None
    assert snapshot.peak_mib == 2048


def test_cancelled_job_keeps_synthetic_terminal_snapshot_when_child_cannot_flush(tmp_path: Path) -> None:
    reset_mlx_profile_for_tests()
    profile = tmp_path / "killed.jsonl"
    begin_mlx_profile(profile)
    finish_mlx_profile(profile, "cancelled")

    snapshot = get_mlx_profile_snapshot()
    assert snapshot is not None
    assert snapshot.status == "cancelled"
    assert snapshot.profile_path == str(profile)
    assert snapshot.peak_mib is None
    reset_mlx_profile_for_tests()
