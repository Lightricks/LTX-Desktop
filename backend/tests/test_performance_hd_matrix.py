from __future__ import annotations

import ctypes
import fcntl
import json
from pathlib import Path

from performance_runner import analyze_metal_trace
from performance_runner import hd_matrix as bench
from performance_runner import lease_interop


def test_rusage_info_v2_buffer_matches_darwin_abi() -> None:
    assert ctypes.sizeof(bench._RusageInfoV2) >= 160


def test_matrix_covers_multiple_resolutions_durations_and_i2v() -> None:
    cases = bench.default_matrix()
    assert {case.resolution for case in cases} == {"540p", "720p", "1080p"}
    assert {case.duration for case in cases} >= {5, 8}
    assert any(case.image_conditioned for case in cases)


def test_distilled_payload_has_no_fake_teacache_or_tiling_switch() -> None:
    payload = bench.default_matrix()[0].payload(None)
    assert not any("tea" in key.lower() for key in payload)
    assert not any("tile" in key.lower() for key in payload)


def test_shared_flock_treats_payload_as_untrusted_diagnostics(tmp_path: Path) -> None:
    lock_path = tmp_path / "local-metal.lock"
    lock_path.write_text(json.dumps({"schema": bench.LOCK_SCHEMA, "pid": 999999}), encoding="utf-8")
    assert bench.probe_metal_lock(lock_path)["observed"] == "acquired"
    with open(lock_path, "a+", encoding="utf-8") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert bench.probe_metal_lock(lock_path)["observed"] == "contended"
        fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
    assert bench.probe_metal_lock(lock_path)["observed"] == "acquired"


def test_strict_validator_hard_fails_missing_authoritative_fields() -> None:
    failures = bench.strict_failures({})
    assert "missing authoritative worker physical footprint" in failures
    assert "missing explicit cleanup/post-cleanup telemetry" in failures
    assert "shared Metal lease not held during local generation" in failures


def test_interop_owner_normalizes_live_and_release_diagnostics() -> None:
    live = {"holder_payload": {"schema": bench.LOCK_SCHEMA, "product": "ai-studio", "job_id": "one"}}
    released = {"holder_payload": {"event": "local_metal_lock_released", "owner": live["holder_payload"]}}
    assert lease_interop.owner_record(live)["job_id"] == "one"
    assert lease_interop.owner_record(released)["job_id"] == "one"
    assert lease_interop.canonical_product("ai-studio") == "AI Studio"
    assert lease_interop.canonical_product("LTX Desktop") == "LTX Desktop"


def test_interop_timeline_requires_ordered_owners_cpu_progress_and_release() -> None:
    events = [
        {"owner_product": "LTX Desktop", "electron_request_active": True},
        {"owner_product": "LTX Desktop", "electron_request_active": False},
        {"owner_product": "AI Studio", "electron_request_active": False},
    ]
    assert lease_interop.validate_timeline(events, {"observed": "acquired"}, [{"ok": True}]) == []
    bad = lease_interop.validate_timeline(
        [{"owner_product": "AI Studio", "electron_request_active": True}],
        {"observed": "contended"},
        [{"ok": False}],
    )
    assert "AI Studio held Metal while the Electron request was active" in bad
    assert "production flock remained held after both harness jobs were terminal" in bad


def test_strict_validator_accepts_complete_torch_evidence() -> None:
    row = {
        "runtime_summary": {"peak_rss_gib": 1, "peak_physical_footprint_gib": 2},
        "runtime_policy": {"auto_fast_video_engine": "torch"},
        "progress_phases": ["inference"], "cleanup_evidence": [{"status": "cleanup"}],
        "lease": {"running_probe": {"observed": "contended"}, "terminal_probe": {"observed": "acquired"}},
        "dimensions": {
            "requested": {"resolution": "720p"},
            "resolved": {"width": 1280, "height": 704},
            "actual": {"width": 1280, "height": 704},
        },
        "hashes": {"recipe_sha256": "a", "prompt_sha256": "b", "source_sha256": "c", "repo_head": "d"},
    }
    assert bench.strict_failures(row) == []

    row["dimensions"]["actual"]["height"] = 512
    assert "resolved height 704 does not match actual height 512" in bench.strict_failures(row)


def test_metal_trace_analyzer_resolves_nested_process_references(tmp_path: Path) -> None:
    export = tmp_path / "gpu_intervals.xml"
    export.write_text(
        """<trace-query-result><node>
        <row>
          <start-time id="1">100</start-time><duration id="2">5000</duration>
          <gpu-channel-name id="3" fmt="Compute">Compute</gpu-channel-name><sentinel/>
          <duration id="4">0</duration><metal-nesting-level id="5">0</metal-nesting-level>
          <formatted-label id="6" fmt="Command Buffer 0:Compute Command 0     ( python3.11 (42) )  0xabc">
            <process id="11" fmt="python3.11 (42)"><pid>42</pid></process>
          </formatted-label>
          <gpu-state/><connection-uuid64/><render-buffer-depth/><process ref="11"/>
          <metal-device-name/><metal-object-label/><formatted-label/><size-in-bytes/>
          <metal-command-buffer-id fmt="0x1"/><metal-command-buffer-id fmt="0x2"/><uint64>3</uint64>
        </row>
        </node></trace-query-result>""",
        encoding="utf-8",
    )
    report = analyze_metal_trace.analyze_gpu_intervals(export, 42)
    assert report["interval_count"] == 1
    assert report["channels"] == [{"channel": "Compute", "count": 1, "sum_interval_seconds": 0.000005}]
    assert report["top_dispatches"][0]["command_buffer_id"] == "0x1"
