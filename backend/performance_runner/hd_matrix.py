#!/usr/bin/env python3
"""HD qualification through the production LTX Desktop FastAPI entrypoint.

The default action prints a no-render plan. Use ``--execute`` only after the
backend/app is running and local Metal qualification is authorized.
"""
from __future__ import annotations

import argparse
import array
import concurrent.futures
import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import perf_config
except ImportError:  # direct `python hd_matrix.py` execution
    import perf_config

LOCK_PATH = Path.home() / "Library/Application Support/LTX Shared/local-metal.lock"
LOCK_SCHEMA = "ltx.local-metal-lock.v1"
PROMPT = "A slow, steady lateral camera move reveals the scene with natural subject motion and physically consistent lighting."
TERMINAL = {"complete", "cancelled", "error"}


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    resolution: str
    duration: int
    fps: int = 24
    aspect_ratio: str = "16:9"
    image_conditioned: bool = False

    def payload(self, image_path: str | None) -> dict[str, Any]:
        return {
            "prompt": PROMPT,
            "resolution": self.resolution,
            "model": "fast",
            "cameraMotion": "none",
            "negativePrompt": "",
            "duration": self.duration,
            "fps": self.fps,
            "audio": False,
            "imagePath": image_path if self.image_conditioned else None,
            "audioPath": None,
            "aspectRatio": self.aspect_ratio,
            "seed": 424242,
            "loras": [],
        }


def default_matrix() -> list[MatrixCase]:
    return [
        MatrixCase("fast_t2v_540p_5s", "540p", 5),
        MatrixCase("fast_t2v_720p_5s", "720p", 5),
        MatrixCase("fast_t2v_720p_8s", "720p", 8),
        MatrixCase("fast_t2v_1080p_5s", "1080p", 5),
        MatrixCase("fast_i2v_720p_5s", "720p", 5, image_conditioned=True),
    ]


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_sha(path: str | Path | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_json(method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 1800) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        perf_config.BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **perf_config._auth_headers()},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
    return json.loads(raw) if raw else None


def probe_metal_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Any = None
    observed = "contended"
    with open(path, "a+", encoding="utf-8") as stream:
        stream.seek(0)
        raw = stream.read().strip()
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"unparsed": raw[:1000]}
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            observed = "acquired"
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            observed = "contended"
    return {
        "schema": "ltx.benchmark.lock-probe.v1",
        "lease_path": str(path),
        "observed": observed,
        "holder_payload": payload,
        "holder_payload_schema_valid": isinstance(payload, dict) and payload.get("schema") == LOCK_SCHEMA,
        "timestamp": time.time(),
        "monotonic_ns": time.monotonic_ns(),
    }


def _ps_rows() -> list[tuple[int, int, int, float]]:
    try:
        text = subprocess.check_output(["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "rss=", "-o", "%cpu="], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])))
            except ValueError:
                pass
    return rows


def descendant_pids(root_pid: int, rows: list[tuple[int, int, int, float]]) -> set[int]:
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _rss, _cpu in rows:
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


def physical_footprint_bytes(pid: int) -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        fn = libproc.proc_pid_rusage
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        info = _RusageInfoV2()
        if fn(pid, 2, ctypes.byref(info)) == 0:
            return int(info.ri_phys_footprint)
    except (OSError, AttributeError):
        pass
    return None


def gpu_utilization_percent() -> float | None:
    if sys.platform != "darwin":
        return None
    try:
        text = subprocess.check_output(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    values: list[float] = []
    for pattern in (r'"Device Utilization %"\s*=\s*(\d+)', r'"GPU Activity\(%\)"\s*=\s*(\d+)'):
        values.extend(float(raw) for raw in re.findall(pattern, text))
    return max(values) if values else None


def process_sample(root_pid: int) -> dict[str, Any]:
    rows = _ps_rows()
    pids = descendant_pids(root_pid, rows)
    chosen = [row for row in rows if row[0] in pids]
    footprints = [value for value in (physical_footprint_bytes(pid) for pid in pids) if value is not None]
    return {
        "timestamp": time.time(),
        "root_pid": root_pid,
        "pids": sorted(pids),
        "process_count": len(chosen),
        "rss_gib": sum(row[2] for row in chosen) * 1024 / 1024**3,
        "physical_footprint_gib": sum(footprints) / 1024**3 if footprints else None,
        "cpu_percent": sum(row[3] for row in chosen),
        "gpu_utilization_percent": gpu_utilization_percent(),
    }


def ffprobe(path: str) -> dict[str, Any]:
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], text=True, timeout=30)
    data = json.loads(raw)
    streams = data.get("streams") or []
    video = next((row for row in streams if row.get("codec_type") == "video"), {})
    audio = next((row for row in streams if row.get("codec_type") == "audio"), None)
    duration = video.get("duration") or (data.get("format") or {}).get("duration")
    return {
        "width": video.get("width"), "height": video.get("height"),
        "frames": int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None,
        "duration_seconds": float(duration) if duration is not None else None,
        "has_audio": audio is not None,
        "video_codec": video.get("codec_name"), "audio_codec": audio.get("codec_name") if audio else None,
    }


def _ffmpeg_metric(reference: str, candidate: str, filter_name: str, pattern: str) -> float | None:
    proc = subprocess.run(["ffmpeg", "-v", "info", "-i", reference, "-i", candidate, "-lavfi", filter_name, "-f", "null", "-"], capture_output=True, text=True, timeout=1800)
    matches = re.findall(pattern, proc.stderr)
    return float(matches[-1]) if matches else None


def _audio_pcm(path: str) -> array.array:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0?", "-ac", "1", "-ar", "48000", "-f", "f32le", "-"], capture_output=True, timeout=600)
    values = array.array("f")
    values.frombytes(proc.stdout)
    return values


def quality_metrics(reference: str, candidate: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "psnr_db": _ffmpeg_metric(reference, candidate, "psnr", r"average:([0-9.]+)"),
        "ssim": _ffmpeg_metric(reference, candidate, "ssim", r"All:([0-9.]+)"),
        "audio_mae": None, "audio_snr_db": None,
    }
    a, b = _audio_pcm(reference), _audio_pcm(candidate)
    count = min(len(a), len(b))
    if count:
        signal = sum(float(a[i]) ** 2 for i in range(count)) / count
        noise = sum((float(a[i]) - float(b[i])) ** 2 for i in range(count)) / count
        result["audio_mae"] = sum(abs(float(a[i]) - float(b[i])) for i in range(count)) / count
        result["audio_snr_db"] = math.inf if noise == 0 else 10.0 * math.log10(signal / noise) if signal else None
    return result


def machine_details(repo_root: Path) -> dict[str, Any]:
    def text(argv: list[str]) -> str:
        try:
            return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(), "python": sys.version,
        "sw_vers": text(["sw_vers"]), "hardware": text(["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"]),
        "memsize": text(["sysctl", "-n", "hw.memsize"]), "repo_head": text(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "repo_status": text(["git", "-C", str(repo_root), "status", "--short"]), "ffmpeg": text(["ffmpeg", "-version"]).splitlines()[:1],
    }


def strict_failures(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    telemetry = result.get("runtime_summary") or {}
    lease = result.get("lease") or {}
    dimensions = result.get("dimensions") or {}
    hashes = result.get("hashes") or {}
    phases = result.get("progress_phases") or []
    if telemetry.get("peak_rss_gib") is None:
        failures.append("missing worker/process RSS peak")
    if telemetry.get("peak_physical_footprint_gib") is None:
        failures.append("missing authoritative worker physical footprint")
    if result.get("runtime_policy", {}).get("auto_fast_video_engine") == "mlx":
        if telemetry.get("peak_mlx_mib") is None:
            failures.append("missing MLX allocator peak")
        if not telemetry.get("mlx_runtime_identity"):
            failures.append("missing profiled MLX runtime identity")
        if telemetry.get("mlx_profile_status") not in {"success", "cancelled"}:
            failures.append("missing terminal MLX profile status")
    if not phases:
        failures.append("missing generation phase samples")
    if not result.get("cleanup_evidence"):
        failures.append("missing explicit cleanup/post-cleanup telemetry")
    if lease.get("running_probe", {}).get("observed") != "contended":
        failures.append("shared Metal lease not held during local generation")
    if lease.get("terminal_probe", {}).get("observed") != "acquired":
        failures.append("shared Metal lease not released after cleanup")
    for key in ("requested", "resolved", "actual"):
        if not dimensions.get(key):
            failures.append(f"missing {key} geometry")
    resolved = dimensions.get("resolved") or {}
    actual = dimensions.get("actual") or {}
    for axis in ("width", "height"):
        if resolved.get(axis) is None:
            failures.append(f"missing resolved {axis}")
        elif actual.get(axis) is None:
            failures.append(f"missing actual {axis}")
        elif resolved[axis] != actual[axis]:
            failures.append(
                f"resolved {axis} {resolved[axis]} does not match actual {axis} {actual[axis]}"
            )
    for key in ("recipe_sha256", "prompt_sha256", "source_sha256", "repo_head"):
        if not hashes.get(key):
            failures.append(f"missing {key}")
    return failures


def _peak(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def run_case(case: MatrixCase, image_path: str | None, artifact_dir: Path, expect_mode: str, poll_seconds: float, cancel_during_run: bool = False) -> dict[str, Any]:
    progress_before = http_json("GET", perf_config.STATUS_PATH)
    if progress_before.get("status") in {"running"}:
        raise RuntimeError(f"refusing to overlap existing generation: {progress_before}")
    policy = http_json("GET", "/api/runtime-policy")
    if expect_mode != "any" and policy.get("execution_mode") != expect_mode:
        raise RuntimeError(f"runtime mode {policy.get('execution_mode')} does not match --expect-mode={expect_mode}")
    initial_probe = probe_metal_lock()
    if initial_probe["observed"] != "acquired":
        raise RuntimeError(f"shared Metal lease is already contended: {initial_probe}")

    payload = case.payload(image_path)
    submitted_at = time.time()
    progress_samples: list[dict[str, Any]] = []
    runtime_samples: list[dict[str, Any]] = []
    process_samples: list[dict[str, Any]] = []
    running_probe: dict[str, Any] | None = None
    cancel_sent = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(http_json, "POST", perf_config.GENERATE_PATH, payload, 1800)
        while not future.done():
            progress = http_json("GET", perf_config.STATUS_PATH)
            runtime = http_json("GET", "/api/runtime-telemetry")
            progress_samples.append({"timestamp": time.time(), **progress})
            runtime_samples.append({"timestamp": time.time(), **runtime})
            if runtime.get("local_metal_lease_status") == "held":
                if running_probe is None:
                    running_probe = probe_metal_lock()
                owner = (running_probe or {}).get("holder_payload") or runtime.get("local_metal_lease_owner") or {}
                if owner.get("pid"):
                    process_samples.append(process_sample(int(owner["pid"])))
                if cancel_during_run and not cancel_sent and progress.get("phase") == "inference":
                    http_json("POST", "/api/generate/cancel", {})
                    cancel_sent = True
            time.sleep(poll_seconds)
        response = future.result()

    # The isolated MLX child publishes its terminal allocator peak and identity
    # after process exit. Capture one post-job API sample so strict evidence does
    # not depend on the polling interval racing the final flushed JSONL event.
    terminal_runtime = http_json("GET", "/api/runtime-telemetry")
    runtime_samples.append({"timestamp": time.time(), **terminal_runtime})
    terminal_probe = probe_metal_lock()
    output_path = str((response or {}).get("video_path") or "")
    actual = ffprobe(output_path) if output_path and Path(output_path).is_file() else {}
    resolved_width = response.get("resolved_width")
    resolved_height = response.get("resolved_height")
    resolved_frames = ((case.duration * case.fps) // 8) * 8 + 1
    details = machine_details(Path(__file__).resolve().parents[2])
    phases = sorted({str(row.get("phase")) for row in progress_samples if row.get("phase")})
    cleanup = [row for row in runtime_samples if row.get("local_metal_lease_status") == "held" and row.get("active_pipeline") is None]
    result: dict[str, Any] = {
        "schema": "ltx.hd-benchmark.result.v1", "product": "ltx_desktop_electron", "case": asdict(case),
        "response": response, "submitted_at": submitted_at, "finished_at": time.time(), "wall_seconds": time.time() - submitted_at,
        "output_path": output_path or None, "runtime_policy": policy, "progress_phases": phases, "cleanup_evidence": cleanup,
        "dimensions": {
            "requested": {"resolution": case.resolution, "duration": case.duration, "fps": case.fps, "aspect_ratio": case.aspect_ratio},
            "resolved": {"width": resolved_width, "height": resolved_height, "frames": resolved_frames, "execution_mode": policy.get("execution_mode")},
            "actual": actual,
        },
        "hashes": {
            "recipe_sha256": canonical_sha(payload), "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "source_sha256": file_sha(image_path) if case.image_conditioned else canonical_sha({"source": "text_only"}),
            "output_sha256": file_sha(output_path), "repo_head": details.get("repo_head"), "runtime_policy_sha256": canonical_sha(policy),
        },
        "lease": {"initial_probe": initial_probe, "running_probe": running_probe or {}, "terminal_probe": terminal_probe},
        "runtime_summary": {
            "samples": len(runtime_samples), "peak_rss_gib": max((_row.get("process_rss_mib", 0) for _row in runtime_samples), default=0) / 1024 if runtime_samples else None,
            "peak_mlx_mib": max((_row.get("mlx_peak_mib") for _row in runtime_samples if _row.get("mlx_peak_mib") is not None), default=None),
            "peak_mps_allocated_mib": max((_row.get("mps_allocated_mib") for _row in runtime_samples if _row.get("mps_allocated_mib") is not None), default=None),
            "peak_mps_driver_mib": max((_row.get("mps_driver_mib") for _row in runtime_samples if _row.get("mps_driver_mib") is not None), default=None),
            "peak_process_tree_rss_gib": _peak(process_samples, "rss_gib"), "peak_physical_footprint_gib": _peak(process_samples, "physical_footprint_gib"),
            "peak_cpu_percent": _peak(process_samples, "cpu_percent"), "peak_gpu_utilization_percent": _peak(process_samples, "gpu_utilization_percent"),
            "mlx_profile_status": terminal_runtime.get("mlx_profile_status"),
            "mlx_profile_phase": terminal_runtime.get("mlx_profile_phase"),
            "mlx_profile_path": terminal_runtime.get("mlx_profile_path"),
            "mlx_profile_sampled_at": terminal_runtime.get("mlx_profile_sampled_at"),
            "mlx_runtime_identity": terminal_runtime.get("mlx_runtime_identity"),
        },
        "machine": details, "cancel_sent": cancel_sent,
    }
    result["strict_failures"] = strict_failures(result)
    result["strict_status"] = "PASS" if not result["strict_failures"] and response.get("status") == "complete" else "FAIL"
    case_dir = artifact_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (case_dir / "progress_samples.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in progress_samples), encoding="utf-8")
    (case_dir / "runtime_samples.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in runtime_samples), encoding="utf-8")
    (case_dir / "process_samples.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in process_samples), encoding="utf-8")
    (case_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def add_reference_quality(results: list[dict[str, Any]], reference_manifest: Path | None, artifact_dir: Path) -> None:
    if reference_manifest is None:
        return
    prior = json.loads(reference_manifest.read_text(encoding="utf-8"))
    prior_rows = prior if isinstance(prior, list) else prior.get("results", [])
    by_case = {row.get("case", {}).get("case_id"): row for row in prior_rows}
    blinded, answer = [], []
    for row in results:
        case_id = row["case"]["case_id"]
        ref = by_case.get(case_id) or {}
        reference, candidate = ref.get("output_path"), row.get("output_path")
        if not reference or not candidate or not Path(reference).is_file() or not Path(candidate).is_file():
            continue
        row["quality_vs_reference"] = quality_metrics(reference, candidate)
        if canonical_sha({"case": case_id, "candidate": candidate})[-1] in "02468ace":
            pair = [("reference", reference), ("candidate", candidate)]
        else:
            pair = [("candidate", candidate), ("reference", reference)]
        blinded.append({"pair_id": case_id, "A": pair[0][1], "B": pair[1][1], "questions": ["Which has better motion coherence?", "Which preserves detail?", "Which audio is cleaner?", "Any seams or freezes?"]})
        answer.append({"pair_id": case_id, "A": pair[0][0], "B": pair[1][0]})
    (artifact_dir / "blinded_review.json").write_text(json.dumps(blinded, indent=2), encoding="utf-8")
    (artifact_dir / "blinded_review_key.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")


def verify_kernel_crash_release(artifact_dir: Path) -> dict[str, Any]:
    """Prove kernel-owned flock release after an owner is SIGKILLed, without crashing either app."""
    helper = "import fcntl,os,sys,time; f=open(sys.argv[1],'a+'); fcntl.flock(f.fileno(),fcntl.LOCK_EX); print(os.getpid(),flush=True); time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-c", helper, str(LOCK_PATH)], stdout=subprocess.PIPE, text=True)
    assert process.stdout is not None
    process.stdout.readline()
    while_alive = probe_metal_lock()
    process.kill()
    process.wait(timeout=10)
    after_sigkill = probe_metal_lock()
    result = {
        "scope": "kernel flock owner SIGKILL simulation; neither product server was crashed",
        "while_owner_alive": while_alive,
        "after_sigkill": after_sigkill,
        "pass": while_alive["observed"] == "contended" and after_sigkill["observed"] == "acquired",
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "crash_release.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(results: list[dict[str, Any]], artifact_dir: Path, command: str) -> None:
    lines = ["# LTX Desktop Electron HD benchmark", "", f"Command: `{command}`", "", "| case | mode | status | wall s | footprint GiB | MLX MiB | SHA-256 | strict gaps |", "|---|---|---:|---:|---:|---:|---|---|"]
    for row in results:
        summary = row.get("runtime_summary") or {}
        lines.append(f"| {row['case']['case_id']} | {row.get('runtime_policy', {}).get('execution_mode')} | {row.get('strict_status')} | {row.get('wall_seconds', 0):.2f} | {summary.get('peak_physical_footprint_gib')} | {summary.get('peak_mlx_mib')} | {(row.get('hashes') or {}).get('output_sha256') or '—'} | {'; '.join(row.get('strict_failures') or []) or '—'} |")
    (artifact_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (artifact_dir / "results.json").write_text(json.dumps({"schema": "ltx.hd-benchmark.manifest.v1", "results": results}, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--runs-per-case", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--expect-mode", choices=("any", "eager", "low_ram"), default="any")
    parser.add_argument("--image", type=Path, default=Path(__file__).resolve().parent / "test_assets/reference_image.png")
    parser.add_argument("--artifacts", type=Path, default=perf_config.RUNS_DIR / f"hd_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--verify-cancel-release", action="store_true", help="cancel only the harness's own first selected run during inference")
    parser.add_argument("--verify-crash-release", action="store_true", help="no-model SIGKILL helper proving kernel flock release")
    args = parser.parse_args()
    matrix = default_matrix()
    if args.case:
        wanted = set(args.case)
        matrix = [case for case in matrix if case.case_id in wanted]
        missing = wanted - {case.case_id for case in matrix}
        if missing:
            parser.error(f"unknown cases: {sorted(missing)}")
    plan = {
        "schema": "ltx.hd-benchmark.plan.v1", "product": "ltx_desktop_electron",
        "production_entrypoint": f"POST {perf_config.GENERATE_PATH}", "production_lease": str(LOCK_PATH),
        "expect_mode": args.expect_mode, "runs_per_case": args.runs_per_case, "cases": [asdict(case) for case in matrix],
        "notes": [
            "Run once with --expect-mode=eager and once with --expect-mode=low_ram; runtime policy, not a request-only toggle, owns the app mode.",
            "Distilled Fast T2V/I2V cannot use TeaCache and this harness never presents it as a switch.",
            "Explicit modality-tiling candidates are qualified in the AI Studio MLX HQ matrix; Electron Fast exposes only its production automatic policy.",
            "The production flock serializes local heavy work; cloud/API/CPU-only work remains outside it.",
        ],
    }
    if args.verify_crash_release and not args.execute:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        result = verify_kernel_crash_release(args.artifacts)
        print(json.dumps(result, indent=2))
        return 0 if result["pass"] else 1
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if any(case.image_conditioned for case in matrix) and not args.image.is_file():
        parser.error(f"image fixture missing: {args.image}")
    perf_config.wait_for_backend()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    results: list[dict[str, Any]] = []
    cancel_pending = args.verify_cancel_release
    for case in matrix:
        for run_index in range(args.runs_per_case):
            run_case_value = case if args.runs_per_case == 1 else MatrixCase(f"{case.case_id}_run{run_index + 1}", case.resolution, case.duration, case.fps, case.aspect_ratio, case.image_conditioned)
            print(f"[qa] {run_case_value.case_id}", flush=True)
            results.append(run_case(run_case_value, str(args.image.resolve()), args.artifacts, args.expect_mode, args.poll_seconds, cancel_during_run=cancel_pending))
            cancel_pending = False
    add_reference_quality(results, args.reference_manifest, args.artifacts)
    if args.verify_crash_release:
        verify_kernel_crash_release(args.artifacts)
    write_summary(results, args.artifacts, " ".join(sys.argv))
    print(args.artifacts)
    return 1 if any(row.get("strict_status") != "PASS" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
