#!/usr/bin/env python3
"""Cross-product local-Metal serialization proof for LTX Desktop + AI Studio.

Default is a no-render plan. The explicit execution queues exactly one harness-
owned job in each product and never cancels or mutates pre-existing work.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from . import hd_matrix
except ImportError:
    import hd_matrix

AI_BASE_URL = "http://127.0.0.1:5577"
AI_LOCAL_TYPES = {"i2v3", "i2v3_id_lora", "i2v3_ingredients", "v2v3", "i2va"}
AI_TERMINAL = {"complete", "error", "cancelled"}


def ai_http(base_url: str, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 30) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method, headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"AI Studio {method} {path} -> HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    return json.loads(raw) if raw else None


def queue_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows: list[dict[str, Any]] = []
        for key in ("running", "queued", "recent", "jobs"):
            rows.extend(queue_rows(payload.get(key)))
        if not rows and payload.get("id"):
            rows.append(payload)
        return rows
    return []


def find_job(payload: Any, job_id: str) -> dict[str, Any] | None:
    return next((row for row in queue_rows(payload) if str(row.get("id")) == job_id), None)


def active_ai_local(payload: Any) -> list[dict[str, Any]]:
    return [row for row in queue_rows(payload) if row.get("phase") in {"queued", "running"} and row.get("type") in AI_LOCAL_TYPES]


def owner_record(probe: dict[str, Any]) -> dict[str, Any]:
    """Extract live owner data or the owner nested in release diagnostics."""
    payload = probe.get("holder_payload")
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("owner")
    if isinstance(nested, dict) and nested.get("schema") == hd_matrix.LOCK_SCHEMA:
        return nested
    return payload


def canonical_product(value: Any) -> str | None:
    normalized = str(value or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    if normalized == "ltxdesktop":
        return "LTX Desktop"
    if normalized == "aistudio":
        return "AI Studio"
    return None


def validate_timeline(events: list[dict[str, Any]], terminal_probe: dict[str, Any], cpu_probes: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    electron_seen = any(event.get("owner_product") == "LTX Desktop" for event in events)
    ai_seen = any(event.get("owner_product") == "AI Studio" for event in events)
    if not electron_seen:
        failures.append("Electron never owned the production flock")
    if not ai_seen:
        failures.append("AI Studio never owned the production flock after Electron")
    first_ai = next((index for index, event in enumerate(events) if event.get("owner_product") == "AI Studio"), None)
    last_electron = max((index for index, event in enumerate(events) if event.get("owner_product") == "LTX Desktop"), default=None)
    if first_ai is not None and last_electron is not None and first_ai <= last_electron:
        failures.append("AI Studio ownership appeared before Electron ownership ended")
    for event in events:
        if event.get("electron_request_active") and event.get("owner_product") == "AI Studio":
            failures.append("AI Studio held Metal while the Electron request was active")
            break
    if not cpu_probes or any(not probe.get("ok") for probe in cpu_probes):
        failures.append("control-plane CPU-only probes did not remain responsive during contention")
    if terminal_probe.get("observed") != "acquired":
        failures.append("production flock remained held after both harness jobs were terminal")
    return failures


def timed_probe(product: str, call) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = call()
        return {"product": product, "ok": True, "elapsed_seconds": time.perf_counter() - started, "response_kind": type(payload).__name__}
    except Exception as exc:
        return {"product": product, "ok": False, "elapsed_seconds": time.perf_counter() - started, "error": str(exc)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    electron_before = hd_matrix.http_json("GET", hd_matrix.perf_config.STATUS_PATH)
    if electron_before.get("status") == "running":
        raise RuntimeError(f"Electron already has active work: {electron_before}")
    ai_queue_before = ai_http(args.ai_base_url, "GET", "/api/queue")
    ai_busy_before = ai_http(args.ai_base_url, "GET", "/api/render-busy")
    active = active_ai_local(ai_queue_before)
    if ai_busy_before.get("busy") or active:
        raise RuntimeError(f"AI Studio already has local work: busy={ai_busy_before}, ids={[row.get('id') for row in active]}")
    if hd_matrix.probe_metal_lock()["observed"] != "acquired":
        raise RuntimeError("production flock is already held before the interop run")

    electron_case = hd_matrix.MatrixCase("interop_electron_540p_5s", "540p", 5)
    electron_payload = electron_case.payload(None)
    events: list[dict[str, Any]] = []
    cpu_probes: list[dict[str, Any]] = []
    ai_job_id = ""
    ai_job: dict[str, Any] = {}
    ai_cancel_sent = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        electron_future = pool.submit(hd_matrix.http_json, "POST", hd_matrix.perf_config.GENERATE_PATH, electron_payload, 1800)
        deadline = time.time() + args.timeout_seconds
        while time.time() < deadline:
            telemetry = hd_matrix.http_json("GET", "/api/runtime-telemetry")
            probe = hd_matrix.probe_metal_lock()
            owner = owner_record(probe)
            owner_product = canonical_product(owner.get("product"))
            events.append({
                "timestamp": time.time(), "stage": "electron_acquire", "electron_request_active": not electron_future.done(),
                "electron_lease_status": telemetry.get("local_metal_lease_status"), "lock_observed": probe.get("observed"),
                "owner_product": owner_product, "owner_product_raw": owner.get("product"), "owner_job_id": owner.get("job_id"),
            })
            if telemetry.get("local_metal_lease_status") == "held" and owner_product == "LTX Desktop":
                break
            time.sleep(args.poll_seconds)
        else:
            raise RuntimeError("Electron did not acquire the shared lease before timeout")

        cpu_probes.extend([
            timed_probe("electron_runtime_policy", lambda: hd_matrix.http_json("GET", "/api/runtime-policy")),
            timed_probe("ai_queue_read", lambda: ai_http(args.ai_base_url, "GET", "/api/queue")),
            timed_probe("ai_performance_read", lambda: ai_http(args.ai_base_url, "GET", "/api/ltx/performance?limit=1")),
        ])
        ai_payload = {
            "type": "i2v3", "prompt": hd_matrix.PROMPT, "source_image": str(args.ai_source_image.resolve()),
            "output_folder": args.ai_output_folder, "source": "qa_benchmark", "user_initiated": False,
            "params": {"width": 768, "height": 512, "frames": 129, "fps": 24, "steps": 8, "distilled": True, "seed": 424242, "memory_mode": "low_ram", "model_variant": "bf16", "tile_frames": 1, "tile_spatial": 1, "tile_overlap": 2, "profile_enabled": True},
        }
        queued = ai_http(args.ai_base_url, "POST", "/api/queue", ai_payload)
        ai_job_id = str((queued.get("job") or {}).get("id") or "")
        if not ai_job_id:
            raise RuntimeError(f"AI Studio queue response has no id: {queued}")

        while time.time() < deadline:
            queue = ai_http(args.ai_base_url, "GET", "/api/queue")
            ai_job = find_job(queue, ai_job_id) or {}
            probe = hd_matrix.probe_metal_lock()
            owner = owner_record(probe)
            owner_product = canonical_product(owner.get("product"))
            electron_active = not electron_future.done()
            events.append({
                "timestamp": time.time(), "stage": "contention", "electron_request_active": electron_active,
                "ai_phase": ai_job.get("phase"), "ai_pid": ai_job.get("pid"), "lock_observed": probe.get("observed"),
                "owner_product": owner_product, "owner_product_raw": owner.get("product"), "owner_job_id": owner.get("job_id"),
            })
            if args.cancel_ai_after_acquire and owner_product == "AI Studio" and ai_job.get("phase") == "running" and not ai_cancel_sent:
                ai_http(args.ai_base_url, "POST", f"/api/queue/{ai_job_id}/kill", {})
                ai_cancel_sent = True
            if electron_future.done() and ai_job.get("phase") in AI_TERMINAL:
                break
            time.sleep(args.poll_seconds)
        else:
            raise RuntimeError("cross-product run did not become terminal before timeout")
        electron_response = electron_future.result()

    terminal_probe = hd_matrix.probe_metal_lock()
    failures = validate_timeline(events, terminal_probe, cpu_probes)
    if args.cancel_ai_after_acquire and not ai_cancel_sent:
        failures.append("AI cancellation was requested but the harness never observed AI lease ownership")
    if args.cancel_ai_after_acquire and ai_job.get("phase") != "cancelled":
        failures.append(f"harness-owned AI job ended {ai_job.get('phase')} instead of cancelled")
    result = {
        "schema": "ltx.local-metal-interop.v1", "electron_response": electron_response,
        "ai_job_id": ai_job_id, "ai_terminal_phase": ai_job.get("phase"), "ai_cancel_sent": ai_cancel_sent,
        "events": events, "cpu_only_control_plane_probes": cpu_probes, "terminal_probe": terminal_probe,
        "failures": failures, "status": "PASS" if not failures else "FAIL",
        "qualification": "Actual local-heavy ownership is verified. CPU-only evidence is bounded control-plane work; no paid remote-provider job is submitted.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ai-base-url", default=AI_BASE_URL)
    parser.add_argument("--ai-source-image", type=Path, default=Path("/Users/paulericksen/Documents/ComfyUI/output/video_frames/clipboard_capture_04.png"))
    parser.add_argument("--ai-output-folder", default="qa/lease_interop")
    parser.add_argument("--artifacts", type=Path, default=hd_matrix.perf_config.RUNS_DIR / f"lease_interop_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--cancel-ai-after-acquire", action="store_true", help="kill only this controller's own AI job after it acquires the lease")
    args = parser.parse_args()
    plan = {
        "schema": "ltx.local-metal-interop.plan.v1", "production_lease": str(hd_matrix.LOCK_PATH),
        "sequence": ["Electron local Fast job acquires", "AI Studio local job queues/blocks", "CPU-only status/performance reads continue", "Electron releases", "AI Studio acquires", "both terminal -> lease free"],
        "remote_scope": "No paid remote job is submitted; product-level bypass tests remain the remote-provider authority.",
        "cancel_ai_after_acquire": args.cancel_ai_after_acquire,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.ai_source_image.is_file():
        parser.error(f"AI source image missing: {args.ai_source_image}")
    hd_matrix.perf_config.wait_for_backend()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    result = run(args)
    (args.artifacts / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": result["status"], "artifacts": str(args.artifacts), "failures": result["failures"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
