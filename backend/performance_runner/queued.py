"""Driver for the queued generation surface (``/api/generations/...``, Home/Explore).

The synchronous routes in ``perf_config`` (GenSpace) and this surface share the pipelines
but not the request schemas, input-asset handling, ledger status, cancel/retry, or
job ordering. A scenario with ``queued=[QueuedJob(...)]`` is driven through here instead.

A job is described by the same intent as its GenSpace twin: ``params`` is the create body's
``params`` (``model`` is filled in from the active LTX version), ``inputs`` maps an input
name (``video``, ``audio``, ``startFrame``, ``endFrame``) to a local file that is ingested
as an asset first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import perf_config

QUEUE_PATH = "/api/generation-queue"
_ACTIVE = {"queued", "running", "cancelling"}
_POLL_S = 1.0
_WAIT_TIMEOUT_S = 3600.0
_CANCEL_SETTLE_S = 180.0  # a cancel that has not settled by then is a hang, not a slow job

_ASSET_IDS: dict[str, str] = {}


def reset() -> None:
    """Forget ingested assets (a fresh sweep re-registers its inputs)."""
    _ASSET_IDS.clear()


@dataclass(frozen=True)
class QueuedJob:
    route: str
    params: dict[str, Any]
    inputs: dict[str, str] = field(default_factory=dict)  # input name -> local file path


def offering_id() -> str:
    """The queued ``params.model`` for the active LTX version (Settings → Base model)."""
    active = next((v for v in perf_config.ltx_versions() if v.get("active")), {})
    label = f"{active.get('label') or ''} {active.get('model_id') or ''}"
    for version, offering in (("2.5", "ltx-2.5-fast"), ("2.3", "ltx-2.3-fast")):
        if version in label:
            return offering
    raise RuntimeError(f"cannot map the active LTX version ({label.strip() or 'none'}) to a queued offering")


def _asset_id(path: str) -> str:
    if path not in _ASSET_IDS:
        _ASSET_IDS[path] = perf_config._post("/api/assets", {"path": path})["id"]
    return _ASSET_IDS[path]


def request_body(job: QueuedJob) -> dict[str, Any]:
    body: dict[str, Any] = {"params": {"model": offering_id(), **job.params}}
    if job.inputs:
        body["inputs"] = {name: {"assetId": _asset_id(path)} for name, path in job.inputs.items()}
    return body


def _create(job: QueuedJob) -> str:
    return perf_config._post(job.route, request_body(job))["id"]


def _wait(generation_id: str, timeout_s: float = _WAIT_TIMEOUT_S) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        generation = perf_config._get(f"/api/generations/{generation_id}")
        if generation["status"] not in _ACTIVE:
            return generation
        time.sleep(_POLL_S)
    raise RuntimeError(f"generation {generation_id} still active after {timeout_s:.0f}s")


def _output(generation: dict[str, Any]) -> str | None:
    if generation["status"] != "succeeded":
        raise RuntimeError(
            f"generation {generation['id']} {generation['status']} (error_code={generation.get('error_code')})"
        )
    outputs = generation.get("outputs") or []
    return outputs[0]["path"] if outputs else None


def trigger(jobs: list[QueuedJob]) -> list[str | None]:
    """Submit every job first (so several really queue up), then wait for each in order."""
    ids = [_create(job) for job in jobs]
    return [_output(_wait(generation_id)) for generation_id in ids]


def trigger_cancellable(job: QueuedJob, *, after_s: float, wait_timeout_s: float = 120.0) -> dict[str, Any]:
    """Create the job, wait for ``phase=inference``, sleep ``after_s``, cancel via the ledger.

    Returns the final generation, which must be ``cancelled``. The job is always cancelled and
    awaited, even when inference never starts, so the next scenario finds an idle queue."""
    generation_id = _create(job)
    saw_inference = False
    cancel_error: Exception | None = None
    try:
        deadline = time.time() + wait_timeout_s
        while time.time() < deadline:
            active = (perf_config._get(QUEUE_PATH) or {}).get("active") or {}
            if (active.get("generation") or {}).get("id") == generation_id:
                saw_inference = (active.get("progress") or {}).get("phase") == "inference"
                if saw_inference:
                    break
            elif perf_config._get(f"/api/generations/{generation_id}")["status"] not in _ACTIVE:
                break
            time.sleep(0.2)
        if saw_inference:
            time.sleep(after_s)
    finally:
        try:
            perf_config._post(f"/api/generations/{generation_id}/cancel", {})
        except Exception as exc:  # noqa: BLE001  e.g. 409 because the job already finished
            cancel_error = exc
        final = _wait(generation_id, _CANCEL_SETTLE_S)
    if not saw_inference:
        raise RuntimeError(f"queued job never reached phase=inference (ended {final['status']})")
    if final["status"] != "cancelled":
        raise RuntimeError(f"expected cancelled, got {final['status']} (cancel request: {cancel_error or 'accepted'})")
    return final
