"""queued.py drives the /api/generations ledger; exercised against a fake ledger (no backend)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_PERF = Path(__file__).resolve().parents[1] / "performance_runner"
if str(_PERF) not in sys.path:
    sys.path.insert(0, str(_PERF))

import perf_config  # noqa: E402
import queued  # noqa: E402
from queued import QueuedJob  # noqa: E402


class _Ledger:
    """Just enough of /api/assets, /api/generations/* and /api/generation-queue."""

    def __init__(self, *, polls_until_done: int = 1, final_status: str = "succeeded", phase: str = "inference") -> None:
        self.calls: list[tuple[str, str]] = []
        self.created: list[dict[str, Any]] = []
        self.status: dict[str, str] = {}
        self.polls: dict[str, int] = {}
        self.polls_until_done = polls_until_done
        self.final_status = final_status
        self.phase = phase
        self.cancel_error: Exception | None = None
        self.versions = [{"label": "2.3", "model_id": "ltx-2.3", "active": True}]

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("POST", path))
        if path == "/api/assets":
            return {"id": f"asset-{Path(payload['path']).name}"}
        if path.endswith("/cancel"):
            if self.cancel_error:
                raise self.cancel_error
            self.status[path.split("/")[3]] = "cancelled"
            return {}
        generation_id = f"gen-{len(self.created)}"
        self.created.append(payload)
        self.status[generation_id] = "queued"
        return {"id": generation_id}

    def get(self, path: str) -> Any:
        self.calls.append(("GET", path))
        if path == queued.QUEUE_PATH:
            running = [i for i, s in self.status.items() if s in {"queued", "running"}]
            if not running:
                return {"active": None}
            self.status[running[0]] = "running"
            return {"active": {"generation": {"id": running[0]}, "progress": {"phase": self.phase}}}
        generation_id = path.rsplit("/", 1)[1]
        self.polls[generation_id] = self.polls.get(generation_id, 0) + 1
        if self.status[generation_id] in {"queued", "running"} and self.polls[generation_id] >= self.polls_until_done:
            self.status[generation_id] = self.final_status
        status = self.status[generation_id]
        return {"id": generation_id, "status": status, "error_code": "EXECUTOR_FAILED" if status == "failed" else None,
                "outputs": [{"path": f"/out/{generation_id}.mp4"}] if status == "succeeded" else []}


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> _Ledger:
    fake = _Ledger()
    monkeypatch.setattr(perf_config, "_post", fake.post)
    monkeypatch.setattr(perf_config, "_get", fake.get)
    monkeypatch.setattr(perf_config, "ltx_versions", lambda: fake.versions)
    monkeypatch.setattr(queued, "_POLL_S", 0)
    monkeypatch.setattr(queued, "_ASSET_IDS", {})
    return fake


def test_request_body_fills_the_model_and_ingests_inputs_once(ledger: _Ledger) -> None:
    job = QueuedJob("/api/generations/retake", {"startTime": 1.0}, {"video": "/tmp/a.mp4"})

    body = queued.request_body(job)
    queued.request_body(job)

    assert body == {"params": {"model": "ltx-2.3-fast", "startTime": 1.0}, "inputs": {"video": {"assetId": "asset-a.mp4"}}}
    assert ledger.calls.count(("POST", "/api/assets")) == 1


def test_offering_follows_the_active_ltx_version(ledger: _Ledger) -> None:
    ledger.versions = [{"label": "2.3", "model_id": "a"}, {"label": "2.5", "model_id": "b", "active": True}]

    assert queued.offering_id() == "ltx-2.5-fast"


def test_unknown_active_version_is_an_error_not_a_silent_default(ledger: _Ledger) -> None:
    ledger.versions = [{"label": "9.9", "model_id": "z", "active": True}]

    with pytest.raises(RuntimeError, match="cannot map the active LTX version"):
        queued.offering_id()


def test_trigger_submits_every_job_before_waiting_and_returns_outputs_in_order(ledger: _Ledger) -> None:
    jobs = [QueuedJob("/api/generations/text-to-video", {"seed": seed}) for seed in (1, 2, 3)]

    outs = queued.trigger(jobs)

    assert outs == ["/out/gen-0.mp4", "/out/gen-1.mp4", "/out/gen-2.mp4"]
    first_poll = next(i for i, (verb, path) in enumerate(ledger.calls) if verb == "GET")
    assert sum(verb == "POST" for verb, _ in ledger.calls[:first_poll]) == 3


def test_a_failed_job_raises_with_its_error_code(ledger: _Ledger) -> None:
    ledger.final_status = "failed"

    with pytest.raises(RuntimeError, match="EXECUTOR_FAILED"):
        queued.trigger([QueuedJob("/api/generations/text-to-video", {})])


def test_cancel_waits_for_inference_then_cancels_through_the_ledger(ledger: _Ledger) -> None:
    ledger.polls_until_done = 99

    final = queued.trigger_cancellable(QueuedJob("/api/generations/extend", {}), after_s=0)

    assert final["status"] == "cancelled"
    assert ("POST", "/api/generations/gen-0/cancel") in ledger.calls


def test_cancel_that_arrives_after_the_job_finished_is_reported(ledger: _Ledger) -> None:
    ledger.cancel_error = RuntimeError("409 INVALID_GENERATION_STATUS")

    with pytest.raises(RuntimeError, match="expected cancelled, got succeeded"):
        queued.trigger_cancellable(QueuedJob("/api/generations/extend", {}), after_s=0)


def test_cancel_still_cancels_when_inference_never_starts(ledger: _Ledger) -> None:
    ledger.phase = "loading"
    ledger.polls_until_done = 99

    with pytest.raises(RuntimeError, match="never reached phase=inference"):
        queued.trigger_cancellable(QueuedJob("/api/generations/extend", {}), after_s=0, wait_timeout_s=0.3)

    assert ("POST", "/api/generations/gen-0/cancel") in ledger.calls


def _unknown_keys(model: Any, data: Any, path: str = "") -> list[str]:
    """Keys the model would silently ignore (pydantic drops extras, so a typo validates)."""
    from pydantic import BaseModel

    if not (isinstance(model, type) and issubclass(model, BaseModel)) or not isinstance(data, dict):
        return []
    known = {}
    for name, field in model.model_fields.items():
        known[name] = field.annotation
        if field.alias:
            known[field.alias] = field.annotation
    unknown = [f"{path}{key}" for key in data if key not in known]
    for key, value in data.items():
        if key in known:
            unknown += _unknown_keys(known[key], value, f"{path}{key}.")
    return unknown


_REQUEST_MODELS = {
    "/api/generations/text-to-video": "CreateTextToVideoRequest",
    "/api/generations/image-to-video": "CreateImageToVideoRequest",
    "/api/generations/audio-to-video": "CreateAudioToVideoRequest",
    "/api/generations/extend": "CreateExtendRequest",
    "/api/generations/retake": "CreateRetakeRequest",
}


def _queued_jobs() -> list[tuple[str, QueuedJob]]:
    import scenarios as scn

    return [(s.key, job) for s in scn.all_scenarios() for job in s.queued or []]


@pytest.mark.parametrize("offering", ["ltx-2.3-fast", "ltx-2.5-fast"])
def test_every_queued_scenario_body_is_valid_for_the_backend_request_model(offering: str) -> None:
    """Scenario bodies are checked against the real Create*Request models, so a schema
    change (or a typo in a scenario) fails here instead of on the first hardware run."""
    import api_types

    jobs = _queued_jobs()
    assert jobs
    for key, job in jobs:
        body: dict[str, Any] = {"params": {"model": offering, **job.params}}
        if job.inputs:
            body["inputs"] = {name: {"assetId": "asset"} for name in job.inputs}
        model = getattr(api_types, _REQUEST_MODELS[job.route])
        try:
            model.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{key}: {exc}")
        assert not _unknown_keys(model, body), f"{key}: unknown keys {_unknown_keys(model, body)}"


_SYNC_MODELS = {
    None: "GenerateVideoRequest",
    "/api/generate-image": "GenerateImageRequest",
    "/api/extend": "ExtendRequest",
    "/api/retake": "RetakeRequest",
    "/api/ic-lora/generate": "IcLoraGenerateRequest",
}


def test_every_regression_sync_body_is_valid_for_the_backend_request_model() -> None:
    """GenSpace scenarios post raw bodies; check them against the real request models too."""
    import api_types
    import perf_config
    import scenarios as scn

    checked = 0
    for s in scn.all_scenarios():
        if "regression" not in s.tags or s.queued:
            continue
        body = dict(s.overrides) if s.route else {**perf_config.GEN_PAYLOAD_TEMPLATE, **s.overrides}
        model = getattr(api_types, _SYNC_MODELS[s.route])
        try:
            model.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{s.key}: {exc}")
        assert not _unknown_keys(model, body), f"{s.key}: unknown keys {_unknown_keys(model, body)}"
        checked += 1
    assert checked > 10


def test_progress_poll_gives_up_when_the_backend_stays_idle(monkeypatch) -> None:
    import perf_config

    polls: list[str] = []

    def idle(_path):
        polls.append("idle")
        return {"status": "idle"}

    monkeypatch.setattr(perf_config, "_get", idle)

    with pytest.raises(RuntimeError, match="backend is idle"):
        perf_config._poll_until_done(interval_s=0)
    assert len(polls) == perf_config._MAX_IDLE_POLLS


def test_progress_poll_returns_the_output_once_complete(monkeypatch) -> None:
    import perf_config

    states = iter([{"status": "running"}, {"status": "idle"}, {"status": "running"},
                   {"status": "complete", "result": ["/out.mp4"]}])
    monkeypatch.setattr(perf_config, "_get", lambda _path: next(states))

    assert perf_config._poll_until_done(interval_s=0) == "/out.mp4"
