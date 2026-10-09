"""Queued Home Retake/Extend must claim the GPU slot like T2V / GenSpace."""

from __future__ import annotations

import pytest

from services.features.video.edit_executor import run_gpu_job
from services.generation_interrupt import GenerationCancelledError


class _Slot:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.completed: list[str | None] = []
        self.failed: list[str] = []

    def start_generation(self, generation_id: str) -> None:
        self.started.append(generation_id)

    def complete_generation(self, result: str | list[str] | None = None) -> None:
        self.completed.append(result if isinstance(result, str) or result is None else result[0])

    def fail_generation(self, error: str) -> None:
        self.failed.append(error)


def test_run_gpu_job_starts_and_completes_on_success() -> None:
    slot = _Slot()
    ran = []
    run_gpu_job(slot, "gen-1", "/out.mp4", lambda: ran.append(True))
    assert slot.started == ["gen-1"]
    assert ran == [True]
    assert slot.completed == ["/out.mp4"]
    assert slot.failed == []


def test_run_gpu_job_fails_the_slot_when_the_body_raises() -> None:
    slot = _Slot()

    def _boom() -> None:
        raise RuntimeError("encode hung")

    with pytest.raises(RuntimeError, match="encode hung"):
        run_gpu_job(slot, "gen-1", "/out.mp4", _boom)
    assert slot.started == ["gen-1"]
    assert slot.failed == ["encode hung"]
    assert slot.completed == []


def test_run_gpu_job_does_not_overwrite_cancel_with_fail() -> None:
    slot = _Slot()

    def _cancel() -> None:
        raise GenerationCancelledError()

    with pytest.raises(GenerationCancelledError):
        run_gpu_job(slot, "gen-1", "/out.mp4", _cancel)
    assert slot.started == ["gen-1"]
    assert slot.failed == []
    assert slot.completed == []
