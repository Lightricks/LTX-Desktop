from __future__ import annotations

import logging
import sqlite3
import sys
import threading
import time
import wave
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from PIL import Image

from _routes._errors import HTTPError
from api_types import (
    AudioToVideoParams,
    GenerateVideoCompleteResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    GenerationErrorCode,
)
from frame_math import compute_num_frames
from services.analytics import AnalyticsService
from services.features.video import TextToVideoExecutor
from services.features.video.audio_to_video import AudioToVideoExecutor
from services.features.video.image_to_video import ImageToVideoExecutor
from services.features.video.text_to_video import ReservedVideoGenerator
from services.generation_queue.control import (
    SharedGenerationSlot,
    cancel_queued_generation,
)
from services.generation_queue.registry import ExecutorRegistry
from services.generation_queue.runner import QueueRunner
from services.generation_queue.types import (
    GenerationExecutor,
    OutputAllocation,
    OutputPlan,
)
from services.media_probe import probe_file
from services.records import AttemptError, CapabilityFailedError, GenerationRecord
from services.sqlite_store import SqliteStore
from services.generation_interrupt import (
    GenerationCancelledError,
    is_requested,
    request,
)
from tests.fakes import FakeResponse
from tests.fakes.generation import FakeGenerationExecutor
from tests.fakes.services import FakeHTTPClient, FakeTaskRunner


class _SkipParamValidation:
    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        del spec, contract_version


def _spec(prompt: str) -> dict[str, JsonValue]:
    return {"params": {"prompt": prompt, "model": "ltx-2.5-fast"}, "inputs": {}}


def _png(path: Path, size: tuple[int, int] = (16, 16)) -> Path:
    Image.new("RGB", size, color=(1, 2, 3)).save(path)
    return path


def _wav(path: Path, *, duration_seconds: float = 8.0) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * 8000))
    return path


def _prompt(generation: GenerationRecord) -> object:
    params = generation.spec.get("params")
    if isinstance(params, dict):
        return params.get("prompt")
    return None


def _a2v_params(**overrides: JsonValue) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {
        "prompt": "the character speaks",
        "resolution": "540p",
        "model": "ltx-2.5-fast",
        "fps": 24,
        "numFrames": 185,
    }
    params.update(overrides)
    return params


def _generation(
    *,
    params: dict[str, JsonValue] | None = None,
    inputs: dict[str, JsonValue] | None = None,
    generation_id: str = "t2v-1",
    feature: str = "text-to-video",
    contract_version: int = 1,
) -> GenerationRecord:
    return GenerationRecord(
        id=generation_id,
        feature=feature,
        contract_version=contract_version,
        status="running",
        spec={
            "params": params
            if params is not None
            else {"prompt": "fox", "model": "ltx-2.5-fast"},
            "inputs": inputs if inputs is not None else {},
        },
        error_code=None,
        created_at=1,
        queued_at=1,
        attempt_count=1,
        started_at=1,
        finished_at=None,
        outputs=(),
    )


def _enable_local_t2v(test_state, create_fake_model_files) -> None:
    create_fake_model_files()
    test_state.state.app_settings.use_local_text_encoder = True


def make_runner(
    test_state,
    db: SqliteStore,
    executor: GenerationExecutor,
    *,
    generation: SharedGenerationSlot | None = None,
    stop_join_timeout_seconds: float = 600.0,
) -> QueueRunner:
    return QueueRunner(
        db=db,
        generation=test_state.generation if generation is None else generation,
        executors=ExecutorRegistry({"text-to-video": executor}),
        stop_join_timeout_seconds=stop_join_timeout_seconds,
    )


_WAIT_UNTIL_TIMEOUT = 10.0 if sys.platform == "win32" else 3.0


def wait_until(
    predicate: Callable[[], bool], timeout: float = _WAIT_UNTIL_TIMEOUT
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


_RUNNER_VIDEO_PLAN = OutputPlan(
    slot="output",
    media_kind="video",
    mime_type="video/mp4",
    name="output.mp4",
)


class _OpenGenerationSlot:
    def wait_for_generation_slot(
        self, shutdown: threading.Event
    ) -> AbstractContextManager[None]:
        del shutdown
        return nullcontext()

    def cancel_generation(self) -> object:
        return None


class _PartialFailureExecutor(_SkipParamValidation):
    def __init__(self) -> None:
        self.output_path: str | None = None

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        return (_RUNNER_VIDEO_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        del generation
        self.output_path = outputs[0].dest_path
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial output")
        raise RuntimeError("partial failure")


class _DelayedSuccessExecutor(_SkipParamValidation):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.output_path: str | None = None

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        return (_RUNNER_VIDEO_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        self.output_path = outputs[0].dest_path
        self.started.set()
        try:
            if not self.release.wait(timeout=3.0):
                raise RuntimeError("delayed executor was not released")
            FakeGenerationExecutor().execute(generation, outputs)
        finally:
            self.finished.set()


class _BlockingFileWritingExecutor(_SkipParamValidation):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.output_path: str | None = None

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        return (_RUNNER_VIDEO_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        del generation
        self.output_path = outputs[0].dest_path
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"active output")
        self.started.set()
        try:
            if not self.release.wait(timeout=3.0):
                raise RuntimeError("blocking executor was not released")
            raise GenerationCancelledError()
        finally:
            self.finished.set()


class _BlockingCancelSlot:
    def __init__(self, inner: SharedGenerationSlot) -> None:
        self._inner = inner
        self.cancel_entered = threading.Event()
        self.allow_cancel = threading.Event()

    def wait_for_generation_slot(
        self, shutdown: threading.Event
    ) -> AbstractContextManager[None]:
        return self._inner.wait_for_generation_slot(shutdown)

    def cancel_generation(self) -> object:
        self.cancel_entered.set()
        if not self.allow_cancel.wait(timeout=3.0):
            raise RuntimeError("cancel was not released")
        return self._inner.cancel_generation()


class _BlockingClaimDb(SqliteStore):
    def __init__(self, app_data_dir: Path) -> None:
        super().__init__(app_data_dir)
        self.claim_started = threading.Event()
        self.allow_claim = threading.Event()

    def claim_next_queued(self) -> GenerationRecord | None:
        self.claim_started.set()
        if not self.allow_claim.wait(timeout=2.0):
            raise RuntimeError("claim was not released")
        return super().claim_next_queued()


class _BlockAfterClaimDb(SqliteStore):
    def __init__(self, app_data_dir: Path) -> None:
        super().__init__(app_data_dir)
        self.claimed = threading.Event()
        self.allow_continue = threading.Event()

    def claim_next_queued(self) -> GenerationRecord | None:
        claimed = super().claim_next_queued()
        if claimed is not None:
            self.claimed.set()
            if not self.allow_continue.wait(timeout=2.0):
                raise RuntimeError("post-claim was not released")
        return claimed


class _ClaimBeforeCancellationDb(SqliteStore):
    def __init__(self, app_data_dir: Path) -> None:
        super().__init__(app_data_dir)
        self.claimed: GenerationRecord | None = None

    def request_generation_cancellation(self, generation_id: str) -> GenerationRecord:
        self.claimed = super().claim_next_queued()
        return super().request_generation_cancellation(generation_id)


class _ExecuteRecorder(_SkipParamValidation):
    def __init__(self) -> None:
        self.executed = threading.Event()
        self.planned = threading.Event()

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        self.planned.set()
        return (_RUNNER_VIDEO_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        self.executed.set()
        FakeGenerationExecutor().execute(generation, outputs)


class _FailingFinalizationDb(SqliteStore):
    def __init__(self, app_data_dir: Path, failure: Exception) -> None:
        super().__init__(app_data_dir)
        self._failure: Exception | None = failure

    def mark_failed(
        self,
        generation_id: str,
        error_code: GenerationErrorCode,
        partial_paths: list[str] | None = None,
        *,
        attempt_count: int,
    ) -> GenerationRecord:
        if self._failure is not None:
            failure = self._failure
            self._failure = None
            raise failure
        return super().mark_failed(
            generation_id,
            error_code,
            partial_paths,
            attempt_count=attempt_count,
        )


class _AlwaysFailingFinalizationDb(SqliteStore):
    def __init__(self, app_data_dir: Path) -> None:
        super().__init__(app_data_dir)
        self.finalization_attempted = threading.Event()

    def mark_failed(
        self,
        generation_id: str,
        error_code: GenerationErrorCode,
        partial_paths: list[str] | None = None,
        *,
        attempt_count: int,
    ) -> GenerationRecord:
        self.finalization_attempted.set()
        raise sqlite3.OperationalError("failure terminalization unavailable")


class _FailFirstExecutor(_SkipParamValidation):
    def __init__(self) -> None:
        self.execution_ids: list[str] = []

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        return (_RUNNER_VIDEO_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        self.execution_ids.append(generation.id)
        if _prompt(generation) == "first":
            raise RuntimeError("executor failed")
        FakeGenerationExecutor().execute(generation, outputs)


class _DirectoryFailureExecutor(_SkipParamValidation):
    def __init__(self) -> None:
        self.partial_paths: dict[str, str] = {}

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        return (_RUNNER_VIDEO_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        path = Path(outputs[0].dest_path)
        if _prompt(generation) == "first":
            path.mkdir()
            self.partial_paths[generation.id] = str(path)
            raise RuntimeError("executor failed after creating a directory")
        FakeGenerationExecutor().execute(generation, outputs)


class _DeleteGenerationExecutor(_SkipParamValidation):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self.output_path: str | None = None

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        return (_RUNNER_VIDEO_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        FakeGenerationExecutor().execute(generation, outputs)
        if _prompt(generation) != "vanish":
            return
        self.output_path = outputs[0].dest_path
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("DELETE FROM generations WHERE id = ?", (generation.id,))
            conn.commit()
        finally:
            conn.close()


def test_runner_claims_and_succeeds_one_generation(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(test_state, db, FakeGenerationExecutor())
    generation = db.insert_generation("text-to-video", _spec("fox"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(
            lambda: db.get_generation(generation.id).status == "succeeded"
        )
    finally:
        runner.stop()


class _OptionalOutputExecutor(_SkipParamValidation):
    """Writes the required output. Writes the optional one only when asked to."""

    def __init__(self, *, write_optional: bool) -> None:
        self._write_optional = write_optional

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        return (
            OutputPlan(slot="main", media_kind="image", mime_type="image/png", name="main.png"),
            OutputPlan(
                slot="extra",
                media_kind="image",
                mime_type="image/png",
                name="extra.png",
                optional=True,
            ),
        )

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        del generation
        for output in outputs[: 2 if self._write_optional else 1]:
            _png(Path(output.dest_path))


@pytest.mark.parametrize(("write_optional", "expected_outputs"), [(True, 2), (False, 1)])
def test_runner_leaves_out_an_optional_output_the_executor_did_not_write(
    tmp_path: Path, write_optional: bool, expected_outputs: int
) -> None:
    db = SqliteStore(tmp_path / "app_data")
    runner = QueueRunner(
        db=db,
        generation=_OpenGenerationSlot(),
        executors=ExecutorRegistry(
            {"text-to-video": _OptionalOutputExecutor(write_optional=write_optional)}
        ),
    )
    generation = db.insert_generation("text-to-video", _spec("fox"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "succeeded")
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert len(loaded.outputs) == expected_outputs
        assert all(Path(output.path).is_file() for output in loaded.outputs)
    finally:
        runner.stop()


def test_runner_fails_when_a_required_output_is_missing(tmp_path: Path) -> None:
    class _WritesNothing(_OptionalOutputExecutor):
        def execute(self, generation, outputs) -> None:
            del generation, outputs

    db = SqliteStore(tmp_path / "app_data")
    runner = QueueRunner(
        db=db,
        generation=_OpenGenerationSlot(),
        executors=ExecutorRegistry({"text-to-video": _WritesNothing(write_optional=False)}),
    )
    generation = db.insert_generation("text-to-video", _spec("fox"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "failed")
    finally:
        runner.stop()


def test_runner_names_output_from_prompt_slug(tmp_path: Path) -> None:
    db = SqliteStore(tmp_path / "app_data")
    runner = QueueRunner(
        db=db,
        generation=_OpenGenerationSlot(),
        executors=ExecutorRegistry({"text-to-video": FakeGenerationExecutor()}),
    )
    generation = db.insert_generation("text-to-video", _spec("Fox run"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(
            lambda: db.get_generation(generation.id).status == "succeeded"
        )
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert len(loaded.outputs) == 1
        name = loaded.outputs[0].name
        assert name != "output.mp4"
        stamp = datetime.fromtimestamp(
            loaded.created_at / 1000, tz=timezone.utc
        ).strftime("%Y%m%d-%H%M%S")
        assert name == f"fox-run-{stamp}-{loaded.outputs[0].id[:6]}.mp4"
    finally:
        runner.stop()


def test_runner_processes_queued_generations_fifo(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    seen: list[str] = []
    runner = make_runner(test_state, db, FakeGenerationExecutor(seen_prompts=seen))
    first = db.insert_generation("text-to-video", _spec("first"))
    second = db.insert_generation("text-to-video", _spec("second"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(
            lambda: all(
                db.get_generation(generation.id).status == "succeeded"
                for generation in (first, second)
            )
        )
        assert seen == ["first", "second"]
    finally:
        runner.stop()


def test_runner_marks_executor_failure_without_retry(test_state, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="logging_policy")
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(
        test_state, db, FakeGenerationExecutor(error=RuntimeError("bad input"))
    )
    generation = db.insert_generation("text-to-video", _spec("bad"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "failed")
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.attempt_count == 1
        assert loaded.error_code == "EXECUTOR_FAILED"
        traceback_records = [
            record
            for record in caplog.records
            if record.name == "logging_policy"
            and "generation-queue-runner" in record.getMessage()
            and record.exc_info is not None
        ]
        assert len(traceback_records) == 1
        assert traceback_records[0].exc_info[0] is RuntimeError
    finally:
        runner.stop()


def test_runner_marks_unsupported_contract_version_as_executor_failure(
    test_state,
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(
        test_state, db, TextToVideoExecutor(test_state.video_generation)
    )
    generation = db.insert_generation(
        "text-to-video", _spec("fox"), contract_version=2
    )
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "failed")
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.error_code == "EXECUTOR_FAILED"
    finally:
        runner.stop()


def test_runner_cancellation_marks_generation_cancelled(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(test_state, db, FakeGenerationExecutor(block=True))
    generation = db.insert_generation("text-to-video", _spec("stop"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "running")
        cancelled = cancel_queued_generation(db, generation.id, runner.interrupt)
        assert cancelled.status == "cancelling"
        assert wait_until(
            lambda: db.get_generation(generation.id).status == "cancelled"
        )
    finally:
        request()
        runner.stop()


def test_runner_leaves_queued_generation_when_shared_slot_is_busy(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(test_state, db, FakeGenerationExecutor())
    generation = db.insert_generation("text-to-video", _spec("wait"))
    assert test_state.generation.try_reserve_generation_start() is True
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "queued")
        test_state.generation.release_generation_start_reservation()
        assert wait_until(
            lambda: db.get_generation(generation.id).status == "succeeded"
        )
    finally:
        runner.stop()


def test_runner_stops_without_claiming_new_work(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(test_state, db, FakeGenerationExecutor())
    runner.start()
    try:
        runner.stop()
        generation = db.insert_generation("text-to-video", _spec("after stop"))
        runner.wake()
        assert db.get_generation(generation.id).status == "queued"
    finally:
        runner.stop()


def test_runner_cleans_partial_output_when_executor_fails(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    executor = _PartialFailureExecutor()
    runner = make_runner(test_state, db, executor)
    generation = db.insert_generation("text-to-video", _spec("partial"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "failed")
        assert executor.output_path is not None
        assert wait_until(lambda: not Path(executor.output_path).exists())
    finally:
        runner.stop()


def test_runner_cancel_queued_generation_does_not_request_global_interrupt(
    test_state,
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(test_state, db, FakeGenerationExecutor())
    generation = db.insert_generation("text-to-video", _spec("queued stop"))

    cancelled = cancel_queued_generation(db, generation.id, runner.interrupt)

    assert cancelled.status == "cancelled"
    assert not is_requested()
    loaded = db.get_generation(generation.id)
    assert loaded is not None
    assert loaded.status == "cancelled"


def test_cancel_running_generation_not_owned_by_runner_still_cancels(
    test_state,
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(test_state, db, FakeGenerationExecutor())
    generation = db.insert_generation("text-to-video", _spec("other"))
    claimed = db.claim_next_queued()
    assert claimed is not None
    assert claimed.id == generation.id

    cancelled = cancel_queued_generation(db, generation.id, runner.interrupt)

    assert cancelled.status == "cancelling"
    assert not is_requested()
    loaded = db.get_generation(generation.id)
    assert loaded is not None
    assert loaded.status == "cancelling"
    assert db.fail_running_on_boot() == 1
    assert db.get_generation(generation.id).status == "cancelled"


def test_runner_does_not_overwrite_command_cancellation(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    executor = _DelayedSuccessExecutor()
    runner = make_runner(test_state, db, executor)
    generation = db.insert_generation("text-to-video", _spec("cancelled"))
    runner.start()
    try:
        runner.wake()
        assert executor.started.wait(timeout=2.0)
        cancelled = cancel_queued_generation(db, generation.id, runner.interrupt)
        assert cancelled.status == "cancelling"
        executor.release.set()
        assert wait_until(
            lambda: (
                executor.finished.is_set()
                and executor.output_path is not None
                and db.get_generation(generation.id).status == "cancelled"
            ),
            timeout=3.0,
        )
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.status == "cancelled"
        assert loaded.outputs == ()
        assert executor.output_path is not None
        assert not Path(executor.output_path).exists()
    finally:
        executor.release.set()
        runner.stop()


def test_runner_defers_active_output_cleanup_until_executor_unwinds(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    executor = _BlockingFileWritingExecutor()
    runner = make_runner(test_state, db, executor)
    generation = db.insert_generation("text-to-video", _spec("cancelled"))
    runner.start()
    try:
        runner.wake()
        assert executor.started.wait(timeout=2.0)
        assert executor.output_path is not None
        active_path = Path(executor.output_path)
        assert active_path.is_file()

        cancelled = cancel_queued_generation(db, generation.id, runner.interrupt)

        assert cancelled.status == "cancelling"
        assert active_path.is_file()
        assert executor.finished.is_set() is False

        executor.release.set()
        assert wait_until(
            lambda: executor.finished.is_set() and not active_path.exists(),
            timeout=3.0,
        )
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.status == "cancelled"
    finally:
        executor.release.set()
        runner.stop()


def test_runner_stops_while_waiting_for_shared_slot_without_failing_work(
    test_state,
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(test_state, db, FakeGenerationExecutor())
    generation = db.insert_generation("text-to-video", _spec("shutdown"))
    assert test_state.generation.try_reserve_generation_start() is True
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "queued")
        runner.stop()
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.status == "queued"
        assert loaded.attempt_count == 0
        assert runner.is_running is False
    finally:
        runner.stop()
        test_state.generation.release_generation_start_reservation()


def test_cancel_queued_does_not_wait_for_blocked_claim(test_state) -> None:
    db = _BlockingClaimDb(test_state.config.app_data_dir)
    runner = make_runner(test_state, db, FakeGenerationExecutor(block=True))
    generation = db.insert_generation("text-to-video", _spec("cancel"))
    runner.start()
    try:
        runner.wake()
        assert db.claim_started.wait(timeout=2.0)
        cancelled = cancel_queued_generation(db, generation.id, runner.interrupt)
        assert cancelled.status == "cancelled"
        assert not is_requested()
        db.allow_claim.set()
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.status == "cancelled"
    finally:
        db.allow_claim.set()
        request()
        runner.stop()


def test_cancel_claimed_between_read_and_write_marks_running_attempt_cancelling(
    test_state,
) -> None:
    db = _ClaimBeforeCancellationDb(test_state.config.app_data_dir)
    generation = db.insert_generation("text-to-video", _spec("cancel-race"))
    interrupted: list[tuple[str, int]] = []

    cancelled = cancel_queued_generation(
        db,
        generation.id,
        lambda generation_id, attempt_count: interrupted.append(
            (generation_id, attempt_count)
        ),
    )

    assert db.claimed is not None
    assert db.claimed.id == generation.id
    assert cancelled.status == "cancelling"
    assert cancelled.attempt_count == db.claimed.attempt_count
    assert interrupted == [(generation.id, db.claimed.attempt_count)]


def test_cancel_after_claim_before_active_does_not_execute(test_state) -> None:
    db = _BlockAfterClaimDb(test_state.config.app_data_dir)
    executor = _ExecuteRecorder()
    runner = make_runner(test_state, db, executor)
    generation = db.insert_generation("text-to-video", _spec("cancel-after-claim"))
    runner.start()
    try:
        runner.wake()
        assert db.claimed.wait(timeout=2.0)
        cancelled = cancel_queued_generation(db, generation.id, runner.interrupt)
        assert cancelled.status == "cancelling"
        assert not is_requested()
        db.allow_continue.set()
        assert wait_until(
            lambda: db.get_generation(generation.id).status == "cancelled"
        )
        assert executor.planned.is_set() is False
        assert executor.executed.is_set() is False
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.status == "cancelled"
    finally:
        db.allow_continue.set()
        runner.stop()


def test_cancel_after_claim_emits_a_paired_start_and_cancelled_end(test_state) -> None:
    db = _BlockAfterClaimDb(test_state.config.app_data_dir)
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(), FakeResponse())
    analytics = AnalyticsService(
        http=http,
        task_runner=FakeTaskRunner(),
        sink_url="http://127.0.0.1:9/analytics",
        token="sink-token",
    )
    runner = QueueRunner(
        db=db,
        generation=test_state.generation,
        executors=ExecutorRegistry({"text-to-video": _ExecuteRecorder()}),
        analytics=analytics,
    )
    spec = _spec("cancel-after-claim")
    spec["_analytics"] = {"client": "desktop"}
    generation = db.insert_generation("text-to-video", spec)
    runner.start()
    try:
        runner.wake()
        assert db.claimed.wait(timeout=2.0)
        cancel_queued_generation(db, generation.id, runner.interrupt)
        db.allow_continue.set()
        assert wait_until(
            lambda: (loaded := db.get_generation(generation.id)) is not None
            and loaded.status == "cancelled"
        )

        payloads = [call.json_payload for call in http.calls]
        assert [payload["eventName"] for payload in payloads if payload is not None] == [
            "generate_started",
            "generate_ended",
        ]
        start = payloads[0]
        end = payloads[1]
        assert start is not None and end is not None
        start_details = start["extraDetails"]
        end_details = end["extraDetails"]
        assert isinstance(start_details, dict)
        assert isinstance(end_details, dict)
        assert start_details["generation_id"] == generation.id
        assert start_details["feature"] == "text-to-video"
        assert set(end_details) == {"generation_id", "attempt", "outcome", "runtime_ms"}
        assert end_details["generation_id"] == generation.id
        assert end_details["attempt"] == start_details["attempt"]
        assert end_details["outcome"] == "cancelled"
        assert isinstance(end_details["runtime_ms"], int)
        assert "prompt" not in start_details
    finally:
        db.allow_continue.set()
        runner.stop()


def test_interrupt_holds_slot_until_cancel_generation_returns(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    executor = _DelayedSuccessExecutor()
    slot = _BlockingCancelSlot(test_state.generation)
    runner = make_runner(test_state, db, executor, generation=slot)
    generation = db.insert_generation("text-to-video", _spec("hold-slot"))
    cancel_thread: threading.Thread | None = None
    runner.start()
    try:
        runner.wake()
        assert executor.started.wait(timeout=2.0)
        cancel_thread = threading.Thread(
            target=cancel_queued_generation,
            args=(db, generation.id, runner.interrupt),
        )
        cancel_thread.start()
        assert slot.cancel_entered.wait(timeout=2.0)
        executor.release.set()
        assert executor.finished.wait(timeout=2.0)
        stolen = wait_until(
            lambda: test_state.generation.try_reserve_generation_start(),
            timeout=0.5,
        )
        assert stolen is False
        slot.allow_cancel.set()
        cancel_thread.join(timeout=2.0)
        assert cancel_thread.is_alive() is False
        assert wait_until(
            lambda: test_state.generation.try_reserve_generation_start()
        )
        test_state.generation.release_generation_start_reservation()
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.status == "cancelled"
    finally:
        slot.allow_cancel.set()
        executor.release.set()
        if cancel_thread is not None:
            cancel_thread.join(timeout=2.0)
        runner.stop()


def test_stale_executor_write_after_retry_does_not_stop_new_attempt(
    test_state,
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    executor = _DelayedSuccessExecutor()
    runner = make_runner(test_state, db, executor)
    generation = db.insert_generation("text-to-video", _spec("retried"))
    claimed = db.claim_next_queued()
    assert claimed is not None
    db.mark_failed(
        generation.id, "EXECUTOR_FAILED", attempt_count=claimed.attempt_count
    )
    retried = db.retry_generation(generation.id)
    assert retried.status == "queued"
    assert retried.attempt_count == claimed.attempt_count + 1

    runner.start()
    try:
        runner.wake()
        assert executor.started.wait(timeout=2.0)
        with pytest.raises(AttemptError):
            db.mark_failed(
                generation.id,
                "EXECUTOR_FAILED",
                attempt_count=claimed.attempt_count,
            )
        current = db.get_generation(generation.id)
        assert current is not None
        assert current.status == "running"
        assert current.attempt_count == retried.attempt_count + 1
        executor.release.set()
        assert wait_until(
            lambda: db.get_generation(generation.id).status == "succeeded"
        )
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.status == "succeeded"
        assert loaded.attempt_count == retried.attempt_count + 1
    finally:
        executor.release.set()
        runner.stop()


def test_runner_marks_capability_failed_error(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(
        test_state,
        db,
        FakeGenerationExecutor(
            error=CapabilityFailedError(
                "no local model",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        ),
    )
    generation = db.insert_generation("text-to-video", _spec("caps"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "failed")
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.error_code == "CAPABILITY_FAILED"
    finally:
        runner.stop()


def test_runner_persists_rejected_ltx_api_key(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(
        test_state,
        db,
        FakeGenerationExecutor(
            error=CapabilityFailedError(
                "This LTX API key isn’t valid.",
                code="LTX_INVALID_API_KEY",
            )
        ),
    )
    generation = db.insert_generation("text-to-video", _spec("rejected-key"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "failed")
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.error_code == "LTX_INVALID_API_KEY"
    finally:
        runner.stop()


def test_runner_persists_prompt_embedding_failure(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    runner = make_runner(
        test_state,
        db,
        FakeGenerationExecutor(
            error=CapabilityFailedError(
                "LTX API text encoding failed.",
                code="LTX_API_PROMPT_EMBEDDING_FAILED",
            )
        ),
    )
    generation = db.insert_generation("text-to-video", _spec("embedding-failed"))
    runner.start()
    try:
        runner.wake()
        assert wait_until(lambda: db.get_generation(generation.id).status == "failed")
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.error_code == "LTX_API_PROMPT_EMBEDDING_FAILED"
    finally:
        runner.stop()


@pytest.mark.parametrize(
    "failure",
    [
        sqlite3.OperationalError("database unavailable"),
        OSError("asset cleanup unavailable"),
        RuntimeError("unexpected finalization failure"),
    ],
)
def test_runner_continues_after_terminalization_failure(
    test_state, failure: Exception
) -> None:
    db = _FailingFinalizationDb(test_state.config.app_data_dir, failure)
    executor = _FailFirstExecutor()
    runner = make_runner(test_state, db, executor)
    first = db.insert_generation("text-to-video", _spec("first"))
    second = db.insert_generation("text-to-video", _spec("second"))
    runner.start()
    try:
        runner.wake()

        def second_succeeded() -> bool:
            loaded = db.get_generation(second.id)
            return loaded is not None and loaded.status == "succeeded"

        assert wait_until(second_succeeded)
        first_loaded = db.get_generation(first.id)
        assert first_loaded is not None
        assert first_loaded.status == "running"
        assert executor.execution_ids.count(first.id) == 1
        assert executor.execution_ids.count(second.id) == 1
    finally:
        runner.stop()


def test_runner_start_does_not_fail_running_generations(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    generation = db.insert_generation("text-to-video", _spec("running"))
    claimed = db.claim_next_queued()
    assert claimed is not None
    runner = make_runner(test_state, db, FakeGenerationExecutor())
    runner.start()
    try:
        loaded = db.get_generation(generation.id)
        assert loaded is not None
        assert loaded.status == "running"
        assert loaded.error_code is None
    finally:
        runner.stop()


def test_stale_running_row_is_failed_by_boot_recovery_not_runner_start(
    test_state,
) -> None:
    db = _AlwaysFailingFinalizationDb(test_state.config.app_data_dir)
    executor = _FailFirstExecutor()
    generation = db.insert_generation("text-to-video", _spec("first"))
    runner = make_runner(test_state, db, executor)
    runner.start()
    try:
        runner.wake()
        assert db.finalization_attempted.wait(timeout=2.0)
        runner.stop()

        stale = db.get_generation(generation.id)
        assert stale is not None
        assert stale.status == "running"

        runner.start()
        still_running = db.get_generation(generation.id)
        assert still_running is not None
        assert still_running.status == "running"
        assert executor.execution_ids == [generation.id]

        assert db.fail_running_on_boot() == 1
        recovered = db.get_generation(generation.id)
        assert recovered is not None
        assert recovered.status == "failed"
        assert recovered.error_code == "INTERRUPTED"
        assert executor.execution_ids == [generation.id]
    finally:
        runner.stop()


def test_runner_cleans_output_when_generation_row_disappears(test_state) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    executor = _DeleteGenerationExecutor(
        test_state.config.app_data_dir / "store.sqlite3"
    )
    runner = make_runner(test_state, db, executor)
    vanished = db.insert_generation("text-to-video", _spec("vanish"))
    next_generation = db.insert_generation("text-to-video", _spec("next"))
    runner.start()
    try:
        runner.wake()

        def next_succeeded() -> bool:
            loaded = db.get_generation(next_generation.id)
            return loaded is not None and loaded.status == "succeeded"

        assert wait_until(next_succeeded)
        assert db.get_generation(vanished.id) is None
        assert executor.output_path is not None
        assert not Path(executor.output_path).exists()
    finally:
        runner.stop()


def test_runner_continues_when_partial_directory_cleanup_fails(
    test_state, caplog
) -> None:
    caplog.set_level(logging.WARNING)
    db = SqliteStore(test_state.config.app_data_dir)
    executor = _DirectoryFailureExecutor()
    runner = make_runner(test_state, db, executor)
    first = db.insert_generation("text-to-video", _spec("first"))
    second = db.insert_generation("text-to-video", _spec("second"))
    runner.start()
    try:
        runner.wake()

        def second_succeeded() -> bool:
            loaded = db.get_generation(second.id)
            return loaded is not None and loaded.status == "succeeded"

        assert wait_until(second_succeeded)
        first_loaded = db.get_generation(first.id)
        assert first_loaded is not None
        assert first_loaded.status == "failed"
        partial_path = executor.partial_paths[first.id]
        assert Path(partial_path).is_dir()
        assert any(
            "Could not remove partial output" in record.getMessage()
            for record in caplog.records
        )
    finally:
        runner.stop()


@dataclass(frozen=True)
class _ReservedVideoCall:
    request: GenerateVideoRequest
    generation_id: str
    output_path: Path | None
    prompt_wrap: Callable[[str], str] | None = None
    local_model_id: str | None = None


class _RecordingReservedVideoGenerator:
    def __init__(self, inner: ReservedVideoGenerator) -> None:
        self._inner = inner
        self.calls: list[_ReservedVideoCall] = []

    def validate_local_a2v_request(
        self, params: AudioToVideoParams, audio_duration_ms: int
    ) -> None:
        self._inner.validate_local_a2v_request(params, audio_duration_ms)

    @property
    def models_dir(self) -> Path:
        models_dir = getattr(self._inner, "models_dir", None)
        if not isinstance(models_dir, Path):
            raise AttributeError("inner generator has no models_dir")
        return models_dir

    def generate_local_reserved(
        self,
        req: GenerateVideoRequest,
        *,
        generation_id: str,
        output_path: Path | None = None,
        prompt_wrap: Callable[[str], str] | None = None,
        a2v_num_frames: int | None = None,
        a2v_audio_duration_seconds: float | None = None,
        local_model_id: str | None = None,
    ) -> GenerateVideoResponse:
        self.calls.append(
            _ReservedVideoCall(
                request=req,
                generation_id=generation_id,
                output_path=output_path,
                prompt_wrap=prompt_wrap,
                local_model_id=local_model_id,
            )
        )
        return self._inner.generate_local_reserved(
            req,
            generation_id=generation_id,
            output_path=output_path,
            prompt_wrap=prompt_wrap,
            a2v_num_frames=a2v_num_frames,
            a2v_audio_duration_seconds=a2v_audio_duration_seconds,
            local_model_id=local_model_id,
        )


class _RecordingReservedVideoGeneratorStub:
    def __init__(
        self,
        error: Exception | None = None,
        *,
        models_dir: Path | None = None,
    ) -> None:
        self.calls: list[_ReservedVideoCall] = []
        self._error = error
        self.models_dir = models_dir if models_dir is not None else Path("/nonexistent-models-dir")

    def validate_local_a2v_request(
        self, params: AudioToVideoParams, audio_duration_ms: int
    ) -> None:
        del params, audio_duration_ms

    def generate_local_reserved(
        self,
        req: GenerateVideoRequest,
        *,
        generation_id: str,
        output_path: Path | None = None,
        prompt_wrap: Callable[[str], str] | None = None,
        a2v_num_frames: int | None = None,
        a2v_audio_duration_seconds: float | None = None,
        local_model_id: str | None = None,
    ) -> GenerateVideoCompleteResponse:
        del a2v_num_frames, a2v_audio_duration_seconds
        self.calls.append(
            _ReservedVideoCall(
                request=req,
                generation_id=generation_id,
                output_path=output_path,
                prompt_wrap=prompt_wrap,
                local_model_id=local_model_id,
            )
        )
        if self._error is not None:
            raise self._error
        return GenerateVideoCompleteResponse(
            status="complete",
            video_path=str(output_path) if output_path is not None else "",
        )


def _stub_reserved(
    test_state,
    create_fake_model_files,
    error: Exception | None = None,
) -> _RecordingReservedVideoGeneratorStub:
    create_fake_model_files()
    return _RecordingReservedVideoGeneratorStub(
        error=error,
        models_dir=test_state.config.default_models_dir,
    )


def test_registry_resolves_text_to_video() -> None:
    executor = FakeGenerationExecutor()
    registry = ExecutorRegistry({"text-to-video": executor})
    assert registry.get("text-to-video") is executor


def test_registry_rejects_unknown_feature() -> None:
    registry = ExecutorRegistry({"text-to-video": FakeGenerationExecutor()})
    with pytest.raises(ValueError, match="unknown generation feature: cozy-felt"):
        registry.get("cozy-felt")


def test_text_to_video_plans_one_video_mp4_output(test_state) -> None:
    executor = TextToVideoExecutor(test_state.video_generation)
    assert executor.plan_outputs(_generation()) == (
        OutputPlan(
            slot="output",
            media_kind="video",
            mime_type="video/mp4",
            name="output.mp4",
        ),
    )


def test_text_to_video_rejects_invalid_params(test_state) -> None:
    executor = TextToVideoExecutor(test_state.video_generation)
    with pytest.raises(ValidationError):
        executor.plan_outputs(_generation(params={"prompt": ""}))
    with pytest.raises(ValidationError):
        executor.plan_outputs(
            _generation(params={"prompt": "fox", "imagePath": "/tmp/start.png"})
        )


def test_text_to_video_rejects_unsupported_contract_version(test_state) -> None:
    executor = TextToVideoExecutor(test_state.video_generation)
    spec = {"params": {"prompt": "fox"}, "inputs": {}}
    with pytest.raises(ValueError, match="contract_version"):
        executor.validate_params(spec, contract_version=2)
    with pytest.raises(ValueError, match="contract_version"):
        executor.plan_outputs(_generation(contract_version=2))
    with pytest.raises(ValueError, match="contract_version"):
        executor.execute(_generation(contract_version=2), ())


@pytest.mark.parametrize("pipeline_id", ["fast", "pro", "pro-2.5", "fast-2.5"])
def test_text_to_video_rejects_pipeline_model_ids(
    test_state, pipeline_id: str
) -> None:
    executor = TextToVideoExecutor(test_state.video_generation)
    with pytest.raises(ValidationError):
        executor.plan_outputs(
            _generation(params={"prompt": "fox", "model": pipeline_id})
        )


def test_text_to_video_rejects_capability_invalid_params_before_pipeline(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    _enable_local_t2v(test_state, create_fake_model_files)
    executor = TextToVideoExecutor(test_state.video_generation)
    generation = _generation(
        params={
            "prompt": "a red fox",
            "model": "ltx-2.5-fast",
            "resolution": "1080p",
            "duration": 8,
            "fps": 24,
        }
    )
    dest = tmp_path / "assets" / "invalid.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        with pytest.raises(CapabilityFailedError) as exc_info:
            executor.execute(generation, outputs)

    assert exc_info.value.code == "INVALID_VIDEO_GENERATION_SPEC"
    assert exc_info.value.detail == (
        "Unsupported local text-to-video duration '8' for pipeline 'fast' "
        "at resolution '1080p' and fps '24'"
    )
    assert fake_services.fast_video_pipeline.create_loras == []
    assert fake_services.fast_video_pipeline.generate_calls == []
    assert not dest.exists()


def test_text_to_video_maps_valid_params_to_generation_request(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    _enable_local_t2v(test_state, create_fake_model_files)
    recorder = _RecordingReservedVideoGenerator(test_state.video_generation)
    executor = TextToVideoExecutor(recorder)
    generation = _generation(
        generation_id="t2v-mapped",
        params={
            "prompt": "a red fox",
            "resolution": "540p",
            "model": "ltx-2.5-fast",
            "duration": 8,
            "fps": 24,
            "seed": 12345,
        },
    )
    dest = tmp_path / "assets" / "queued.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        executor.execute(generation, outputs)

    assert dest.exists()
    assert not dest.is_relative_to(test_state.config.outputs_dir)
    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured.request.prompt == "a red fox"
    assert captured.request.resolution == "540p"
    assert captured.request.duration == 8
    assert captured.request.fps == 24
    assert captured.request.model == "fast"
    assert captured.local_model_id == "ltx-2.5-22b-distilled"
    assert captured.request.seed == 12345
    assert captured.generation_id == "t2v-mapped"
    assert captured.output_path == dest
    call = fake_services.fast_video_pipeline.generate_calls[0]
    assert (call["width"], call["height"]) == (1024, 576)
    assert call["num_frames"] == compute_num_frames(8, 24)
    assert call["frame_rate"] == 24
    assert call["output_path"] == str(dest)


def test_text_to_video_maps_structural_params_without_local_validation(
    tmp_path: Path,
) -> None:
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = TextToVideoExecutor(recorder)
    generation = _generation(
        generation_id="t2v-structural-mapping",
        params={
            "prompt": "a blue fox",
            "resolution": "720p",
            "model": "ltx-2.3-fast",
            "duration": 8,
            "fps": 25,
            "seed": 67890,
        },
    )
    dest = tmp_path / "assets" / "stub.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, outputs)

    assert exc_info.value.code == "LTX_MODEL_NOT_INSTALLED"
    assert recorder.calls == []
    assert not dest.exists()


def test_recipe_executor_forwards_studio_scaffold_wrap(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    from services.features.lora_recipes import get_lora_recipe, make_recipe_prompt_wrap

    recipe = get_lora_recipe("cozy-felt")
    assert recipe is not None
    recorder = _stub_reserved(test_state, create_fake_model_files)
    executor = TextToVideoExecutor(recorder, prompt_wrap=make_recipe_prompt_wrap(recipe))
    generation = _generation(
        generation_id="cozy-felt-wrap",
        params={
            "prompt": "a blue fox",
            "model": "ltx-2.5-fast",
            "resolution": "720p",
            "duration": 8,
            "fps": 24,
        },
    )
    dest = tmp_path / "assets" / "felt.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    executor.execute(generation, outputs)

    assert len(recorder.calls) == 1
    wrap = recorder.calls[0].prompt_wrap
    assert wrap is not None
    # Post-enhance the executor wraps the user scene in the felt trigger + scaffold.
    wrapped = wrap("a blue fox")
    assert wrapped.startswith("F3ltCut0u7 handcrafted felt")
    assert "a blue fox" in wrapped


def test_i2v_recipe_executor_forwards_scaffold_and_start_image(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    from runtime_config.model_download_specs import resolve_lora_path
    from services.features.lora_recipes import (
        get_lora_recipe,
        make_recipe_lora_resolver,
        make_recipe_prompt_wrap,
    )

    recipe = get_lora_recipe("dolly-in")
    assert recipe is not None
    db = SqliteStore(test_state.config.app_data_dir)
    start = db.ingest_upload(str(_png(tmp_path / "dolly-start.png")))
    recorder = _stub_reserved(test_state, create_fake_model_files)
    resolved = resolve_lora_path(
        test_state.config.default_models_dir,
        "dolly-in",
        "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(b"\x00" * 64)
    executor = ImageToVideoExecutor(
        recorder,
        db,
        prompt_wrap=make_recipe_prompt_wrap(recipe),
        lora_resolver=make_recipe_lora_resolver(
            recipe,
            catalog=test_state._lora_catalog_provider,
            models_dir=lambda: test_state.config.default_models_dir,
        ),
    )
    generation = _generation(
        generation_id="dolly-in-wrap",
        feature="dolly-in",
        params={
            "prompt": "push in on her face",
            "model": "ltx-2.5-fast",
            "aspectRatio": "auto",
            "resolution": "720p",
            "duration": 8,
            "fps": 24,
            "loras": [{"ref": "", "scale": 1.0, "catalogId": "dolly-in"}],
        },
        inputs={"startFrame": {"assetId": start.id}},
    )
    dest = tmp_path / "assets" / "dolly-in.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    executor.execute(generation, outputs)

    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured.request.imagePath == start.path
    wrap = captured.prompt_wrap
    assert wrap is not None
    wrapped = wrap("push in on her face")
    assert wrapped.startswith("A smooth dolly-in camera move")
    assert "push in on her face" in wrapped
    assert captured.request.loras is not None
    assert len(captured.request.loras) == 1
    assert captured.request.loras[0].catalogId == "dolly-in"
    assert captured.request.loras[0].scale == 1.0
    assert captured.request.loras[0].ref == str(resolved)
    assert captured.request.loras[0].ref != ""


def test_text_to_video_maps_2_3_offering_without_changing_active_settings(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"
    recorder = _RecordingReservedVideoGenerator(test_state.video_generation)
    executor = TextToVideoExecutor(recorder)
    generation = _generation(
        generation_id="t2v-offering-2.3",
        params={
            "prompt": "a red fox",
            "resolution": "540p",
            "model": "ltx-2.3-fast",
            "duration": 5,
            "fps": 24,
        },
    )
    dest = tmp_path / "assets" / "queued-2.3.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        executor.execute(generation, outputs)

    captured = recorder.calls[0]
    assert captured.request.model == "fast"
    assert captured.local_model_id == "ltx-2.3-22b-distilled-1.1"
    slot = test_state.state.gpu_slot
    assert slot is not None
    assert slot.active_pipeline.ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.5-22b-distilled"


@pytest.mark.parametrize("pipeline_id", ["fast", "pro", "pro-2.5", "fast-2.5"])
def test_image_to_video_rejects_pipeline_model_ids(
    test_state, tmp_path: Path, pipeline_id: str
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    start = db.ingest_upload(str(_png(tmp_path / "start.png")))
    executor = ImageToVideoExecutor(test_state.video_generation, db)
    with pytest.raises(ValidationError):
        executor.plan_outputs(
            _generation(
                feature="image-to-video",
                params={"prompt": "fox", "model": pipeline_id},
                inputs={"startFrame": {"assetId": start.id}},
            )
        )


def test_image_to_video_maps_2_3_offering_without_changing_active_settings(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"
    db = SqliteStore(test_state.config.app_data_dir)
    start = db.ingest_upload(str(_png(tmp_path / "start.png")))
    recorder = _RecordingReservedVideoGenerator(test_state.video_generation)
    executor = ImageToVideoExecutor(recorder, db)
    generation = _generation(
        generation_id="i2v-offering-2.3",
        feature="image-to-video",
        params={
            "prompt": "a red fox",
            "resolution": "540p",
            "model": "ltx-2.3-fast",
            "duration": 5,
            "fps": 24,
        },
        inputs={"startFrame": {"assetId": start.id}},
    )
    dest = tmp_path / "assets" / "queued-i2v-2.3.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        executor.execute(generation, outputs)

    captured = recorder.calls[0]
    assert captured.request.model == "fast"
    assert captured.request.imagePath == start.path
    assert captured.local_model_id == "ltx-2.3-22b-distilled-1.1"
    slot = test_state.state.gpu_slot
    assert slot is not None
    assert slot.active_pipeline.ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.5-22b-distilled"


def test_text_to_video_rejects_undownloaded_offering(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.use_local_text_encoder = True
    executor = TextToVideoExecutor(test_state.video_generation)
    generation = _generation(
        params={
            "prompt": "a red fox",
            "model": "ltx-2.3-fast",
            "resolution": "540p",
            "duration": 5,
            "fps": 24,
        }
    )
    dest = tmp_path / "assets" / "missing.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        with pytest.raises(CapabilityFailedError) as exc_info:
            executor.execute(generation, outputs)

    assert exc_info.value.code == "LTX_MODEL_NOT_INSTALLED"


def test_image_to_video_maps_asset_ids_to_executor_paths(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    start = db.ingest_upload(str(_png(tmp_path / "start.png")))
    end = db.ingest_upload(str(_png(tmp_path / "end.png")))
    recorder = _stub_reserved(test_state, create_fake_model_files)
    executor = ImageToVideoExecutor(recorder, db)
    generation = _generation(
        generation_id="i2v-mapped",
        feature="image-to-video",
        params={"prompt": "", "model": "ltx-2.5-fast"},
        inputs={
            "startFrame": {"assetId": start.id},
            "endFrame": {"assetId": end.id},
        },
    )
    dest = tmp_path / "assets" / "i2v.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    executor.execute(generation, outputs)

    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured.request.prompt == ""
    assert captured.request.imagePath == start.path
    assert captured.request.lastImagePath == end.path
    assert captured.request.audio is False
    assert captured.request.keyframes == []
    assert captured.generation_id == "i2v-mapped"
    assert captured.output_path == dest
    assert start.path not in str(generation.spec)
    assert end.path not in str(generation.spec)


@pytest.mark.parametrize(
    ("requested_ratio", "start_size", "expected_ratio"),
    [
        ("auto", (32, 16), "16:9"),
        ("auto", (16, 32), "9:16"),
        ("auto", (32, 32), "1:1"),
        ("16:9", (16, 32), "16:9"),
        ("9:16", (32, 16), "9:16"),
    ],
)
def test_image_to_video_resolves_auto_aspect_ratio_from_start_frame(
    test_state,
    create_fake_model_files,
    tmp_path: Path,
    requested_ratio: str,
    start_size: tuple[int, int],
    expected_ratio: str,
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    # Windows rejects ':' in filenames (`16:9` would be an ADS path).
    case_id = f"{requested_ratio.replace(':', '-')}-{start_size[0]}x{start_size[1]}"
    start = db.ingest_upload(str(_png(tmp_path / f"start-{case_id}.png", start_size)))
    recorder = _stub_reserved(test_state, create_fake_model_files)
    executor = ImageToVideoExecutor(recorder, db)
    generation = _generation(
        feature="image-to-video",
        params={"prompt": "fox", "model": "ltx-2.5-fast", "aspectRatio": requested_ratio},
        inputs={"startFrame": {"assetId": start.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / f"{case_id}.mp4"),
    )

    executor.execute(generation, (output,))

    assert recorder.calls[0].request.aspectRatio == expected_ratio
    assert generation.spec["params"] == {
        "prompt": "fox",
        "model": "ltx-2.5-fast",
        "aspectRatio": requested_ratio,
    }


def test_image_to_video_auto_on_api_only_resolution_is_a_capability_failure(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    start = db.ingest_upload(str(_png(tmp_path / "start.png", (32, 32))))
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = ImageToVideoExecutor(recorder, db)
    generation = _generation(
        feature="image-to-video",
        params={
            "prompt": "fox",
            "model": "ltx-2.5-fast",
            "aspectRatio": "auto",
            "resolution": "1440p",
        },
        inputs={"startFrame": {"assetId": start.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "i2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INVALID_LOCAL_RESOLUTION"
    assert recorder.calls == []


def test_image_to_video_rejects_non_image_asset(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "input.wav")))
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = ImageToVideoExecutor(recorder, db)
    generation = _generation(
        feature="image-to-video",
        inputs={"startFrame": {"assetId": audio.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "i2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "startFrame image is no longer available"
    assert audio.id not in exc_info.value.detail
    assert audio.path not in exc_info.value.detail
    assert recorder.calls == []


def test_image_to_video_execute_rejects_unavailable_start_frame(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    start = db.ingest_upload(str(_png(tmp_path / "start.png")))
    Path(start.path).unlink()
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = ImageToVideoExecutor(recorder, db)
    generation = _generation(
        feature="image-to-video",
        inputs={"startFrame": {"assetId": start.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "i2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "startFrame image is no longer available"
    assert start.id not in exc_info.value.detail
    assert start.path not in exc_info.value.detail
    assert "/" not in exc_info.value.detail
    assert "\\" not in exc_info.value.detail
    assert recorder.calls == []


def test_image_to_video_execute_rejects_missing_start_frame_record(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    missing_id = "missing-start-frame"
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = ImageToVideoExecutor(recorder, db)
    generation = _generation(
        feature="image-to-video",
        inputs={"startFrame": {"assetId": missing_id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "i2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "startFrame image is no longer available"
    assert missing_id not in exc_info.value.detail
    assert recorder.calls == []


def test_image_to_video_execute_hides_http_error_filesystem_paths(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    start = db.ingest_upload(str(_png(tmp_path / "start.png")))
    leaked_path = "/secret/models/start.png"
    recorder = _stub_reserved(
        test_state,
        create_fake_model_files,
        error=HTTPError(400, f"Image file not found: {leaked_path}"),
    )
    executor = ImageToVideoExecutor(recorder, db)
    generation = _generation(
        feature="image-to-video",
        inputs={"startFrame": {"assetId": start.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "i2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "image input is no longer available"
    assert leaked_path not in exc_info.value.detail
    assert start.id not in exc_info.value.detail
    assert start.path not in exc_info.value.detail
    assert "/" not in exc_info.value.detail
    assert "\\" not in exc_info.value.detail
    assert len(recorder.calls) == 1


def test_audio_to_video_maps_audio_asset_to_executor_path(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    recorder = _stub_reserved(test_state, create_fake_model_files)
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        generation_id="a2v-mapped",
        feature="audio-to-video",
        params=_a2v_params(),
        inputs={"audio": {"assetId": audio.id}},
    )
    dest = tmp_path / "assets" / "a2v.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    executor.execute(generation, outputs)

    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured.request.prompt == "the character speaks"
    assert captured.request.audioPath == audio.path
    assert captured.request.audio is True
    assert captured.request.imagePath is None
    assert captured.request.lastImagePath is None
    assert captured.request.resolution == "540p"
    assert captured.request.model == "fast"
    assert captured.local_model_id == "ltx-2.5-22b-distilled"
    assert captured.request.aspectRatio == "16:9"
    assert captured.generation_id == "a2v-mapped"
    assert captured.output_path == dest
    assert audio.path not in str(generation.spec)


def test_audio_to_video_executes_advertised_1080p(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.use_local_text_encoder = True
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav", duration_seconds=8.0)))
    recorder = _RecordingReservedVideoGenerator(test_state.video_generation)
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        generation_id="a2v-1080p",
        feature="audio-to-video",
        params=_a2v_params(resolution="1080p", numFrames=185),
        inputs={"audio": {"assetId": audio.id}},
    )
    dest = tmp_path / "assets" / "a2v-1080p.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        executor.execute(generation, outputs)

    captured = recorder.calls[0]
    assert captured.request.resolution == "1080p"
    call = fake_services.a2v_pipeline.generate_calls[0]
    assert call["audio_start_time"] == 0.0
    assert call["num_frames"] == 185


def test_audio_to_video_hears_only_the_1080p_prefix_of_a_longer_clip(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.use_local_text_encoder = True
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav", duration_seconds=20.0)))
    recorder = _RecordingReservedVideoGenerator(test_state.video_generation)
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        generation_id="a2v-1080p-prefix",
        feature="audio-to-video",
        params=_a2v_params(resolution="1080p", numFrames=233),
        inputs={"audio": {"assetId": audio.id}},
    )
    dest = tmp_path / "assets" / "a2v-1080p-prefix.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        executor.execute(generation, outputs)

    call = fake_services.a2v_pipeline.generate_calls[0]
    assert call["audio_start_time"] == 0.0
    assert call["num_frames"] == 233
    assert call["audio_max_duration"] == 10.0


def test_audio_to_video_maps_2_3_offering_without_changing_active_settings(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    recorder = _RecordingReservedVideoGenerator(test_state.video_generation)
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        generation_id="a2v-offering-2.3",
        feature="audio-to-video",
        params=_a2v_params(model="ltx-2.3-fast"),
        inputs={"audio": {"assetId": audio.id}},
    )
    dest = tmp_path / "assets" / "a2v-2.3.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        executor.execute(generation, outputs)

    captured = recorder.calls[0]
    assert captured.request.model == "fast"
    assert captured.local_model_id == "ltx-2.3-22b-distilled-1.1"
    slot = test_state.state.gpu_slot
    assert slot is not None
    assert slot.active_pipeline.ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.5-22b-distilled"


def test_audio_to_video_maps_start_frame_to_image_path(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    start = db.ingest_upload(str(_png(tmp_path / "start.png")))
    recorder = _stub_reserved(test_state, create_fake_model_files)
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        generation_id="a2v-start",
        feature="audio-to-video",
        params=_a2v_params(prompt=""),
        inputs={
            "audio": {"assetId": audio.id},
            "startFrame": {"assetId": start.id},
        },
    )
    dest = tmp_path / "assets" / "a2v-start.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    executor.execute(generation, outputs)

    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured.request.prompt == ""
    assert captured.request.audioPath == audio.path
    assert captured.request.audio is True
    assert captured.request.imagePath == start.path
    assert captured.request.lastImagePath is None
    assert captured.request.resolution == "540p"
    assert captured.request.model == "fast"
    assert captured.local_model_id == "ltx-2.5-22b-distilled"
    assert captured.generation_id == "a2v-start"
    assert captured.output_path == dest
    assert audio.path not in str(generation.spec)
    assert start.path not in str(generation.spec)


@pytest.mark.parametrize(
    ("requested_ratio", "start_size", "expected_ratio"),
    [
        ("auto", (32, 16), "16:9"),
        ("auto", (16, 32), "9:16"),
        ("auto", (32, 32), "1:1"),
        ("16:9", (16, 32), "16:9"),
        ("9:16", (32, 16), "9:16"),
    ],
)
def test_audio_to_video_resolves_auto_aspect_ratio_from_start_frame(
    test_state,
    create_fake_model_files,
    tmp_path: Path,
    requested_ratio: str,
    start_size: tuple[int, int],
    expected_ratio: str,
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    case_id = f"{requested_ratio.replace(':', '-')}-{start_size[0]}x{start_size[1]}"
    audio = db.ingest_upload(str(_wav(tmp_path / f"line-{case_id}.wav")))
    start = db.ingest_upload(str(_png(tmp_path / f"start-{case_id}.png", start_size)))
    recorder = _stub_reserved(test_state, create_fake_model_files)
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(aspectRatio=requested_ratio),
        inputs={
            "audio": {"assetId": audio.id},
            "startFrame": {"assetId": start.id},
        },
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / f"{case_id}.mp4"),
    )

    executor.execute(generation, (output,))

    assert recorder.calls[0].request.aspectRatio == expected_ratio
    assert generation.spec["params"]["aspectRatio"] == requested_ratio


def test_audio_to_video_auto_aspect_ratio_without_start_frame_uses_16_9(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    recorder = _stub_reserved(test_state, create_fake_model_files)
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(aspectRatio="auto"),
        inputs={"audio": {"assetId": audio.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v-auto.mp4"),
    )

    executor.execute(generation, (output,))

    assert recorder.calls[0].request.aspectRatio == "16:9"
    assert generation.spec["params"]["aspectRatio"] == "auto"


def test_audio_to_video_rejects_non_audio_asset(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    image = db.ingest_upload(str(_png(tmp_path / "start.png")))
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(),
        inputs={"audio": {"assetId": image.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "audio is no longer available"
    assert image.id not in exc_info.value.detail
    assert image.path not in exc_info.value.detail
    assert recorder.calls == []


def test_audio_to_video_rejects_non_image_start_frame(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    other_audio = db.ingest_upload(str(_wav(tmp_path / "other.wav")))
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(),
        inputs={
            "audio": {"assetId": audio.id},
            "startFrame": {"assetId": other_audio.id},
        },
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "startFrame image is no longer available"
    assert other_audio.id not in exc_info.value.detail
    assert other_audio.path not in exc_info.value.detail
    assert recorder.calls == []


def test_audio_to_video_execute_rejects_unavailable_audio(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    Path(audio.path).unlink()
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(),
        inputs={"audio": {"assetId": audio.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "audio is no longer available"
    assert audio.id not in exc_info.value.detail
    assert audio.path not in exc_info.value.detail
    assert "/" not in exc_info.value.detail
    assert "\\" not in exc_info.value.detail
    assert recorder.calls == []


def test_audio_to_video_execute_rejects_unavailable_start_frame(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    start = db.ingest_upload(str(_png(tmp_path / "start.png")))
    Path(start.path).unlink()
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(),
        inputs={
            "audio": {"assetId": audio.id},
            "startFrame": {"assetId": start.id},
        },
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "startFrame image is no longer available"
    assert start.id not in exc_info.value.detail
    assert start.path not in exc_info.value.detail
    assert "/" not in exc_info.value.detail
    assert "\\" not in exc_info.value.detail
    assert recorder.calls == []


def test_audio_to_video_execute_rejects_missing_audio_record(
    test_state, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    missing_id = "missing-audio"
    recorder = _RecordingReservedVideoGeneratorStub()
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(),
        inputs={"audio": {"assetId": missing_id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "audio is no longer available"
    assert missing_id not in exc_info.value.detail
    assert recorder.calls == []


def test_audio_to_video_execute_hides_http_error_filesystem_paths(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    leaked_path = "/secret/models/line.wav"
    recorder = _stub_reserved(
        test_state,
        create_fake_model_files,
        error=HTTPError(400, f"Audio file not found: {leaked_path}"),
    )
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(),
        inputs={"audio": {"assetId": audio.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v.mp4"),
    )

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(generation, (output,))

    assert exc_info.value.code == "INPUT_ASSET_NOT_FOUND"
    assert exc_info.value.detail == "input media is no longer available"
    assert "image" not in exc_info.value.detail
    assert leaked_path not in exc_info.value.detail
    assert audio.id not in exc_info.value.detail
    assert audio.path not in exc_info.value.detail
    assert "/" not in exc_info.value.detail
    assert "\\" not in exc_info.value.detail
    assert len(recorder.calls) == 1


def test_audio_to_video_converts_cancelled_to_cancelled_error(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(str(_wav(tmp_path / "line.wav")))
    recorder = _stub_reserved(
        test_state, create_fake_model_files, error=GenerationCancelledError()
    )
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        feature="audio-to-video",
        params=_a2v_params(),
        inputs={"audio": {"assetId": audio.id}},
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v.mp4"),
    )

    with pytest.raises(GenerationCancelledError):
        executor.execute(generation, (output,))


def test_audio_to_video_uses_local_a2v_pipeline_not_ltx_api(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.use_local_text_encoder = True
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(
        str(_wav(tmp_path / "line.wav", duration_seconds=10.563))
    )
    recorder = _RecordingReservedVideoGenerator(test_state.video_generation)
    executor = AudioToVideoExecutor(recorder, db)
    generation = _generation(
        generation_id="a2v-local",
        feature="audio-to-video",
        params=_a2v_params(numFrames=249),
        inputs={"audio": {"assetId": audio.id}},
    )
    dest = tmp_path / "assets" / "a2v-local.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        executor.execute(generation, outputs)

    assert len(recorder.calls) == 1
    captured = recorder.calls[0]
    assert captured.request.audioPath == audio.path
    assert captured.request.audio is True
    assert captured.request.imagePath is None
    assert captured.request.lastImagePath is None
    assert captured.request.resolution == "540p"
    assert captured.request.model == "fast"
    assert captured.local_model_id == "ltx-2.5-22b-distilled"
    assert dest.exists()
    assert fake_services.ltx_api_client.audio_to_video_calls == []
    assert len(fake_services.a2v_pipeline.generate_calls) == 1
    call = fake_services.a2v_pipeline.generate_calls[0]
    assert call["audio_path"] == audio.path
    assert call["num_frames"] == 249
    assert call["audio_max_duration"] == 10.563
    assert call["output_path"] == str(dest)


def test_audio_to_video_rejects_retried_job_with_stale_persisted_frames(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    db = SqliteStore(test_state.config.app_data_dir)
    audio = db.ingest_upload(
        str(_wav(tmp_path / "line.wav", duration_seconds=10.563))
    )
    executor = AudioToVideoExecutor(test_state.video_generation, db)
    generation = db.insert_generation(
        "audio-to-video",
        {
            "params": _a2v_params(numFrames=481),
            "inputs": {"audio": {"assetId": audio.id}},
        },
    )
    plans = executor.plan_outputs(generation)
    output = OutputAllocation(
        plan=plans[0],
        asset_id="asset-1",
        dest_path=str(tmp_path / "assets" / "a2v.mp4"),
    )
    claimed = db.claim_next_queued()
    assert claimed is not None
    db.mark_failed(
        generation.id,
        "EXECUTOR_FAILED",
        attempt_count=claimed.attempt_count,
    )
    retried = db.retry_generation(generation.id)

    with pytest.raises(CapabilityFailedError) as exc_info:
        executor.execute(retried, (output,))

    assert exc_info.value.code == "INVALID_VIDEO_GENERATION_SPEC"
    assert exc_info.value.detail == (
        "Audio-to-video frame count no longer matches the selected audio."
    )


def test_text_to_video_requires_single_output_allocation(
    test_state, tmp_path: Path
) -> None:
    executor = TextToVideoExecutor(test_state.video_generation)
    generation = _generation()
    dest = tmp_path / "assets" / "wrong.mp4"
    output_plan = OutputPlan(
        slot="output",
        media_kind="video",
        mime_type="video/mp4",
        name="output.mp4",
    )
    preview_plan = OutputPlan(
        slot="preview",
        media_kind="video",
        mime_type="video/mp4",
        name="preview.mp4",
    )
    output = OutputAllocation(plan=output_plan, asset_id="asset-1", dest_path=str(dest))
    preview = OutputAllocation(
        plan=preview_plan, asset_id="asset-2", dest_path=str(dest)
    )
    match = "text-to-video requires a single 'output' allocation"
    with pytest.raises(ValueError, match=match):
        executor.execute(generation, ())
    with pytest.raises(ValueError, match=match):
        executor.execute(generation, (output, output))
    with pytest.raises(ValueError, match=match):
        executor.execute(generation, (preview,))


def test_text_to_video_converts_cancelled_response_to_cancelled_error(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    _enable_local_t2v(test_state, create_fake_model_files)
    fake_services.fast_video_pipeline.raise_on_generate = GenerationCancelledError()
    executor = TextToVideoExecutor(test_state.video_generation)
    generation = _generation()
    dest = tmp_path / "assets" / "cancelled.mp4"
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id="asset-1", dest_path=str(dest)),
    )

    shutdown = threading.Event()
    with test_state.generation.wait_for_generation_slot(shutdown):
        with pytest.raises(GenerationCancelledError):
            executor.execute(generation, outputs)


def test_fake_generation_executor_writes_valid_video(tmp_path: Path) -> None:
    dest = tmp_path / "assets" / "out.mp4"
    plan = OutputPlan(
        slot="output",
        media_kind="video",
        mime_type="video/mp4",
        name="output.mp4",
    )
    FakeGenerationExecutor().execute(
        _generation(),
        (OutputAllocation(plan=plan, asset_id="asset-1", dest_path=str(dest)),),
    )
    kind, mime_type, metadata = probe_file(dest)
    assert kind == "video"
    assert mime_type == "video/mp4"
    assert metadata.mediaType == "video"


def test_fake_generation_executor_records_prompts_and_configured_error() -> None:
    seen: list[str] = []
    executor = FakeGenerationExecutor(
        seen_prompts=seen, error=RuntimeError("bad input")
    )
    with pytest.raises(RuntimeError, match="bad input"):
        executor.execute(_generation(params={"prompt": "first"}), ())
    assert seen == ["first"]


def test_fake_generation_executor_block_raises_after_interrupt() -> None:
    executor = FakeGenerationExecutor(block=True)
    started = threading.Event()
    result: dict[str, BaseException] = {}

    def target() -> None:
        started.set()
        try:
            executor.execute(_generation(), ())
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    assert started.wait(timeout=2.0)
    request()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert isinstance(result.get("error"), GenerationCancelledError)
