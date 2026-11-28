from services.generation_queue.finished_tray import FINISHED_TRAY_LIMIT, FinishedTray
from services.records import GenerationRecord, QueueGenerationRecord


def _entry(
    generation_id: str,
    status: str,
    *,
    feature: str = "text-to-video",
    error_code: str | None = None,
) -> QueueGenerationRecord:
    return QueueGenerationRecord(
        generation=GenerationRecord(
            id=generation_id,
            feature=feature,
            contract_version=1,
            status=status,  # type: ignore[arg-type]
            spec={"params": {}, "inputs": {}},
            error_code=error_code,  # type: ignore[arg-type]
            created_at=1,
            queued_at=1,
            attempt_count=1,
            started_at=1,
            finished_at=2 if status in {"succeeded", "failed", "cancelled"} else None,
            outputs=(),
        ),
        input_assets=(),
    )


def test_first_snapshot_does_not_backfill_history() -> None:
    tray = FinishedTray()
    tray.observe([_entry("job", "running")], lambda _generation_id: None)

    done, failed, unseen = tray.lists()
    assert done == []
    assert failed == []
    assert unseen == []


def test_departed_jobs_split_into_done_and_failed() -> None:
    tray = FinishedTray()
    running = _entry("job-a", "running")
    failing = _entry("job-d", "running")
    tray.observe([running, failing], lambda _generation_id: None)

    def load(generation_id: str) -> GenerationRecord:
        if generation_id == "job-d":
            return _entry("job-d", "failed", error_code="EXECUTOR_FAILED").generation
        return _entry(generation_id, "succeeded").generation

    tray.observe([], load)

    done, failed, unseen = tray.lists()
    assert [entry.generation.id for entry in done] == ["job-a"]
    assert done[0].generation.status == "succeeded"
    assert [entry.generation.id for entry in failed] == ["job-d"]
    assert failed[0].generation.status == "failed"
    assert unseen == ["job-a"]


def test_cancelled_and_cancelling_jobs_stay_out() -> None:
    tray = FinishedTray()
    tray.observe(
        [_entry("cancelled", "running"), _entry("cancelling", "cancelling")],
        lambda _generation_id: None,
    )

    def load(generation_id: str) -> GenerationRecord:
        if generation_id == "cancelled":
            return _entry("cancelled", "cancelled").generation
        return _entry(generation_id, "succeeded").generation

    tray.observe([], load)

    done, failed, unseen = tray.lists()
    assert done == []
    assert failed == []
    assert unseen == []


def test_seen_dismiss_and_clear_are_shared_and_do_not_resurrect() -> None:
    tray = FinishedTray()
    tray.observe([_entry("job-a", "running")], lambda _generation_id: None)
    tray.observe([], lambda _generation_id: _entry("job-a", "succeeded").generation)
    assert tray.mark_seen("job-a") is True
    assert tray.lists()[2] == []

    assert tray.dismiss_done("job-a") is True
    tray.observe([], lambda _generation_id: _entry("job-a", "succeeded").generation)
    assert tray.lists()[0] == []
    assert tray.mark_seen("job-a") is False
    assert tray.dismiss_done("job-a") is False


def test_clear_keeps_rows_the_caller_cannot_see() -> None:
    tray = FinishedTray()
    tray.remember(_entry("shown", "succeeded"))
    tray.remember(_entry("hidden", "succeeded", feature="desktop-only"))
    tray.remember(_entry("broken", "failed", error_code="EXECUTOR_FAILED"))

    tray.clear_done(lambda feature: feature == "text-to-video")
    tray.clear_failed(lambda feature: feature == "text-to-video")

    done, failed, unseen = tray.lists()
    assert [entry.generation.id for entry in done] == ["hidden"]
    assert unseen == ["hidden"]
    assert failed == []


def test_retry_returns_to_the_tray_unseen() -> None:
    tray = FinishedTray()
    running = _entry("job", "running")
    tray.observe([running], lambda _generation_id: None)
    tray.observe(
        [],
        lambda _generation_id: _entry(
            "job", "failed", error_code="EXECUTOR_FAILED"
        ).generation,
    )

    tray.observe([running], lambda _generation_id: None)
    assert tray.lists() == ([], [], [])

    tray.observe([], lambda _generation_id: _entry("job", "succeeded").generation)
    done, failed, unseen = tray.lists()
    assert [entry.generation.id for entry in done] == ["job"]
    assert failed == []
    assert unseen == ["job"]


def test_stale_live_snapshot_does_not_resurrect_a_dismissed_row() -> None:
    tray = FinishedTray()
    running = _entry("job", "running")
    tray.observe([running], lambda _generation_id: None)
    tray.remember(_entry("job", "succeeded"))
    assert tray.dismiss_done("job") is True

    tray.observe(
        [running],
        lambda _generation_id: _entry("job", "succeeded").generation,
    )
    tray.observe([], lambda _generation_id: _entry("job", "succeeded").generation)

    assert tray.lists()[0] == []


def test_release_lets_a_retried_completion_return_without_a_live_poll() -> None:
    tray = FinishedTray()
    tray.remember(_entry("job", "failed", error_code="EXECUTOR_FAILED"))
    tray.clear_failed()
    assert tray.lists()[1] == []

    tray.release("job")
    tray.remember(_entry("job", "succeeded"))

    done, failed, unseen = tray.lists()
    assert [entry.generation.id for entry in done] == ["job"]
    assert failed == []
    assert unseen == ["job"]


def test_tray_keeps_the_newest_completions() -> None:
    tray = FinishedTray()
    live = [_entry(f"job-{index}", "running") for index in range(FINISHED_TRAY_LIMIT + 1)]
    tray.observe(live, lambda _generation_id: None)
    tray.observe(
        [],
        lambda generation_id: _entry(generation_id, "succeeded").generation,
    )

    done, _failed, unseen = tray.lists()
    assert [entry.generation.id for entry in done] == [
        f"job-{index}" for index in range(FINISHED_TRAY_LIMIT, 0, -1)
    ]
    assert unseen == [entry.generation.id for entry in done]
    assert "job-0" not in unseen
