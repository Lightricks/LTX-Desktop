"""Logging policy tests for traceback ownership and duplication prevention."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Event

from services.task_runner.threading_runner import ThreadingRunner


def _policy_records(caplog, *, contains: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "logging_policy" and contains in record.getMessage()]


def test_http_500_logs_single_traceback(caplog, client, fake_services) -> None:
    caplog.set_level(logging.WARNING)
    fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("GPU OOM")

    response = client.post("/api/generate-image", json={"prompt": "test"})

    assert response.status_code == 500
    records = _policy_records(caplog, contains="HTTP error on POST /api/generate-image: [500]")
    assert len(records) == 1
    assert records[0].exc_info is not None


def test_http_400_logs_without_traceback(caplog, client) -> None:
    caplog.set_level(logging.WARNING)

    response = client.post(
        "/api/generate",
        json={
            "prompt": "test",
            "resolution": "540p",
            "model": "fast",
            "duration": 2,
            "fps": 24,
            "imagePath": "/no/such/file.png",
        },
    )

    assert response.status_code == 400
    records = _policy_records(caplog, contains="HTTP error on POST /api/generate: [400]")
    assert len(records) == 1
    assert records[0].exc_info is None


def test_unhandled_exception_logs_single_traceback(caplog, test_state, monkeypatch) -> None:
    caplog.set_level(logging.ERROR)

    def _raise_unhandled() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(test_state.health, "get_health", _raise_unhandled)
    from starlette.testclient import TestClient
    from app_factory import create_app

    with TestClient(create_app(handler=test_state), raise_server_exceptions=False) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 500
    records = _policy_records(caplog, contains="Unhandled error on GET /health")
    assert len(records) == 1
    assert records[0].exc_info is not None


def test_background_runner_logs_once_and_calls_error_callback(caplog) -> None:
    caplog.set_level(logging.ERROR)
    runner = ThreadingRunner()
    callback_done = Event()
    callback_errors: list[Exception] = []

    def _worker() -> None:
        raise RuntimeError("background boom")

    def _on_error(exc: Exception) -> None:
        callback_errors.append(exc)
        callback_done.set()

    runner.run_background(
        _worker,
        task_name="test-background-task",
        on_error=_on_error,
        daemon=True,
    )

    assert callback_done.wait(timeout=1.0)
    assert len(callback_errors) == 1
    assert isinstance(callback_errors[0], RuntimeError)

    records = _policy_records(caplog, contains="Unhandled background error in task 'test-background-task'")
    traceback_records = [record for record in records if record.exc_info is not None]
    assert len(traceback_records) == 1


def test_logger_exception_usage_is_restricted_to_boundaries() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    allowed = {
        Path("app_factory.py"),
        Path("services/task_runner/threading_runner.py"),
    }

    for path in backend_dir.rglob("*.py"):
        if "tests" in path.parts or ".venv" in path.parts or "tmp" in path.parts:
            continue
        content = path.read_text(encoding="utf-8")
        if "logger.exception(" in content:
            rel_path = path.relative_to(backend_dir)
            assert rel_path in allowed, f"logger.exception usage is only allowed in boundary files: {rel_path}"
