"""MPS weight transfers must wait for queued compute first, without holding the GIL.

See ``mps_transfer_sync``. The two tests marked ``requires_mps`` need a real MPS device and are
skipped elsewhere.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest
import torch
from ltx_core.loader import streaming_transport

from services.patches import mps_transfer_sync as patch

requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")


def _record_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    @contextmanager
    def fake_transfer(_self, after):
        calls.append(f"enter after={after}")
        try:
            yield "handle"
        finally:
            calls.append("exit")

    monkeypatch.setattr(patch, "_orig_transfer", fake_transfer)
    monkeypatch.setattr(patch, "_wait_for_queued_compute", lambda: calls.append("wait"))
    return calls


def _queue_gpu_work(seconds: float) -> torch.Tensor:
    """Queue about ``seconds`` of MPS work and return before it finishes."""
    a = torch.randn(4096, 4096, device="mps", dtype=torch.bfloat16)

    def queue(count: int) -> torch.Tensor:
        x = a
        for _ in range(count):
            x = x @ a
            x = x / (x.abs().max() + 1)
        return x

    queue(5)
    torch.mps.synchronize()
    started = time.perf_counter()
    queue(20)
    torch.mps.synchronize()
    per_matmul = (time.perf_counter() - started) / 20
    return queue(max(20, int(seconds / per_matmul)))


def test_patch_replaces_mps_transport_transfer_only() -> None:
    assert streaming_transport.MpsTransport.transfer is patch._transfer_after_compute  # pyright: ignore[reportPrivateUsage]
    assert streaming_transport.CudaTransport.transfer is not patch._transfer_after_compute  # pyright: ignore[reportPrivateUsage]
    assert streaming_transport.CpuTransport.transfer is not patch._transfer_after_compute  # pyright: ignore[reportPrivateUsage]


def test_waits_before_the_upstream_transfer_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_calls(monkeypatch)
    with patch._transfer_after_compute(object(), "prev") as handle:  # pyright: ignore[reportPrivateUsage]
        assert handle == "handle"
        calls.append("body")
    assert calls == ["wait", "enter after=prev", "body", "exit"]


def test_body_errors_propagate_and_upstream_exit_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_calls(monkeypatch)
    with pytest.raises(ValueError, match="boom"):
        with patch._transfer_after_compute(object(), None):  # pyright: ignore[reportPrivateUsage]
            raise ValueError("boom")
    assert calls == ["wait", "enter after=None", "exit"]


def test_wait_stops_polling_and_falls_back_to_the_blocking_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    queries = 0
    synchronized = 0

    class NeverSignalledEvent:
        def record(self) -> None:
            pass

        def query(self) -> bool:
            nonlocal queries
            queries += 1
            return False

    def count_synchronize() -> None:
        nonlocal synchronized
        synchronized += 1

    monkeypatch.setattr(torch.mps, "Event", NeverSignalledEvent)
    monkeypatch.setattr(torch.mps, "synchronize", count_synchronize)
    monkeypatch.setattr(patch, "_POLL_TIMEOUT_S", 0.05)

    started = time.perf_counter()
    patch._wait_for_queued_compute()  # pyright: ignore[reportPrivateUsage]
    elapsed = time.perf_counter() - started

    assert queries > 1, "the poll never ran"
    assert synchronized == 1, "the blocking sync must run once after the poll gives up"
    assert elapsed < 1.0, f"the poll ignored its timeout: {elapsed:.3f}s"


@requires_mps
def test_wait_returns_only_after_queued_work_is_done() -> None:
    work = _queue_gpu_work(1.0)

    started = time.perf_counter()
    patch._wait_for_queued_compute()  # pyright: ignore[reportPrivateUsage]
    waited = time.perf_counter() - started

    started = time.perf_counter()
    torch.mps.synchronize()
    residual = time.perf_counter() - started

    del work
    assert waited > 0.3, "the queued work finished before the wait started, so this test proves nothing"
    assert residual < 0.05, f"queued work was still running after the wait: {residual:.3f}s"


@requires_mps
def test_wait_does_not_hold_the_gil() -> None:
    ticks: list[float] = []
    stop = threading.Event()

    def ticker() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.005)

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    time.sleep(0.1)
    work = _queue_gpu_work(1.5)
    ticks.clear()

    started = time.perf_counter()
    patch._wait_for_queued_compute()  # pyright: ignore[reportPrivateUsage]
    ended = time.perf_counter()
    stop.set()
    thread.join()
    del work

    window = [t for t in ticks if started <= t <= ended]
    gaps = [later - earlier for earlier, later in zip(window, window[1:])]
    assert ended - started > 0.5, "the wait was too short to show a held GIL"
    assert len(window) >= 20, f"the other thread ran only {len(window)} times during the wait"
    assert max(gaps) < 0.25, f"another thread was blocked for {max(gaps):.3f}s during the wait"
