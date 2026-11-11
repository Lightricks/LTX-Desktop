"""MPS weight transfers must wait for queued compute first (see ``mps_transfer_sync``)."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from ltx_core.loader import streaming_transport

from services.patches import mps_transfer_sync as patch


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
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: calls.append("sync"))
    return calls


def test_patch_replaces_mps_transport_transfer_only() -> None:
    assert streaming_transport.MpsTransport.transfer is patch._transfer_after_compute  # pyright: ignore[reportPrivateUsage]
    assert streaming_transport.CudaTransport.transfer is not patch._transfer_after_compute  # pyright: ignore[reportPrivateUsage]
    assert streaming_transport.CpuTransport.transfer is not patch._transfer_after_compute  # pyright: ignore[reportPrivateUsage]


def test_syncs_before_the_upstream_transfer_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_calls(monkeypatch)
    with patch._transfer_after_compute(object(), "prev") as handle:  # pyright: ignore[reportPrivateUsage]
        assert handle == "handle"
        calls.append("body")
    assert calls == ["sync", "enter after=prev", "body", "exit"]


def test_body_errors_propagate_and_upstream_exit_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_calls(monkeypatch)
    with pytest.raises(ValueError, match="boom"):
        with patch._transfer_after_compute(object(), None):  # pyright: ignore[reportPrivateUsage]
            raise ValueError("boom")
    assert calls == ["sync", "enter after=None", "exit"]
