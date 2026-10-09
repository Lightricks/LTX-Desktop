"""OffloadMode mapping for local streaming (LTX-Desktop#163)."""

from __future__ import annotations

import pytest
import torch
from ltx_pipelines.utils.types import OffloadMode

from services.ltx_pipeline_common import offload_mode_for_prefetch_count


def test_full_resident_is_none() -> None:
    assert offload_mode_for_prefetch_count(None, torch.device("cuda")) is OffloadMode.NONE


def test_mps_streaming_is_disk() -> None:
    assert offload_mode_for_prefetch_count(2, torch.device("mps")) is OffloadMode.DISK


def test_linux_cuda_streaming_on_32gb_is_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.ltx_pipeline_common.platform.system", lambda: "Linux")
    monkeypatch.setattr("services.ltx_pipeline_common.host_total_gib", lambda: 32)

    assert offload_mode_for_prefetch_count(2, torch.device("cuda")) is OffloadMode.DISK


def test_linux_cuda_streaming_on_64gb_is_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.ltx_pipeline_common.platform.system", lambda: "Linux")
    monkeypatch.setattr("services.ltx_pipeline_common.host_total_gib", lambda: 64)

    assert offload_mode_for_prefetch_count(2, torch.device("cuda")) is OffloadMode.CPU


def test_linux_cuda_streaming_with_unknown_ram_is_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.ltx_pipeline_common.platform.system", lambda: "Linux")
    monkeypatch.setattr("services.ltx_pipeline_common.host_total_gib", lambda: None)

    assert offload_mode_for_prefetch_count(2, torch.device("cuda")) is OffloadMode.DISK


def test_windows_cuda_streaming_on_32gb_stays_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.ltx_pipeline_common.platform.system", lambda: "Windows")
    monkeypatch.setattr("services.ltx_pipeline_common.host_total_gib", lambda: 32)

    assert offload_mode_for_prefetch_count(2, torch.device("cuda")) is OffloadMode.CPU
