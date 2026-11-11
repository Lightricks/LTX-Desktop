"""OffloadMode mapping for local streaming (LTX-Desktop#163)."""

from __future__ import annotations

import torch
from ltx_pipelines.utils.types import OffloadMode

from services.ltx_pipeline_common import offload_mode_for_prefetch_count

_GIB = 1024**3


def test_full_resident_is_none_without_consulting_routing() -> None:
    assert (
        offload_mode_for_prefetch_count(
            None, torch.device("cuda"), drive_type="nvme", system="Windows", ram_bytes=64 * _GIB
        )
        is OffloadMode.NONE
    )


def test_mps_streaming_is_disk_without_consulting_routing() -> None:
    assert (
        offload_mode_for_prefetch_count(
            2, torch.device("mps"), drive_type=None, system="Darwin", ram_bytes=None
        )
        is OffloadMode.DISK
    )


def test_linux_cuda_streaming_on_32gb_is_disk() -> None:
    assert (
        offload_mode_for_prefetch_count(
            2, torch.device("cuda"), drive_type=None, system="Linux", ram_bytes=32 * _GIB
        )
        is OffloadMode.DISK
    )


def test_linux_cuda_streaming_on_64gb_is_cpu_even_on_nvme() -> None:
    assert (
        offload_mode_for_prefetch_count(
            2, torch.device("cuda"), drive_type="nvme", system="Linux", ram_bytes=64 * _GIB
        )
        is OffloadMode.CPU
    )


def test_linux_cuda_streaming_with_unknown_ram_is_disk() -> None:
    assert (
        offload_mode_for_prefetch_count(
            2, torch.device("cuda"), drive_type=None, system="Linux", ram_bytes=None
        )
        is OffloadMode.DISK
    )


def test_windows_cuda_streaming_routes_disk_on_nvme() -> None:
    for ram_bytes in (32 * _GIB, None):
        assert (
            offload_mode_for_prefetch_count(
                2, torch.device("cuda"), drive_type="nvme", system="Windows", ram_bytes=ram_bytes
            )
            is OffloadMode.DISK
        )
    for ram_bytes in (40 * _GIB, 64 * _GIB):
        assert (
            offload_mode_for_prefetch_count(
                2, torch.device("cuda"), drive_type="nvme", system="Windows", ram_bytes=ram_bytes
            )
            is OffloadMode.CPU
        )
    # A nominal 64 GB Windows box reports ~63.45 GiB usable; ceiling GiB keeps it on CPU at the 64 GB floor.
    assert (
        offload_mode_for_prefetch_count(
            2,
            torch.device("cuda"),
            drive_type="nvme",
            system="Windows",
            ram_bytes=int(63.45 * _GIB),
            cpu_ram_floor_gb=64,
        )
        is OffloadMode.CPU
    )
    assert (
        offload_mode_for_prefetch_count(
            2,
            torch.device("cuda"),
            drive_type="nvme",
            system="Windows",
            ram_bytes=48 * _GIB,
            cpu_ram_floor_gb=64,
        )
        is OffloadMode.DISK
    )
    assert (
        offload_mode_for_prefetch_count(
            2,
            torch.device("cuda"),
            drive_type="nvme",
            system="Windows",
            ram_bytes=64 * _GIB,
            cpu_ram_floor_gb=None,
        )
        is OffloadMode.DISK
    )


def test_windows_cuda_streaming_stays_cpu_without_nvme() -> None:
    for drive_type in ("ssd", "hdd", None):
        assert (
            offload_mode_for_prefetch_count(
                2, torch.device("cuda"), drive_type=drive_type, system="Windows", ram_bytes=64 * _GIB
            )
            is OffloadMode.CPU
        )
