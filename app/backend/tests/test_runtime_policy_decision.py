"""Tests for runtime policy decision helpers."""

from __future__ import annotations

import pytest

from runtime_config.runtime_policy import (
    CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
    DARWIN_ADVERTISED_FLOOR_GB,
    DARWIN_MIN_MACOS_MAJOR,
    DARWIN_STREAMING_FLOOR_GB,
    decide_local_generation_mode,
    should_disk_stream_cuda_weights,
    streaming_prefetch_count_for_mode,
)


def test_darwin_without_mps_unsupported() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=False, ram_gb=64)
        == "unsupported"
    )


def test_darwin_with_low_ram_unsupported() -> None:
    assert (
        decide_local_generation_mode(
            system="Darwin",
            cuda_available=False,
            vram_gb=None,
            mps_available=True,
            ram_gb=DARWIN_STREAMING_FLOOR_GB - 1,
        )
        == "unsupported"
    )


def test_darwin_with_unknown_ram_unsupported() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=None)
        == "unsupported"
    )


def test_darwin_always_streams_when_ram_is_sufficient() -> None:
    for ram_gb in (DARWIN_STREAMING_FLOOR_GB, DARWIN_ADVERTISED_FLOOR_GB, 48, 64, 128):
        assert (
            decide_local_generation_mode(
                system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=ram_gb
            )
            == "streaming_models_loading"
        )


def test_darwin_below_minimum_macos_unsupported() -> None:
    for macos_major in (13, 14, 15, DARWIN_MIN_MACOS_MAJOR - 1):
        assert (
            decide_local_generation_mode(
                system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=64, macos_major=macos_major
            )
            == "unsupported"
        )


def test_darwin_at_or_above_minimum_macos_streams() -> None:
    for macos_major in (DARWIN_MIN_MACOS_MAJOR, DARWIN_MIN_MACOS_MAJOR + 1):
        assert (
            decide_local_generation_mode(
                system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=64, macos_major=macos_major
            )
            == "streaming_models_loading"
        )


def test_darwin_unknown_macos_does_not_block() -> None:
    assert (
        decide_local_generation_mode(
            system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=64, macos_major=None
        )
        == "streaming_models_loading"
    )


@pytest.mark.parametrize("system", ["Windows", "Linux"])
def test_macos_major_does_not_affect_other_platforms(system: str) -> None:
    assert (
        decide_local_generation_mode(system=system, cuda_available=True, vram_gb=24, macos_major=14)
        == "streaming_models_loading"
    )


def test_darwin_defaults_to_unsupported_without_mps_kwargs() -> None:
    """Backward-compat: existing call sites that don't pass mps_available/ram_gb stay unsupported."""
    assert decide_local_generation_mode(system="Darwin", cuda_available=True, vram_gb=64) == "unsupported"


def test_windows_without_cuda_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=False, vram_gb=24) == "unsupported"


def test_windows_with_low_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=8) == "unsupported"


def test_windows_with_unknown_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=None) == "unsupported"


def test_windows_streaming_range() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=9) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=10) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=11) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=12) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=24) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=30) == "streaming_models_loading"


def test_windows_non_fp8_9_to_14gb_is_unsupported() -> None:
    """Pre-Ada cards must not ride the fp8-measured 9 floor on the legacy curve."""
    for vram_gb in (9, 11.99, 12, 14):
        assert (
            decide_local_generation_mode(
                system="Windows", cuda_available=True, vram_gb=vram_gb, fp8_capable=False
            )
            == "unsupported"
        )


def test_windows_non_fp8_15gb_streams() -> None:
    assert (
        decide_local_generation_mode(
            system="Windows", cuda_available=True, vram_gb=15, fp8_capable=False
        )
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(
            system="Windows", cuda_available=True, vram_gb=24, fp8_capable=False
        )
        == "streaming_models_loading"
    )


def test_exact_vram_just_under_nominal_streams_on_cuda() -> None:
    """A 12 GB card reporting 11.99 GiB is a 12 GB card, not an 11 GB one."""
    assert (
        decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=11.99)
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=15.99)
        == "streaming_models_loading"
    )


def test_exact_vram_below_early_out_floor_stays_unsupported_on_cuda() -> None:
    for system in ("Windows", "Linux"):
        assert (
            decide_local_generation_mode(system=system, cuda_available=True, vram_gb=8.8)
            == "unsupported"
        )


def test_windows_full_loading_range() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=31) == "full_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=96) == "full_models_loading"


def test_linux_without_cuda_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=False, vram_gb=24) == "unsupported"


def test_linux_with_low_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=8) == "unsupported"


def test_linux_with_unknown_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=None) == "unsupported"


def test_linux_9_to_14gb_is_unsupported() -> None:
    """Linux rides the legacy curve, unmeasured below 15 — the 9 floor is Windows-only."""
    for vram_gb in (9, 12, 14):
        assert (
            decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=vram_gb)
            == "unsupported"
        )


def test_linux_15gb_streams() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=15) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=15.99) == "streaming_models_loading"


def test_linux_streaming_range() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=15) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=16) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=30) == "streaming_models_loading"


def test_linux_full_loading_range() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=31) == "full_models_loading"


def test_linux_without_fp8_still_unsupported_below_streaming_floor() -> None:
    assert (
        decide_local_generation_mode(
            system="Linux", cuda_available=True, vram_gb=8, fp8_capable=False
        )
        == "unsupported"
    )


def test_linux_without_fp8_streams_even_above_full_floor() -> None:
    """ROCm reports as CUDA but has no fp8; the 31 GB floor assumes an fp8 transformer."""
    assert (
        decide_local_generation_mode(
            system="Linux", cuda_available=True, vram_gb=31, fp8_capable=False
        )
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(
            system="Linux", cuda_available=True, vram_gb=96, fp8_capable=False
        )
        == "streaming_models_loading"
    )


def test_windows_without_fp8_streams_even_above_full_floor() -> None:
    assert (
        decide_local_generation_mode(
            system="Windows", cuda_available=True, vram_gb=31, fp8_capable=False
        )
        == "streaming_models_loading"
    )


def test_fp8_capable_default_preserves_cuda_full_loading() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=31) == "full_models_loading"
    assert (
        decide_local_generation_mode(
            system="Linux", cuda_available=True, vram_gb=31, fp8_capable=True
        )
        == "full_models_loading"
    )


def test_darwin_ignores_fp8_capable() -> None:
    assert (
        decide_local_generation_mode(
            system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=48, fp8_capable=False
        )
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(
            system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=85, fp8_capable=False
        )
        == "streaming_models_loading"
    )


def test_other_systems_fail_closed() -> None:
    assert decide_local_generation_mode(system="FreeBSD", cuda_available=True, vram_gb=48) == "unsupported"


def test_linux_32gb_must_disk_stream_cuda_weights() -> None:
    """LTX-Desktop#163: advertised 32 GB Linux floor cannot pin OffloadMode.CPU."""
    assert should_disk_stream_cuda_weights("Linux", 32, drive_type="nvme") is True
    assert should_disk_stream_cuda_weights("Linux", CUDA_CPU_OFFLOAD_RAM_FLOOR_GB - 1, drive_type=None) is True


def test_linux_high_ram_keeps_cpu_weight_streaming() -> None:
    # Drive class does not route Linux: high RAM pins CPU even on NVMe.
    assert should_disk_stream_cuda_weights("Linux", CUDA_CPU_OFFLOAD_RAM_FLOOR_GB, drive_type="nvme") is False
    assert should_disk_stream_cuda_weights("Linux", 64, drive_type=None) is False


def test_linux_unknown_ram_fails_closed_to_disk_streaming() -> None:
    assert should_disk_stream_cuda_weights("Linux", None, drive_type="nvme") is True


def test_windows_routes_disk_on_nvme_only() -> None:
    """Windows NVMe falls back to disk below the CPU floor (or for Edit); SATA stays on CPU."""
    assert should_disk_stream_cuda_weights("Windows", 32, drive_type="nvme") is True
    assert should_disk_stream_cuda_weights("Windows", 40, drive_type="nvme") is False
    assert should_disk_stream_cuda_weights("Windows", 64, drive_type="nvme") is False
    assert (
        should_disk_stream_cuda_weights(
            "Windows", 48, drive_type="nvme", cpu_ram_floor_gb=64
        )
        is True
    )
    assert (
        should_disk_stream_cuda_weights(
            "Windows", 64, drive_type="nvme", cpu_ram_floor_gb=64
        )
        is False
    )
    assert (
        should_disk_stream_cuda_weights(
            "Windows", 64, drive_type="nvme", cpu_ram_floor_gb=None
        )
        is True
    )
    assert should_disk_stream_cuda_weights("Windows", None, drive_type="nvme") is True
    assert should_disk_stream_cuda_weights("Windows", 64, drive_type="ssd") is False
    assert should_disk_stream_cuda_weights("Windows", 64, drive_type="hdd") is False
    assert should_disk_stream_cuda_weights("Windows", 64, drive_type=None) is False
    assert should_disk_stream_cuda_weights("Windows", None, drive_type=None) is False


def test_darwin_disk_vs_cpu_is_not_this_gate() -> None:
    assert should_disk_stream_cuda_weights("Darwin", 16, drive_type="nvme") is False


def test_streaming_prefetch_count_for_full_loading_is_none() -> None:
    assert streaming_prefetch_count_for_mode("full_models_loading") is None


def test_streaming_prefetch_count_for_streaming_mode_is_two() -> None:
    assert streaming_prefetch_count_for_mode("streaming_models_loading") == 2


def test_streaming_prefetch_count_for_unsupported_asserts() -> None:
    with pytest.raises(AssertionError):
        streaming_prefetch_count_for_mode("unsupported")
