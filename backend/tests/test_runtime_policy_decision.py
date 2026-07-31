"""Tests for runtime policy decision helpers."""

from __future__ import annotations

import pytest

from runtime_config.runtime_policy import (
    decide_fast_video_engine,
    decide_fast_video_execution_mode,
    decide_local_generation_mode,
    streaming_prefetch_count_for_mode,
)


def test_darwin_without_mps_unsupported() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=False, ram_gb=64)
        == "unsupported"
    )


def test_darwin_with_low_ram_unsupported() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=14)
        == "unsupported"
    )


def test_darwin_with_unknown_ram_unsupported() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=None)
        == "unsupported"
    )


def test_darwin_streams_below_full_resident_floor() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=15)
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=48)
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=84)
        == "streaming_models_loading"
    )


def test_darwin_full_resident_at_and_above_floor() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=85)
        == "full_models_loading"
    )
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=128)
        == "full_models_loading"
    )


def test_darwin_defaults_to_unsupported_without_mps_kwargs() -> None:
    """Backward-compat: existing call sites that don't pass mps_available/ram_gb stay unsupported."""
    assert decide_local_generation_mode(system="Darwin", cuda_available=True, vram_gb=64) == "unsupported"


def test_windows_without_cuda_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=False, vram_gb=24) == "unsupported"


def test_windows_with_low_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=14) == "unsupported"


def test_windows_with_unknown_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=None) == "unsupported"


def test_windows_streaming_range() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=15) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=24) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=30) == "streaming_models_loading"


def test_windows_full_loading_range() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=31) == "full_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=96) == "full_models_loading"


def test_linux_without_cuda_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=False, vram_gb=24) == "unsupported"


def test_linux_with_low_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=14) == "unsupported"


def test_linux_with_unknown_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=None) == "unsupported"


def test_linux_streaming_range() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=15) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=30) == "streaming_models_loading"


def test_linux_full_loading_range() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=31) == "full_models_loading"


def test_other_systems_fail_closed() -> None:
    assert decide_local_generation_mode(system="FreeBSD", cuda_available=True, vram_gb=48) == "unsupported"


def test_streaming_prefetch_count_for_full_loading_is_none() -> None:
    assert streaming_prefetch_count_for_mode("full_models_loading") is None


def test_streaming_prefetch_count_for_streaming_mode_is_two() -> None:
    assert streaming_prefetch_count_for_mode("streaming_models_loading") == 2


def test_streaming_prefetch_count_for_unsupported_asserts() -> None:
    with pytest.raises(AssertionError):
        streaming_prefetch_count_for_mode("unsupported")


def test_fast_auto_uses_mlx_only_for_cached_local_text_path() -> None:
    decision = decide_fast_video_engine(
        preference="auto",
        mlx_runtime_eligible=True,
        mlx_model_cached=True,
        use_local_text_encoding=True,
    )
    assert decision.engine == "mlx"


def test_fast_auto_preserves_prepared_embeddings_on_torch() -> None:
    decision = decide_fast_video_engine(
        preference="auto",
        mlx_runtime_eligible=True,
        mlx_model_cached=True,
        use_local_text_encoding=False,
    )
    assert decision.engine == "torch"
    assert "embeddings" in decision.reason


def test_fast_auto_does_not_trigger_hidden_model_download() -> None:
    decision = decide_fast_video_engine(
        preference="auto",
        mlx_runtime_eligible=True,
        mlx_model_cached=False,
        use_local_text_encoding=True,
    )
    assert decision.engine == "torch"
    assert "not cached" in decision.reason


def test_explicit_mlx_can_use_uncached_bf16_model() -> None:
    decision = decide_fast_video_engine(
        preference="mlx",
        mlx_runtime_eligible=True,
        mlx_model_cached=False,
        use_local_text_encoding=False,
    )
    assert decision.engine == "mlx"
    assert "download" in decision.reason


def test_mlx_bf16_uses_low_ram_below_eager_floor() -> None:
    assert (
        decide_fast_video_execution_mode("mlx", "streaming_models_loading", 48)
        == "low_ram"
    )
    assert (
        decide_fast_video_execution_mode("mlx", "streaming_models_loading", 64)
        == "eager"
    )


def test_fast_auto_never_selects_unqualified_q8() -> None:
    decision = decide_fast_video_engine(
        preference="auto",
        mlx_runtime_eligible=True,
        mlx_model_cached=True,
        use_local_text_encoding=True,
        mlx_quality_qualified=False,
    )
    assert decision.engine == "torch"
    assert "never auto-selected" in decision.reason
