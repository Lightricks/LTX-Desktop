"""Stable-budget decode-tiling retry: pure budget math and refusal matching."""

from __future__ import annotations

import torch

from services.ltx_pipeline_common import is_decode_tile_shortage, stable_cuda_decode_budget_bytes

_GIB = 1024**3


def test_stable_budget_binds_the_fraction_cap_on_an_idle_card() -> None:
    """A 0.5 fraction on a 32 GiB card budgets 16, not the 28 GiB sitting free —
    forgetting the fraction resolves tiles for the whole card under a cap."""
    assert stable_cuda_decode_budget_bytes(
        fraction=0.5,
        device_total_bytes=32 * _GIB,
        raw_free_bytes=20 * _GIB,
        reserved_bytes=8 * _GIB,
    ) == 16 * _GIB


def test_stable_budget_binds_unheld_vram_when_a_second_process_holds_memory() -> None:
    """Uncapped card with a neighbor holding 22 GiB: raw free 4 plus our 6 reserved
    budgets 10 — our pool frees at evict, the neighbor's never does. Ignoring
    contention (or subtracting our pool momentary-style) over- or under-admits."""
    assert stable_cuda_decode_budget_bytes(
        fraction=1.0,
        device_total_bytes=32 * _GIB,
        raw_free_bytes=4 * _GIB,
        reserved_bytes=6 * _GIB,
    ) == 10 * _GIB


def test_stable_budget_floors_at_zero_on_degenerate_reads() -> None:
    """A zero fraction, or a corrupt negative free read, budgets 0 — never negative.
    A dropped floor would hand the resolver a negative budget."""
    assert stable_cuda_decode_budget_bytes(
        fraction=0.0,
        device_total_bytes=32 * _GIB,
        raw_free_bytes=20 * _GIB,
        reserved_bytes=8 * _GIB,
    ) == 0
    assert stable_cuda_decode_budget_bytes(
        fraction=0.5,
        device_total_bytes=32 * _GIB,
        raw_free_bytes=-5 * _GIB,
        reserved_bytes=1 * _GIB,
    ) == 0


def test_shortage_matcher_fires_only_on_the_cannot_fit_refusal() -> None:
    """Only the resolver's cannot-fit message retries — a broad match would burn a
    full second attempt (and mask the real error) on geometry/checkpoint bugs."""
    assert is_decode_tile_shortage(
        ValueError(
            "Cannot fit a DiffVAE decode tile under the memory budget: "
            "min tile ~80f x 320x320px (overlaps T=40, HW=160), mode=chunked_eager, "
            "keyframes=False, coef=5, stage5_channels=256, "
            "stage4_feature_bytes=7620526080, usable_bytes=6155852190. "
            "Reduce resolution, reduce num_frames (stage-4 features and the per-chunk "
            "encode buffers both scale with it), or free GPU memory."
        )
    )
    assert not is_decode_tile_shortage(ValueError("tile_size must be positive, got 0"))
    assert not is_decode_tile_shortage(ValueError("could not read VAE checkpoint metadata"))


def test_shortage_matcher_rejects_non_value_errors() -> None:
    """A CUDA OOM (or any non-ValueError) must not retry as a tiling shortage: the
    OOM handler owns it, and a second attempt would just OOM again minutes later."""
    assert not is_decode_tile_shortage(torch.cuda.OutOfMemoryError("CUDA out of memory."))
    assert not is_decode_tile_shortage(RuntimeError("CUDA out of memory."))
    assert not is_decode_tile_shortage(TypeError("device must be cuda"))
