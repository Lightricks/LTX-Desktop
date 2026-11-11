"""Tests for the diffusion_stage_cache patch (experiment/diffusion-stage-cache).

Uses the real (but cheap-to-construct, side-effect-free) SimpleBuilder /
StreamingBuilder classes from ltx_core as data holders, and a minimal fake
DiffusionStage duck-typing only the private surface diffusion_stage_cache.py reads
(_is_streaming, _prepared_builder, _build_transformer, _quantization,
_compilation_config, _dtype, _device), plus the ``_transformer_lifetime``
members the upstream ``_transformer_ctx`` reads when the cache is bypassed -- no mocking
library, no real GPU, no real checkpoint files. The cache's own internal functions are exercised directly rather
than through the real (monkeypatched) DiffusionStage._transformer_ctx, so these
tests do not depend on the patch actually being installed on the class.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from ltx_core.loader.simple_builder import LoRARequest, SimpleBuilder
from ltx_core.loader.streaming_builder import StreamingBuilder, StreamingConfig
from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.gpu_model import gpu_model
from services.patches import diffusion_stage_cache as dsc


class _FakeModel:
    def __init__(self) -> None:
        self.freed_to: str | None = None
        self.disposed = False

    def to(self, device: str) -> "_FakeModel":
        self.freed_to = device
        return self

    def dispose(self) -> None:
        self.disposed = True


class _FakeStage:
    """Duck-types the private DiffusionStage surface diffusion_stage_cache.py reads."""

    def __init__(
        self,
        builder: object,
        *,
        is_streaming: bool = False,
        quantization: object | None = None,
        compilation_config: object | None = None,
        dtype: str = "bf16",
        device: str = "cpu",
    ) -> None:
        self._builder = builder
        self._is_streaming = is_streaming
        self._quantization = quantization
        self._compilation_config = compilation_config
        self._dtype = dtype
        self._device = device
        self._model_wrapper = None
        self._alloc_trim_strategy = AllocatorTrimStrategy.TRIM
        self.build_count = 0

    def _prepared_builder(self) -> object:
        return self._builder

    def _build_transformer(self, **_kwargs: object) -> _FakeModel:
        self.build_count += 1
        return _FakeModel()

    def _transformer_lifetime(self, transformer: _FakeModel) -> AbstractContextManager[_FakeModel]:
        return gpu_model(transformer, alloc_trim_strategy=self._alloc_trim_strategy)


def _single_gpu_builder(model_path: str, loras: tuple[LoRARequest, ...] = ()) -> SimpleBuilder:
    return SimpleBuilder(object, (Path(model_path),), loras=loras)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _reset_cache_state():
    dsc.set_enabled(True)
    dsc.evict()
    yield
    dsc.set_enabled(True)
    dsc.evict()


def test_cache_hit_reuses_model_without_rebuilding() -> None:
    stage_1 = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    stage_2 = _FakeStage(_single_gpu_builder("ckpt.safetensors"))  # separate instance, identical content

    with dsc._cached_transformer_ctx(stage_1) as model_1:
        pass
    with dsc._cached_transformer_ctx(stage_2) as model_2:
        pass

    assert stage_1.build_count == 1
    assert stage_2.build_count == 0, "stage_2 should reuse stage_1's cached build"
    assert model_1 is model_2


def test_cache_miss_on_different_checkpoint_evicts_old_model() -> None:
    stage_a = _FakeStage(_single_gpu_builder("a.safetensors"))
    stage_b = _FakeStage(_single_gpu_builder("b.safetensors"))

    with dsc._cached_transformer_ctx(stage_a) as model_a:
        pass
    with dsc._cached_transformer_ctx(stage_b) as model_b:
        pass

    assert stage_a.build_count == 1
    assert stage_b.build_count == 1
    assert model_a.freed_to == "meta", "old model must be freed before building the new one"
    assert model_b is not model_a


def test_different_loras_are_treated_as_a_different_config() -> None:
    stage_a = _FakeStage(_single_gpu_builder("ckpt.safetensors", loras=()))
    stage_b = _FakeStage(_single_gpu_builder("ckpt.safetensors", loras=(LoRARequest((Path("lora.safetensors"),), 1.0),)))

    with dsc._cached_transformer_ctx(stage_a):
        pass
    with dsc._cached_transformer_ctx(stage_b):
        pass

    assert stage_a.build_count == 1
    assert stage_b.build_count == 1, "different LoRA set must miss the cache, not reuse stage_a's build"


def test_cacheable_returns_false_for_a_streaming_stage() -> None:
    streaming_builder = StreamingBuilder(
        _single_gpu_builder("ckpt.safetensors"), StreamingConfig("transformer_blocks", "disk")
    )
    streaming_stage = _FakeStage(streaming_builder, is_streaming=True)

    assert dsc._cacheable(streaming_stage) is False


def test_cacheable_skips_when_model_wrapper_is_set() -> None:
    stage = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    stage._model_wrapper = object()
    assert dsc._cacheable(stage) is False


class _OtherBuilder:
    """Stand-in for a non-SimpleBuilder, non-streaming builder (e.g. a multi-GPU tiled builder)."""

    paths = (Path("ckpt.safetensors"),)
    sd_ops: tuple[object, ...] = ()
    module_ops: tuple[object, ...] = ()
    loras: tuple[object, ...] = ()


def test_non_simple_builder_bypasses_cache_and_evicts_resident_model() -> None:
    cacheable_stage = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    with dsc._cached_transformer_ctx(cacheable_stage) as cached_model:
        pass
    assert dsc._cached_model is cached_model

    other_stage = _FakeStage(_OtherBuilder(), is_streaming=False)

    with dsc._cached_transformer_ctx(other_stage) as other_model:
        pass

    assert cached_model.freed_to == "meta", "resident cache must be freed before the non-cacheable build"
    assert dsc._cached_model is None
    assert other_stage.build_count == 1
    assert other_model is not cached_model


def test_disabled_bypasses_cache_and_evicts_immediately() -> None:
    stage = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    with dsc._cached_transformer_ctx(stage) as cached_model:
        pass
    assert dsc._cached_model is not None

    dsc.set_enabled(False)
    assert cached_model.freed_to == "meta"
    assert dsc._cached_model is None

    with dsc._cached_transformer_ctx(stage):
        pass
    with dsc._cached_transformer_ctx(stage):
        pass

    assert stage.build_count == 3, "every call rebuilds while disabled, none of them cache"
    assert dsc._cached_model is None


def test_evict_frees_and_clears_cache() -> None:
    stage = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    with dsc._cached_transformer_ctx(stage) as model:
        pass

    dsc.evict()

    assert model.freed_to == "meta"
    assert dsc._cached_model is None
    assert dsc._cached_key is None


def test_cache_key_self_check() -> None:
    """Mirrors the module's own __main__ self-check as a real test."""
    a = ("ckpt.safetensors", None, (), (), 1, None, "bf16", "cuda:0")
    b = ("ckpt.safetensors", None, (), (), 1, None, "bf16", "cuda:0")
    c = ("ckpt.safetensors", None, (), (), 2, None, "bf16", "cuda:0")
    assert a == b
    assert a != c


# --- hardening: dead-reference gc pass + in-use concurrency guard ------------ #


def test_evict_empty_cache_still_cleans_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second Home job starts encode with the cache already empty after DiffVAE evict.

    Skipping cleanup_memory in that case left Windows CUDA at reserved>>device
    and free=0; the next tiled encode never reached 'Building transformer'.
    """
    cleaned: list[bool] = []
    dsc.evict()
    monkeypatch.setattr(dsc, "cleanup_memory", lambda: cleaned.append(True))
    dsc.evict()
    assert cleaned == [True]


def test_evict_runs_gc_collect_before_meta_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-free gc.collect() must run BEFORE .to('meta') (ComfyUI ordering)."""
    stage = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    with dsc._cached_transformer_ctx(stage) as model:
        pass

    events: list[str] = []
    monkeypatch.setattr(dsc.gc, "collect", lambda *a, **k: events.append("gc"))
    original_to = model.to

    def _record_to(device: str) -> "_FakeModel":
        events.append(f"to:{device}")
        return original_to(device)

    monkeypatch.setattr(model, "to", _record_to)

    dsc.evict()

    # cleanup_memory() runs its own gc.collect() AFTER the meta-swap, so a trailing
    # "gc" is expected; assert only that the pre-free gc precedes the meta-swap.
    assert events[:2] == ["gc", "to:meta"], f"pre-free gc must precede the meta-swap, got {events}"


def test_in_use_is_tracked_across_the_yield() -> None:
    stage = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    assert dsc._in_use == 0
    with dsc._cached_transformer_ctx(stage):
        assert dsc._in_use == 1, "model is checked out for the duration of the yield"
    assert dsc._in_use == 0, "in-use must return to zero once the caller is done"


def test_evict_while_in_use_raises_and_does_not_free() -> None:
    stage = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    with dsc._cached_transformer_ctx(stage) as model:
        # Simulate a concurrent evict() (e.g. an overlapping generation) while the
        # transformer is mid-denoise. It must fail loud rather than free the model.
        with pytest.raises(RuntimeError, match="in use"):
            dsc.evict()
        assert model.freed_to is None, "model must NOT be freed while in use"
        assert dsc._cached_model is model, "cache must remain intact after a rejected evict"

    # Once the caller is done, eviction works normally again.
    dsc.evict()
    assert model.freed_to == "meta"
    assert dsc._cached_model is None


def test_set_enabled_false_while_in_use_raises() -> None:
    """Toggling the cache off mid-denoise also routes through the in-use guard."""
    stage = _FakeStage(_single_gpu_builder("ckpt.safetensors"))
    with dsc._cached_transformer_ctx(stage):
        with pytest.raises(RuntimeError, match="in use"):
            dsc.set_enabled(False)
        # Evict runs before the flag flips, so a rejected toggle stays un-applied.
        assert dsc._enabled is True, "failed toggle-off must not half-apply"


def test_patch_is_installed_on_a_real_stage_that_the_cache_accepts() -> None:
    """A rename of ``_transformer_ctx`` or of a private attribute the cache reads turns the cache off silently."""
    import torch
    from ltx_pipelines.utils.blocks import DiffusionStage

    stage = DiffusionStage.from_checkpoint("ckpt.safetensors", torch.bfloat16, torch.device("cpu"))
    twin = DiffusionStage.from_checkpoint("ckpt.safetensors", torch.bfloat16, torch.device("cpu"))

    assert DiffusionStage._transformer_ctx is dsc._cached_transformer_ctx  # noqa: SLF001
    assert dsc._cacheable(stage)  # noqa: SLF001
    assert dsc._cache_key(stage) == dsc._cache_key(twin)  # noqa: SLF001
