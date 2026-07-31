"""Monkey-patch (EXPERIMENTAL): cache the built transformer across DiffusionStage
calls within one generation when nothing about its config actually changed.

``DiffusionStage`` "builds on each call, frees on exit" (ltx_pipelines/utils/
blocks.py) unconditionally -- it rebuilds the transformer from the checkpoint
file (disk read + fp8 cast + H2D copy) on *every* call and tears it down
(or ``.to("meta")`` + reclaim) on exit, regardless of whether the checkpoint/
LoRAs/quantization are identical to the previous call. For a two-stage
distilled pipeline where stage_1 and stage_2 build from the SAME checkpoint
with the SAME LoRAs (the common "fast" t2v/i2v case), that means loading +
fp8-casting a ~22B-param transformer from disk TWICE per generation. Measured
on an RTX 5090 (32 GB): ~40s per rebuild, ~80s of a ~132s 540p/8s generation
spent rebuilding the identical transformer twice.

Patches ``_transformer_ctx``, not the public ``model_context()`` wrapper:
``DiffusionStage.__call__`` invokes ``self._transformer_ctx(video_tools=...)``
directly (blocks.py:514), never ``model_context()`` -- an earlier version of
this patch wrapped the wrong method and silently never fired. ``model_context()``
itself just forwards to ``_transformer_ctx()``, so patching the latter covers
both call sites.

Scoped to the standard (non-streaming, non-multi-GPU) build path: caching only
applies when ``self._is_streaming`` is False (i.e. ``local_generations_mode ==
"full_models_loading"`` -- high-VRAM cards where holding one build resident is
affordable) AND the prepared builder is a plain ``SingleGPUModelBuilder``.
Streaming instances (the low-VRAM path) are untouched -- forcing residency
there would defeat the point of streaming. Multi-GPU tiled builders are also
excluded: they use ``video_tools`` to shard identically-shaped work across
devices, whereas ``SingleGPUModelBuilder.build()`` ignores all extra kwargs
(``**kwargs: object,  # noqa: ARG002`` in single_gpu_model_builder.py) --
confirmed inert for the path we cache, not assumed.

REUSE-SCOPED, not generation- or session-scoped: the cache exists only to bridge
the first and second identical diffusion stages. A live 720p Apple Silicon run
showed why the narrower lifetime matters: leaving the reused ~37 GiB transformer
resident through tiled VAE decode raised MPS driver memory to 46.7 GiB versus
44.5 GiB with the cache disabled, and left 35.4 GiB torch-allocated after output
encoding. The cache now evicts immediately when a hit's denoising context exits,
before decoder construction. The generation-start eviction remains a defensive
backstop for a pipeline that builds a cacheable first stage but never reaches an
identical second stage (cancellation, error, or a different stage configuration).

NON-CACHEABLE TRANSITIONS also evict (found live on the same RTX 5090, IC-LoRA
this time): IC-LoRA's ``use_lora_in_stage_2`` forces stage_2 onto the streaming
path (``LTXIcLoraPipeline._ensure_stage_2_streams_for_lora``, a deliberate
existing VRAM-safety mechanism -- streaming halves stage_2's resident
footprint since it conditions on the full-res reference video). That stage_2
call correctly skips the cache (``_is_streaming`` is True), but skipping the
cache branch also means the cache-key-mismatch eviction below never fires --
so stage_1's cached transformer stayed resident while stage_2 built its own
streaming transformer AND did its tiled conditioning VAE encode on top of it.
Observed: reserved VRAM climbed to 41.68 GB on the 31.82 GB card and hung
there for 150+ seconds with no progress (denoising loop never started).
Fix: the non-cacheable bypass branch (streaming stages, disabled setting)
evicts unconditionally before delegating to the original method, not just the
cache-hit/miss branch.

Single-slot cache: only the most recently built transformer stays resident.
Switching to a different checkpoint/LoRA/quantization config (or starting a
new generation, per above) evicts the old one (frees it, mirroring
gpu_model's own TRIM teardown: sync + dead-reference ``gc.collect()`` +
``.to("meta")`` + ``cleanup_memory()``) before building the new one -- never
holds two builds resident at once. The pre-free ``gc.collect()`` mirrors
ComfyUI's mandatory cleanup_models_gc: ``.to("meta")`` only frees storage once
no live reference (a compiled wrapper / cudagraph pool) remains, so collecting
dead references first is what turns a cumulative compile-path creep into a flat
reserved floor.

CONCURRENCY: the single-slot module-global cache is only correct under strict
sequentiality. That is enforced upstream by
``GenerationHandler.start_generation``/``start_api_generation``, which raise
"Generation already in progress" (under the shared state lock) for both the
GPU and API slots. As defense-in-depth for any future path that bypasses that
serialization, an ``_in_use`` counter marks a transformer checked out for the
duration of its ``yield`` (the denoising loop); ``_evict_locked`` raises rather
than freeing a model that is still in use. Note ``_lock`` guards the cache dict,
not the in-use model: it is released before ``yield``, so this counter -- not
the lock -- is what protects a mid-denoise transformer from a concurrent evict.

Cache key is the *content* of the prepared builder (checkpoint path, sd_ops,
module_ops, LoRAs) plus quantization identity, compilation config, dtype, and
device -- not object identity -- so two independently-constructed
DiffusionStage instances (e.g. a pipeline's stage_1 and stage_2) that happen to
build the same thing correctly share the cache within one generation.
``SDOps``/``ModuleOps`` are a frozen dataclass / NamedTuple respectively
(structural equality), so this is safe: a real config difference always
produces a different key (cache miss, falls back to a normal rebuild), never
a false hit.

Toggle: ``AppSettings.diffusion_stage_cache_enabled`` (default on, surfaced in
Settings next to Torch Compile). ``GenerationHandler.start_generation``/
``start_api_generation`` push the live setting into :func:`set_enabled` on
every generation, so flipping it in Settings takes effect on the next
generation without a restart. Also settable via env
``DIFFUSION_STAGE_CACHE_ENABLED`` (default "1") as the initial value before
any setting is pushed -- e.g. for headless/dev runs.

EXPERIMENTAL: depends on DiffusionStage's private ``_is_streaming``,
``_prepared_builder()``, ``_build_transformer()``, ``_quantization``,
``_compilation_config``, ``_dtype``, ``_device`` staying as-is, and on
``__call__`` continuing to route through ``_transformer_ctx`` -- re-verify
against ltx_pipelines.utils.blocks.DiffusionStage on rev bumps.

Usage:
    import services.patches.diffusion_stage_cache  # noqa: F401
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from ltx_core.devices import synchronize_device
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.helpers import cleanup_memory

logger = logging.getLogger(__name__)

_CacheKey = tuple[object, ...]

_lock = threading.Lock()
_enabled = os.environ.get("DIFFUSION_STAGE_CACHE_ENABLED", "1") != "0"
_cached_key: _CacheKey | None = None
_cached_model: object | None = None
# >0 while a cached transformer is checked out (yielded to a caller) and possibly
# mid-denoise. The single-slot cache is only safe under strict sequentiality; this
# counter lets _evict_locked() fail loud if something tries to free a model that is
# still in use, instead of ``.to("meta")``-ing tensors another generation is reading.
_in_use = 0


def set_enabled(value: bool) -> None:
    """Turn the cache on/off, checked on every ``_transformer_ctx`` call.

    Pushed from ``AppSettings.diffusion_stage_cache_enabled`` by
    ``GenerationHandler.start_generation``/``start_api_generation`` on every
    generation, so a Settings toggle takes effect on the next generation
    without a server restart. Turning off evicts immediately.
    """
    global _enabled
    with _lock:
        # Evict before flipping the flag: if _evict_locked raises (a transformer is
        # in use), the toggle stays un-applied rather than half-applied.
        if not value:
            _evict_locked()
        _enabled = value


def _cacheable(stage: DiffusionStage) -> bool:
    return not stage._is_streaming and isinstance(stage._prepared_builder(), SingleGPUModelBuilder)  # noqa: SLF001


def _cache_key(stage: DiffusionStage) -> _CacheKey:
    builder = stage._prepared_builder()  # noqa: SLF001
    return (
        builder.model_path,
        builder.model_sd_ops,
        builder.module_ops,
        builder.loras,
        # Policy objects aren't structurally comparable; identity is safe here
        # since pipelines construct one QuantizationPolicy and share it by
        # reference across stage_1/stage_2.
        id(stage._quantization),  # noqa: SLF001
        stage._compilation_config,  # noqa: SLF001
        stage._dtype,  # noqa: SLF001
        stage._device,  # noqa: SLF001
    )


def _evict_locked() -> None:
    global _cached_key, _cached_model
    if _in_use > 0:
        # Overlapping/concurrent generation: someone is trying to free a transformer
        # that is currently checked out (mid-denoise). The single-slot module-global
        # cache requires strict sequentiality, which GenerationHandler.start_generation
        # /start_api_generation already enforce (they raise "Generation already in
        # progress" under the shared state lock). This is defense-in-depth for any
        # future path that bypasses that serialization: fail loud instead of doing a
        # ``.to("meta")`` on tensors another generation is still reading.
        raise RuntimeError(
            "[diffusion-stage-cache] evict requested while a cached transformer is "
            "in use (mid-denoise) -- overlapping generation detected; the cache "
            "requires strict sequentiality."
        )
    if _cached_model is not None:
        synchronize_device()
        # Dead-reference GC BEFORE the meta-swap: a compiled wrapper or cudagraph
        # pool can still hold the module's storage, and ``.to("meta")`` only frees
        # storage once no live reference remains. ComfyUI runs the same mandatory
        # pre-free GC (cleanup_models_gc). This is the mechanism that decides whether
        # the compile-on soak shows a flat reserved floor (reclaim) or a slow creep
        # (leak) -- see the module docstring's GENERATION-SCOPED repro.
        gc.collect()
        _cached_model.to("meta")  # type: ignore[attr-defined]
        cleanup_memory()
    _cached_key, _cached_model = None, None


def evict() -> None:
    """Free and drop any resident cached transformer.

    Called from ``GenerationHandler.start_generation``/``start_api_generation``
    so the cache never survives past the generation it was built for -- see
    the module docstring's GENERATION-SCOPED section for why that matters.
    Safe to call even when nothing is cached (no-op) or when the patch is
    disabled (module-level cache is simply always empty). Raises if a cached
    transformer is currently in use (see :func:`_evict_locked`).
    """
    with _lock:
        _evict_locked()


def _release(*, evict_after_use: bool) -> None:
    """Release one checkout and optionally retire the now-consumed cache entry.

    A hit means the cache fulfilled its only purpose: carrying one identical
    transformer from the first diffusion stage into the second. Retire it while
    leaving that second context so VAE/audio decode cannot overlap its weights.
    """
    global _in_use
    with _lock:
        _in_use = max(0, _in_use - 1)
        if evict_after_use:
            _evict_locked()


_orig_transformer_ctx = DiffusionStage._transformer_ctx  # noqa: SLF001


@contextmanager
def _cached_transformer_ctx(self: DiffusionStage, **kwargs: object) -> Iterator[object]:
    if not _enabled or not _cacheable(self):
        # A non-cacheable build (e.g. IC-LoRA's use_lora_in_stage_2 forcing stage_2
        # onto the streaming path -- see module docstring's NON-CACHEABLE TRANSITIONS
        # section) needs the VRAM a still-resident cached transformer is holding.
        # Evicting only on a cache-key mismatch (below) never fires for this case,
        # since this path never touches the cache-key branch at all.
        evict()
        with _orig_transformer_ctx(self, **kwargs) as model:
            yield model
        return

    global _in_use
    key = _cache_key(self)
    with _lock:
        if _cached_key == key and _cached_model is not None:
            model = _cached_model
            hit = True
        else:
            _evict_locked()
            model = self._build_transformer(**kwargs)  # noqa: SLF001
            globals()["_cached_key"], globals()["_cached_model"] = key, model
            hit = False
        # Mark in use BEFORE releasing the lock (not after): _evict_locked runs only
        # under _lock, so bumping the counter here closes the window between resolving
        # the model and marking it -- otherwise a concurrent evict could see _in_use==0
        # and free the model we're about to yield. This counter (not the lock, which is
        # released before the yield) is what protects the model for the whole denoise.
        _in_use += 1

    logger.info("[diffusion-stage-cache] %s resident transformer", "reusing" if hit else "built + cached")
    try:
        yield model
    finally:
        _release(evict_after_use=hit)


DiffusionStage._transformer_ctx = _cached_transformer_ctx  # type: ignore[method-assign]  # noqa: SLF001


if __name__ == "__main__":
    a = ("ckpt.safetensors", None, (), (), 1, None, "bf16", "cuda:0")
    b = ("ckpt.safetensors", None, (), (), 1, None, "bf16", "cuda:0")
    c = ("ckpt.safetensors", None, (), (), 2, None, "bf16", "cuda:0")
    assert a == b, "identical content must compare equal (cache hit path)"
    assert a != c, "different quantization identity must compare unequal (cache miss path)"
    print("diffusion_stage_cache: key-equality self-check OK")
