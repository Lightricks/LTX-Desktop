"""CUDA patches + SageAttention SDPA wrap (gated). Call after CUDA alloc env is set."""

from __future__ import annotations

import logging
import os
from typing import Any, cast

logger = logging.getLogger(__name__)

_sageattention_runtime_fallback_logged = False
_sdpa_wrapped = False


def apply_patches(*, smart_app_control_on: bool = False, allow_sage_attention: bool = True) -> bool:
    """Apply Linux CUDA monkey-patches and optionally wrap SDPA with SageAttention.

    SageAttention stays gated on ``USE_SAGE_ATTENTION`` (default ``"1"``). Returns
    whether the wrap is enabled after import/setup (False if the env is off, Windows
    Smart App Control blocks it, or sageattention fails to load). A second call does
    not wrap SDPA again.
    """
    global _sdpa_wrapped
    if smart_app_control_on:
        # Before the patch imports below pull in ltx-core: its VAE kernels use Triton even
        # when SageAttention is off, and SAC blocks the .pyd files Triton builds.
        from inference.triton_block import block_triton_import

        if not block_triton_import():
            logger.warning(
                "Smart App Control is on but Triton was already imported; "
                "VAE kernels may still hit blocked Triton files."
            )
    # LLM: bumping ltx-core / ltx-pipelines (backend/pyproject.toml) MUST re-verify
    # every import below. These monkey-patch private ltx-core / ltx-pipelines /
    # safetensors symbols. Delete-first: drop a patch if upstream now includes the
    # fix (see that file's "Remove once ..." docstring). Callers must set
    # PYTORCH_CUDA_ALLOC_CONF before importing torch; Darwin mps_prebuilt_ext /
    # MPS_SDPA_SKIP_CALIBRATION must run before this function (patches transitively
    # import mps_sdpa).
    import inference.patches.safetensors_metadata_fix as _safetensors_metadata_fix  # pyright: ignore[reportUnusedImport]  # Remove once safetensors supports read-only mmap
    del _safetensors_metadata_fix
    import inference.patches.gemma_assets_positional_read as _gemma_assets_positional_read  # pyright: ignore[reportUnusedImport]  # Remove once GemmaAssets.from_single_file reads the packed text encoder without mmap and Windows generates without an access violation (LTX-Desktop#203)
    del _gemma_assets_positional_read
    import inference.patches.pinned_pool_fix as _pinned_pool_fix  # pyright: ignore[reportUnusedImport]  # Remove once ltx-core alloc_buffer does not report pinned-host failure as CUDA VRAM OOM (LTX-Desktop#141)
    del _pinned_pool_fix
    import inference.patches.diffusion_stage_cache as _diffusion_stage_cache  # pyright: ignore[reportUnusedImport]  # EXPERIMENTAL: remove once DiffusionStage caches/reuses identical builds upstream
    del _diffusion_stage_cache
    import inference.patches.diffvae_decode_vram as _diffvae_decode_vram  # pyright: ignore[reportUnusedImport]  # Remove once upstream decode tiling is confirmed sufficient (E1)
    del _diffvae_decode_vram
    import inference.patches.natten_libnatten_gate as _natten_libnatten_gate  # pyright: ignore[reportUnusedImport]  # Remove once ltx-core natten_available checks HAS_LIBNATTEN
    del _natten_libnatten_gate
    import inference.patches.diffusion_interrupt as _diffusion_interrupt  # pyright: ignore[reportUnusedImport]  # Remove once ltx-pipelines denoiser/loop accepts an interrupt callback
    del _diffusion_interrupt
    import inference.patches.nag_cross_attention as _nag_cross_attention  # pyright: ignore[reportUnusedImport]  # EXPERIMENTAL: remove once ltx-core supports a negative text context in the distilled path
    del _nag_cross_attention

    use_sage_attention = allow_sage_attention and os.environ.get("USE_SAGE_ATTENTION", "1") == "1"
    if not use_sage_attention:
        return False
    if smart_app_control_on:
        # SageAttention launches Triton kernels. Triton compiles unsigned .pyd files at
        # run time, which Smart App Control blocks. Plain torch SDPA needs no such code.
        logger.warning(
            "Windows Smart App Control is on and blocks the unsigned code that "
            "SageAttention and Triton build at run time. SageAttention and torch.compile "
            "are off; generation uses the default attention and runs slower."
        )
        return False
    if _sdpa_wrapped:
        return True

    try:
        from sageattention import sageattn  # type: ignore[reportMissingImports]
        import torch
        import torch.nn.functional as F

        _original_sdpa = F.scaled_dot_product_attention

        _SAGE_SUPPORTED_HEADDIMS = {64, 96, 128}

        @torch.compiler.disable(recursive=True)  # type: ignore[reportUntypedFunctionDecorator]
        def _sageattn_call(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool) -> torch.Tensor:
            # torch.compile must not trace into sageattn's Python internals (quant.py,
            # core.py): it bakes the current call's shapes into the compiled graph's
            # guards, which then hard-fails (ConstraintViolationError) on any later call
            # with different video/audio dimensions. Disabling keeps this call eager —
            # a graph break here, not a recompile — while the surrounding transformer
            # forward stays compiled.
            return cast(torch.Tensor, sageattn(query, key, value, is_causal=is_causal, tensor_layout="HND"))  # type: ignore[reportUnnecessaryCast]

        def patched_sdpa(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            global _sageattention_runtime_fallback_logged
            try:
                if (
                    query.dim() == 4
                    and attn_mask is None
                    and dropout_p == 0.0
                    and query.shape[-1] in _SAGE_SUPPORTED_HEADDIMS
                ):
                    return _sageattn_call(query, key, value, is_causal)
                else:
                    return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
            except Exception:
                if not _sageattention_runtime_fallback_logged:
                    logger.warning("SageAttention failed during runtime; falling back to default attention", exc_info=True)
                    _sageattention_runtime_fallback_logged = True
                return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

        F.scaled_dot_product_attention = patched_sdpa
        _sdpa_wrapped = True
        logger.info("SageAttention enabled - attention operations will be faster")
        return True
    except ImportError:
        logger.warning("SageAttention not installed - using default attention")
        return False
    except Exception:
        logger.warning("Failed to enable SageAttention", exc_info=True)
        return False
