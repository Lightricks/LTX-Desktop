"""NF4 (4-bit BitsAndBytes) quantized LTX video pipeline.

Uses BitsAndBytes NF4 quantization to reduce the LTX transformer's VRAM
footprint from ~22 GB (BF16) to ~6 GB (NF4).  The base BF16 checkpoint
is loaded and the transformer's Linear layers are replaced with
``bitsandbytes.nn.Linear4bit`` before being moved to GPU — the
quantization happens automatically during the ``.to(device)`` call.

Requires:
  - ``bitsandbytes`` package (``pip install bitsandbytes``)
  - Base BF16 checkpoint in the same directory or auto-discoverable
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType

logger = logging.getLogger(__name__)

_BASE_CHECKPOINT_NAME = "ltx-2.3-22b-distilled.safetensors"


def _find_base_checkpoint(nf4_path: str, upsampler_path: str) -> str:
    """Locate the base BF16 safetensors checkpoint for all components.

    NF4 runtime quantization starts from the BF16 weights and quantizes
    on the fly.  Searches same directory as the NF4 folder, then models_dir.
    """
    search_dirs = [
        Path(nf4_path).parent,
        Path(upsampler_path).parent,
    ]
    for d in search_dirs:
        candidate = d / _BASE_CHECKPOINT_NAME
        if candidate.is_file():
            return str(candidate)

    # Fallback: largest safetensors in models dir
    for d in search_dirs:
        safetensors = sorted(d.glob("*.safetensors"), key=lambda p: p.stat().st_size, reverse=True)
        for sf in safetensors:
            if sf.stat().st_size > 10_000_000_000:
                return str(sf)

    raise FileNotFoundError(
        f"Cannot find base BF16 checkpoint ({_BASE_CHECKPOINT_NAME}) near {nf4_path}. "
        "The base model is required for NF4 runtime quantization. "
        "Download it from the Models tab in Settings."
    )


def _apply_nf4_quantization(model: torch.nn.Module) -> None:
    """Replace all ``nn.Linear`` layers in *model* with ``bnb.nn.Linear4bit`` (NF4).

    Weights are stored as ``Params4bit`` objects that quantize to NF4 when
    moved to a CUDA device.  The replacement happens in-place.
    """
    import bitsandbytes as bnb  # type: ignore[import-untyped]

    replacements: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            nf4_linear = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
            )
            # Store BF16 weights as Params4bit — quantized on .to(cuda)
            nf4_linear.weight = bnb.nn.Params4bit(
                module.weight.data.to(torch.bfloat16),
                requires_grad=False,
                quant_type="nf4",
            )
            if module.bias is not None:
                nf4_linear.bias = module.bias

            # Find parent module to set attribute
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, child_name = parts
            else:
                parent_name, child_name = "", parts[0]

            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, child_name, nf4_linear))

    for parent, child_name, new_module in replacements:
        setattr(parent, child_name, new_module)

    n_replaced = len(replacements)
    logger.info("Replaced %d Linear layers with NF4 Linear4bit", n_replaced)


class NF4FastVideoPipeline:
    """FastVideoPipeline that quantizes the transformer to NF4 at runtime.

    Uses the ``DistilledPipeline`` for two-stage video generation but
    overrides the transformer loading to:
      1. Build the BF16 model on CPU
      2. Replace all ``nn.Linear`` with ``bnb.nn.Linear4bit`` (NF4)
      3. Move to GPU — this triggers 4-bit quantization

    Peak CPU RAM: ~43 GB (full BF16 model during build)
    Peak GPU VRAM: ~6 GB (NF4 transformer) + ~3 GB (VAE) = ~9 GB
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> "NF4FastVideoPipeline":
        return NF4FastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            lora_path=lora_path,
            lora_weight=lora_weight,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> None:
        try:
            import bitsandbytes  # noqa: F401  # pyright: ignore[reportUnusedImport,reportMissingImports]
        except ImportError:
            raise RuntimeError(
                "NF4 quantization requires the 'bitsandbytes' package. "
                "Install it with: pip install bitsandbytes"
            ) from None

        # checkpoint_path is the NF4 folder (from model scanner selection).
        # We need the base BF16 model for actual loading + runtime quantization.
        base_checkpoint = _find_base_checkpoint(checkpoint_path, upsampler_path)
        logger.info("NF4 pipeline: base model from %s, runtime NF4 quantization", base_checkpoint)

        self._device = device

        # Build DistilledPipeline with the base BF16 model.
        # LoRA is passed normally — the BF16 model builds with fused LoRA,
        # then we quantize the result to NF4.
        from ltx_pipelines.distilled import DistilledPipeline

        lora_entries: list[Any] = []
        if lora_path:
            from ltx_core.loader.primitives import LoraPathStrengthAndSDOps  # pyright: ignore[reportMissingImports]

            sd_ops: Any = None
            try:
                import importlib
                _ser = importlib.import_module("ltx_core.loader.serialization")
                sd_ops = getattr(_ser, "LTXV_LORA_COMFY_RENAMING_MAP", None)
            except (ImportError, AttributeError):
                pass

            lora_entries = [LoraPathStrengthAndSDOps(
                path=lora_path,
                strength=lora_weight,
                sd_ops=sd_ops,
            )]

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=base_checkpoint,
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=lora_entries,
            device=device,
            quantization=None,  # We handle quantization ourselves
        )

        # Override model_ledger.transformer() to build BF16 on CPU then quantize to NF4
        self._original_transformer = self.pipeline.model_ledger.transformer
        self.pipeline.model_ledger.transformer = self._build_nf4_transformer  # type: ignore[assignment]

    def _build_nf4_transformer(self) -> Any:
        """Build the transformer with NF4 quantization."""
        # Temporarily redirect model building to CPU
        original_device = self.pipeline.model_ledger.device
        self.pipeline.model_ledger.device = torch.device("cpu")

        try:
            # Build full BF16 model on CPU (uses ~43GB CPU RAM)
            model = self._original_transformer()
            logger.info("Built BF16 transformer on CPU, applying NF4 quantization...")
        finally:
            self.pipeline.model_ledger.device = original_device

        # Replace Linear layers with NF4
        # model is an X0Model wrapping the actual transformer
        inner_model = model.model if hasattr(model, "model") else model
        _apply_nf4_quantization(inner_model)

        # Move to GPU — triggers NF4 quantization of Params4bit
        logger.info("Moving NF4 transformer to %s", self._device)
        return model.to(self._device).eval()

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            tiling_config=tiling_config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        logger.info("Skipping torch.compile for NF4 pipeline — not supported with quantized weights")
