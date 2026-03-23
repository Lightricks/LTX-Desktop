"""GGUF quantized LTX video pipeline.

Loads LTX-Video transformer weights from a GGUF file using the diffusers
GGUFLinear layers (on-the-fly dequantization), while loading VAE, text
encoder, audio models, and upsampler from the base BF16 checkpoint.

The base BF16 checkpoint must be present in the same directory as the
GGUF file (auto-discovered) because the GGUF file only contains the
transformer weights.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CHECKPOINT_NAME = "ltx-2.3-22b-distilled.safetensors"


def _find_base_checkpoint(gguf_path: str, upsampler_path: str) -> str:
    """Locate the base BF16 safetensors checkpoint for non-transformer components.

    Searches in:
      1. Same directory as the GGUF file
      2. Parent directory of the upsampler (models_dir)
    """
    search_dirs = [
        Path(gguf_path).parent,
        Path(upsampler_path).parent,
    ]
    for d in search_dirs:
        candidate = d / _BASE_CHECKPOINT_NAME
        if candidate.is_file():
            return str(candidate)

    # Fallback: largest safetensors file in the models directory
    for d in search_dirs:
        safetensors = sorted(d.glob("*.safetensors"), key=lambda p: p.stat().st_size, reverse=True)
        for sf in safetensors:
            if sf.stat().st_size > 10_000_000_000:  # > 10 GB — likely the base model
                return str(sf)

    raise FileNotFoundError(
        f"Cannot find base BF16 checkpoint ({_BASE_CHECKPOINT_NAME}) near {gguf_path}. "
        "The base model is required for VAE, audio, and text encoder components. "
        "Download it from the Models tab in Settings."
    )


def _read_gguf_state_dict(gguf_path: str) -> dict[str, torch.nn.Parameter]:
    """Read a GGUF file and return a state dict of GGUFParameter/Tensor objects."""
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter

    UNQUANTIZED = {
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    }

    logger.info("Reading GGUF file: %s", gguf_path)
    reader = gguf.GGUFReader(gguf_path)

    state_dict: dict[str, torch.nn.Parameter] = {}
    for tensor in reader.tensors:
        name = str(tensor.name)
        data: np.ndarray[Any, Any] = tensor.data  # type: ignore[assignment]
        quant_type = tensor.tensor_type

        if quant_type in UNQUANTIZED:
            # Unquantized — convert to proper dtype
            if quant_type == gguf.GGMLQuantizationType.F32:
                t = torch.from_numpy(data.copy()).to(torch.float32)
            elif quant_type == gguf.GGMLQuantizationType.F16:
                t = torch.from_numpy(data.copy()).to(torch.float16)
            else:  # BF16
                t = torch.from_numpy(data.copy()).view(torch.bfloat16)
            # Reshape to original shape
            shape = tuple(int(s) for s in tensor.shape)
            if shape:
                t = t.reshape(shape)
            state_dict[name] = torch.nn.Parameter(t, requires_grad=False)
        else:
            # Quantized — wrap as GGUFParameter for on-the-fly dequantization
            t = torch.from_numpy(data.copy())
            param = GGUFParameter(t, requires_grad=False, quant_type=quant_type)
            state_dict[name] = param

    logger.info("Read %d tensors from GGUF file", len(state_dict))
    return state_dict


def _remap_gguf_keys(state_dict: dict[str, torch.nn.Parameter]) -> dict[str, torch.nn.Parameter]:
    """Strip common prefixes from GGUF tensor names to match ltx_core model keys.

    City96 GGUF files use ``model.diffusion_model.`` prefix; ltx_core expects
    keys without that prefix.  Also handles ``diffusion_model.`` prefix.
    """
    PREFIXES_TO_STRIP = [
        "model.diffusion_model.",
        "diffusion_model.",
    ]

    remapped: dict[str, torch.nn.Parameter] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in PREFIXES_TO_STRIP:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        remapped[new_key] = value

    return remapped


def _create_gguf_transformer(
    state_dict: dict[str, torch.nn.Parameter],
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Create an X0Model transformer with GGUFLinear layers and load quantized weights."""
    from diffusers.quantizers.gguf.utils import GGUFParameter, _replace_with_gguf_linear
    from ltx_core.model.transformer import LTXModelConfigurator, X0Model

    # Build model config with defaults for LTX-Video 2.3 22B
    # The GGUF file doesn't contain model config metadata, so we use defaults
    # which match the 22B distilled model architecture.
    default_config: dict[str, Any] = {}

    with torch.device("meta"):
        model = LTXModelConfigurator.from_config(default_config)

    # Replace nn.Linear with GGUFLinear where state dict has GGUFParameter
    _replace_with_gguf_linear(model, compute_dtype, state_dict)

    # Load state dict — assign=True replaces meta tensors with real data
    missing, unexpected = model.load_state_dict(
        {k: v for k, v in state_dict.items()},
        strict=False,
        assign=True,
    )
    if missing:
        logger.warning("GGUF transformer load — missing keys: %s", missing[:10])
    if unexpected:
        logger.debug("GGUF transformer load — unexpected keys: %s", unexpected[:10])

    x0_model = X0Model(model)
    return x0_model.to(device).eval()


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class GGUFFastVideoPipeline:
    """FastVideoPipeline that loads the transformer from a GGUF file.

    Uses the ``DistilledPipeline`` from ltx_pipelines for two-stage video
    generation (half-res → upsample+refine), but overrides the transformer
    loading to use GGUF quantized weights with on-the-fly dequantization
    via diffusers' ``GGUFLinear`` layers.

    Requires the base BF16 checkpoint to be present for VAE, audio, and
    text encoder components.

    Note: LoRA fusion with GGUF quantized weights is not yet supported.
    The distilled LoRA must be baked into the GGUF file.
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
    ) -> "GGUFFastVideoPipeline":
        return GGUFFastVideoPipeline(
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
            import gguf  # noqa: F401  # pyright: ignore[reportUnusedImport]
        except ImportError:
            raise RuntimeError(
                "GGUF model support requires the 'gguf' package. "
                "Install it with: pip install gguf>=0.10.0"
            ) from None

        gguf_path = checkpoint_path
        base_checkpoint = _find_base_checkpoint(gguf_path, upsampler_path)
        logger.info("GGUF pipeline: transformer from %s, base model from %s", gguf_path, base_checkpoint)

        # Pre-read and remap the GGUF state dict (stays on CPU, cached for reuse)
        raw_sd = _read_gguf_state_dict(gguf_path)
        self._gguf_state_dict = _remap_gguf_keys(raw_sd)
        self._device = device

        # Build DistilledPipeline with the base BF16 checkpoint for all non-transformer components.
        # LoRA is NOT passed here because it can't be fused with GGUF weights.
        from ltx_pipelines.distilled import DistilledPipeline

        if lora_path:
            logger.warning(
                "LoRA fusion with GGUF quantized weights is not supported. "
                "The LoRA at %s will be ignored. Use a GGUF file that includes the LoRA baked in.",
                lora_path,
            )

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=base_checkpoint,
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=[],
            device=device,
            quantization=None,
        )

        # Override model_ledger.transformer() to load from GGUF instead of safetensors
        self.pipeline.model_ledger.transformer = self._build_gguf_transformer  # type: ignore[assignment]

    def _build_gguf_transformer(self) -> Any:
        """Build the GGUF transformer model on each call (matches ModelLedger contract)."""
        return _create_gguf_transformer(
            state_dict=self._gguf_state_dict,
            device=self._device,
        )

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
        logger.info("Skipping torch.compile for GGUF pipeline — not supported with quantized weights")
