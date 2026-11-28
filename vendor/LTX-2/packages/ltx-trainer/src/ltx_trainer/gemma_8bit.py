# ruff: noqa: PLC0415

"""
8-bit quantization for the Gemma text encoder.
Quantizing the Gemma backbone to LLM.int8() roughly halves the VRAM it occupies, which
matters because the encoder only runs long enough to precompute caption embeddings and
then competes with the transformer and optimizer state for the same GPU.
The encoder is built first through ltx-core's normal loading path and quantized here,
rather than being loaded by ``transformers`` directly. Going through ltx-core is what
makes every supported model work: LTX 2.5 ships a ``gemma4_unified`` text encoder, a
model type that only ltx-core's builder knows how to assemble, and it ships it as a
single packed ``.safetensors`` rather than a Hugging Face directory. Both are invisible
to ``AutoModelForImageTextToText.from_pretrained``.
The cost of that ordering is that the full-precision weights are materialized before
being quantized. They are materialized on CPU, so the peak is host RAM, not VRAM.
Example usage:
    from ltx_trainer.model_loader import load_text_encoder
    text_encoder = load_text_encoder("/path/to/gemma", device="cuda", load_in_8bit=True)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

import torch

from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder
from ltx_trainer import logger

# Outlier threshold from the LLM.int8() paper: columns whose activations exceed it are
# kept in fp16 and the rest go through the int8 matmul. Matches the default that
# ``BitsAndBytesConfig(load_in_8bit=True)`` applies as ``llm_int8_threshold``.
_LLM_INT8_THRESHOLD = 6.0

# The LM head is left in full precision, as it is everywhere else in the ecosystem: it is
# a single wide layer whose quantization error lands directly on output logits. Caption
# precompute does not run it at all (``LTXGemmaTextEncoder.encode`` calls the inner model),
# so this only affects prompt enhancement.
_KEEP_IN_FULL_PRECISION = frozenset({"lm_head"})


def resolve_8bit_device(device: torch.device | str | int | None) -> torch.device:
    """Pick the device an 8-bit encoder should be quantized onto.
    Defaults to this process's ``LOCAL_RANK`` so that a multi-process launch puts each
    rank's encoder on its own GPU rather than colliding on ``cuda:0``.
    """
    if device is not None:
        return torch.device(device) if not isinstance(device, torch.device) else device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    return torch.device("cpu")


def quantize_gemma_to_8bit(
    text_encoder: LTXGemmaTextEncoder,
    device: torch.device | str | int | None = None,
) -> LTXGemmaTextEncoder:
    """Quantize a built text encoder's Gemma backbone to 8-bit and move it to ``device``.
    Args:
        text_encoder: Encoder built by :func:`ltx_trainer.model_loader.load_text_encoder`,
            expected to still be on CPU.
        device: Target device. Defaults to this rank's GPU.
    Returns:
        The same encoder, with its Linear layers replaced by 8-bit equivalents.
    Raises:
        ImportError: If bitsandbytes is not installed or its CUDA binary is unusable.
    """
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "8-bit text encoder loading requires bitsandbytes. Install it with: uv pip install bitsandbytes"
        ) from e

    if text_encoder.model is None:
        raise ValueError("Cannot quantize a text encoder whose Gemma backbone has not been loaded.")

    target_device = resolve_8bit_device(device)
    if target_device.type != "cuda":
        raise ValueError(
            f"8-bit Gemma quantization requires a CUDA device, got {target_device}. "
            "Disable load_text_encoder_in_8bit when training on CPU."
        )

    replaced = _replace_linear_with_8bit(text_encoder.model, bnb)
    logger.debug(f"Quantized {replaced} Gemma Linear layers to 8-bit")

    # bitsandbytes quantizes lazily: Int8Params holds the full-precision weights until it
    # is moved onto CUDA, and does the int8 conversion during that move.
    with _suppress_accelerate_memory_warnings():
        text_encoder.model.to(target_device)

    return text_encoder


def _replace_linear_with_8bit(module: torch.nn.Module, bnb: object, prefix: str = "") -> int:
    """Swap every ``nn.Linear`` under ``module`` for its 8-bit counterpart, in place.
    Returns the number of layers replaced.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        qualified = f"{prefix}.{name}" if prefix else name
        if name in _KEEP_IN_FULL_PRECISION:
            continue
        if type(child) is torch.nn.Linear:
            setattr(module, name, _to_8bit_linear(child, bnb))
            replaced += 1
        else:
            replaced += _replace_linear_with_8bit(child, bnb, qualified)
    return replaced


def _to_8bit_linear(linear: torch.nn.Linear, bnb: object) -> torch.nn.Module:
    """Build an 8-bit Linear carrying ``linear``'s weights, still unquantized."""
    replacement = bnb.nn.Linear8bitLt(  # type: ignore[attr-defined]
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        has_fp16_weights=False,
        threshold=_LLM_INT8_THRESHOLD,
    )
    # int8_vectorwise_quant, which runs on the move to CUDA, expects fp16 input.
    replacement.weight = bnb.nn.Int8Params(  # type: ignore[attr-defined]
        linear.weight.data.to(torch.float16),
        requires_grad=False,
        has_fp16_weights=False,
    )
    if linear.bias is not None:
        replacement.bias = torch.nn.Parameter(linear.bias.data.to(torch.float16), requires_grad=False)
    return replacement


@contextmanager
def _suppress_accelerate_memory_warnings() -> Generator[None, None, None]:
    """Temporarily suppress INFO warnings from accelerate about memory allocation."""
    accelerate_logger = logging.getLogger("accelerate.utils.modeling")
    old_level = accelerate_logger.level
    accelerate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        accelerate_logger.setLevel(old_level)
