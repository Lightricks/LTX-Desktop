"""Local Gemma-based prompt enhancer.

Builds the Gemma text encoder fresh for a single enhance call and frees it — matching every
other local Gemma usage in this codebase (``ltx_pipelines.utils.blocks.PromptEncoder.__call__``
loads/frees it per text-encoding call too; none of them keep it resident). The full
generation model (including ``lm_head``) is what gets built here, which is why direct
``generate()``-based enhancement is available at no extra VRAM cost over plain encoding.

The root passed in is whichever checkpoint can actually generate for the active model, which is
not always the one that encodes it: 2.3's Gemma 3 does both, while 2.5 encodes with an
encode-only ``gemma4_unified`` and enhances with Gemma 4 E2B when present, else Gemma 3 if a
2.3 install already put it on disk. Everything below is model-type agnostic — ``get_gemma_ops``
and the encoder's own defaults resolve the differences — so both roots load through the same
path.

Generation is reimplemented here rather than calling upstream
``GemmaTextEncoder.enhance_t2v``/``enhance_i2v`` — those go through
``_pad_inputs_for_attention_alignment``, which right-pads the prompt to a multiple of 8 for
Flash Attention *before* calling ``.generate()``. Right-padding a decoder-only model's prompt
before generation is a known way to get garbled/misplaced output, and real testing against this
exact checkpoint reproduced that failure shape consistently (a dangling, subject-less fragment
prepended before the real completion). This mirrors ``_enhance()`` exactly, minus that one
padding step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder


def _generation_kwargs(text_encoder: "LTXGemmaTextEncoder") -> dict[str, Any]:
    # Decoding settings differ per enhancer family — Gemma 3 samples at 0.7, Gemma 4 instruct is
    # greedy with an n-gram block — and only the encoder knows which one it loaded.
    return cast(dict[str, Any], cast(Any, text_encoder)._default_generation_kwargs())


def _generate(
    text_encoder: "LTXGemmaTextEncoder",
    messages: list[dict[str, object]],
    image: "torch.Tensor | None",
    seed: int,
) -> str:
    # transformers' Gemma3Processor/Gemma3ForConditionalGeneration stubs don't type these
    # dynamically-attached attributes precisely enough for strict mode; loosen locally rather
    # than suppress line-by-line, matching this codebase's other upstream-internals call sites
    # (e.g. services/text_encoder/ltx_text_encoder.py's PromptEncoder patches).
    encoder = cast(Any, text_encoder)
    assert encoder.processor is not None

    text = encoder.processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = encoder.processor(text=text, images=image, return_tensors="pt").to(encoder.model.device)

    fork_devices = [encoder.model.device] if encoder.model.device.type == "cuda" else []
    with torch.inference_mode(), torch.random.fork_rng(devices=fork_devices):  # type: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # type: ignore[reportUnknownMemberType]
        outputs = encoder.model.generate(**model_inputs, **_generation_kwargs(text_encoder))
        generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
        return cast(str, encoder.processor.tokenizer.decode(generated_ids, skip_special_tokens=True))


def _default_system_prompt(text_encoder: "LTXGemmaTextEncoder", *, t2v: bool) -> str:
    # LTXGemmaTextEncoder no longer exposes this as a public property (renamed from
    # GemmaTextEncoder); its model-type-aware default lives on the private
    # ``_default_system_prompt`` method, so reach into it like the other upstream-internals
    # call sites in this module.
    return cast(Any, text_encoder)._default_system_prompt(t2v=t2v)


class LtxPromptEnhancerPipeline:
    @staticmethod
    def create(gemma_root: str, device: str) -> "LtxPromptEnhancerPipeline":
        return LtxPromptEnhancerPipeline(gemma_root=gemma_root, device=device)

    def __init__(self, gemma_root: str, device: str) -> None:
        self._gemma_root = gemma_root
        self._device = device

    def _build_text_encoder(self) -> "LTXGemmaTextEncoder":
        # Mirrors ltx_pipelines.utils.blocks.PromptEncoder's non-streaming text-encoder build,
        # minus the embeddings-processor half — enhancement never needs video/audio embeddings.
        from ltx_core.loader.registry import DummyRegistry
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.text_encoders.gemma import (
            GemmaTextEncoderConfigurator,
            get_gemma_ops,
            resolve_gemma_weight_paths,
        )

        sd_ops, module_ops = get_gemma_ops(self._gemma_root)
        weight_paths = resolve_gemma_weight_paths(self._gemma_root)
        builder = SingleGPUModelBuilder(
            model_path=weight_paths,
            model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(self._gemma_root),
            model_sd_ops=sd_ops,
            module_ops=module_ops,
            registry=DummyRegistry(),
        )
        return builder.build(device=torch.device(self._device), dtype=torch.bfloat16).eval()

    def enhance_t2v(self, prompt: str, system_prompt: str | None, seed: int) -> str:
        from ltx_pipelines.utils.gpu_model import gpu_model

        with gpu_model(self._build_text_encoder()) as text_encoder:
            resolved_system_prompt = system_prompt or _default_system_prompt(text_encoder, t2v=True)
            messages: list[dict[str, object]] = [
                {"role": "system", "content": resolved_system_prompt},
                {"role": "user", "content": f"user prompt: {prompt}"},
            ]
            return _generate(text_encoder, messages, image=None, seed=seed)

    def enhance_i2v(self, prompt: str, image_path: str, system_prompt: str | None, seed: int) -> str:
        from ltx_pipelines.utils.gpu_model import gpu_model
        from ltx_pipelines.utils.media_io import decode_image, resize_aspect_ratio_preserving

        image = decode_image(image_path=image_path)
        image_tensor = resize_aspect_ratio_preserving(torch.tensor(image), 896).to(torch.uint8)

        with gpu_model(self._build_text_encoder()) as text_encoder:
            resolved_system_prompt = system_prompt or _default_system_prompt(text_encoder, t2v=False)
            messages: list[dict[str, object]] = [
                {"role": "system", "content": resolved_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                    ],
                },
            ]
            return _generate(text_encoder, messages, image=image_tensor, seed=seed)
