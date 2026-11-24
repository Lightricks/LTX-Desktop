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

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder
    from services.prompt_enhancement.i2v_frames import KeyframeStill, SceneFrame, VideoRole


def _generation_kwargs(text_encoder: "LTXGemmaTextEncoder") -> dict[str, Any]:
    # Decoding settings differ per enhancer family — Gemma 3 samples at 0.7, Gemma 4 instruct is
    # greedy with an n-gram block — and only the encoder knows which one it loaded.
    return cast(dict[str, Any], cast(Any, text_encoder)._default_generation_kwargs())


def _generate(
    text_encoder: "LTXGemmaTextEncoder",
    messages: list[dict[str, object]],
    image: "torch.Tensor | list[torch.Tensor] | None",
    seed: int,
    max_new_tokens_floor: int | None = None,
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
        kwargs = dict(_generation_kwargs(text_encoder))
        if max_new_tokens_floor is not None:
            kwargs["max_new_tokens"] = max(int(kwargs.get("max_new_tokens", 0)), max_new_tokens_floor)
        outputs = encoder.model.generate(**model_inputs, **kwargs)
        generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
        return cast(str, encoder.processor.tokenizer.decode(generated_ids, skip_special_tokens=True))


def _default_system_prompt(text_encoder: "LTXGemmaTextEncoder", *, t2v: bool) -> str:
    # LTXGemmaTextEncoder no longer exposes this as a public property (renamed from
    # GemmaTextEncoder); its model-type-aware default lives on the private
    # ``_default_system_prompt`` method, so reach into it like the other upstream-internals
    # call sites in this module.
    return cast(Any, text_encoder)._default_system_prompt(t2v=t2v)


def _image_tensor(path: str) -> torch.Tensor:
    from ltx_pipelines.utils.media_io import decode_image, resize_aspect_ratio_preserving

    image = decode_image(image_path=path)
    return resize_aspect_ratio_preserving(torch.tensor(image), 896).to(torch.uint8)


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
        from ltx_core.loader.simple_builder import SimpleBuilder
        from ltx_core.text_encoders.gemma import (
            GemmaTextEncoderConfigurator,
            get_gemma_ops,
            resolve_gemma_weight_paths,
        )

        sd_ops, module_ops = get_gemma_ops(self._gemma_root)
        weight_paths = resolve_gemma_weight_paths(self._gemma_root)
        builder = SimpleBuilder(
            GemmaTextEncoderConfigurator.with_gemma_model_path(self._gemma_root),
            tuple(Path(path) for path in weight_paths),
            (sd_ops,),
            module_ops,
            registry=DummyRegistry(),
        )
        return builder.build(device=torch.device(self._device), dtype=torch.bfloat16).eval()

    def enhance_t2v(
        self,
        prompt: str,
        system_prompt: str | None,
        seed: int,
        negative_prompt: str = "",
    ) -> str:
        from ltx_pipelines.utils.gpu_model import gpu_model

        from services.prompt_enhancement import (
            with_negative_prompt_rules,
            with_user_negative_prompt,
        )

        with gpu_model(self._build_text_encoder()) as text_encoder:
            resolved_system_prompt = with_negative_prompt_rules(
                system_prompt or _default_system_prompt(text_encoder, t2v=True),
                negative_prompt,
            )
            messages: list[dict[str, object]] = [
                {"role": "system", "content": resolved_system_prompt},
                {"role": "user", "content": with_user_negative_prompt(f"user prompt: {prompt}", negative_prompt)},
            ]
            return _generate(text_encoder, messages, image=None, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image_path: str,
        system_prompt: str | None,
        seed: int,
        last_image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        duration: int | None = None,
        fps: int | None = None,
        negative_prompt: str = "",
    ) -> str:
        from services.prompt_enhancement import build_i2v_user_prompt_text, resolve_i2v_frames

        frames = resolve_i2v_frames(image_path, last_image_path, keyframes, fps=fps)
        user_text = build_i2v_user_prompt_text(
            prompt,
            has_last=last_image_path is not None and not keyframes,
            keyframe_count=len(keyframes) if keyframes else 0,
            duration=duration,
            fps=fps,
        )
        return self._generate_with_images(frames, user_text, system_prompt, seed, negative_prompt=negative_prompt)

    def enhance_with_video_frames(
        self,
        prompt: str,
        frames: Sequence[SceneFrame],
        system_prompt: str | None,
        seed: int,
        role: VideoRole,
        negative_prompt: str = "",
    ) -> str:
        from services.prompt_enhancement import build_video_user_prompt_text, resolve_video_frames

        return self._generate_with_images(
            resolve_video_frames(frames),
            build_video_user_prompt_text(
                prompt, role=role, has_look_reference=any(frame.is_reference for frame in frames)
            ),
            system_prompt,
            seed,
            role=role,
            negative_prompt=negative_prompt,
        )

    def _generate_with_images(
        self,
        frames: Sequence[tuple[str, str | None]],
        user_text: str,
        system_prompt: str | None,
        seed: int,
        role: VideoRole | None = None,
        negative_prompt: str = "",
    ) -> str:
        from ltx_pipelines.utils.gpu_model import gpu_model

        from services.prompt_enhancement import (
            video_role_max_output_tokens,
            with_negative_prompt_rules,
            with_user_negative_prompt,
            with_video_role_rules,
        )

        user_content: list[dict[str, object]] = []
        tensors: list[torch.Tensor] = []
        for path, label in frames:
            if label is not None:
                user_content.append({"type": "text", "text": label})
            user_content.append({"type": "image"})
            tensors.append(_image_tensor(path))
        user_content.append({"type": "text", "text": with_user_negative_prompt(user_text, negative_prompt)})

        with gpu_model(self._build_text_encoder()) as text_encoder:
            resolved_system_prompt = system_prompt or _default_system_prompt(text_encoder, t2v=False)
            if role is not None:
                resolved_system_prompt = with_video_role_rules(resolved_system_prompt, role)
            resolved_system_prompt = with_negative_prompt_rules(resolved_system_prompt, negative_prompt)
            messages: list[dict[str, object]] = [
                {"role": "system", "content": resolved_system_prompt},
                {"role": "user", "content": user_content},
            ]
            images: torch.Tensor | list[torch.Tensor] = tensors if len(tensors) > 1 else tensors[0]
            return _generate(
                text_encoder,
                messages,
                image=images,
                seed=seed,
                max_new_tokens_floor=None if role is None else video_role_max_output_tokens(role),
            )
