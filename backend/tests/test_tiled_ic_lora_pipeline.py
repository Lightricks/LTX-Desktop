"""TiledICLoraPipeline tiles the IC-LoRA reference encode when the canvas exceeds one tile."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from ltx_core.model.video_vae import TileSizeConfig
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.types import ImageConditioningInput

from services.ic_lora_pipeline.ic_lora_inference import InferenceResult, encode_result
from services.ic_lora_pipeline.tiled_ic_lora_pipeline import (
    TiledICLoraPipeline,
    encode_tiling_for_canvas,
    reference_encode_canvas,
    reference_encode_tiling,
)


def test_video_conditionings_signature_matches_upstream() -> None:
    ours = inspect.signature(TiledICLoraPipeline._video_conditionings)
    upstream = inspect.signature(ICLoraPipeline._video_conditionings)
    assert list(ours.parameters) == list(upstream.parameters)


def test_reference_encode_tiling_skips_sub_tile_canvas() -> None:
    tile = TileSizeConfig.default()
    assert reference_encode_tiling(tile.height.tile_size, tile.width.tile_size, 9) is None
    assert reference_encode_tiling(512, 768, 9) is None


def test_reference_encode_tiling_uses_default_when_canvas_exceeds_tile() -> None:
    tile = TileSizeConfig.default()
    assert reference_encode_tiling(1080, 1920, 241) == tile
    assert reference_encode_tiling(512, 768, tile.frames.tile_size + 1) == tile


def test_reference_encode_canvas_aligns_stage1_to_vae_grid() -> None:
    assert reference_encode_canvas(1280, 704, downscale_factor=1) == (1280, 704)
    assert reference_encode_canvas(1280, 704, downscale_factor=2) == (1280, 704)
    assert reference_encode_canvas(1300, 700, downscale_factor=1) == (1280, 672)


def test_encode_tiling_for_canvas_matches_post_downscale_size() -> None:
    assert encode_tiling_for_canvas(512, 768, 9, downscale_factor=1) is None
    assert encode_tiling_for_canvas(1080, 1920, 241, downscale_factor=1) == TileSizeConfig.default()
    assert encode_tiling_for_canvas(1280, 704, 121, downscale_factor=1) == TileSizeConfig.default()
    # append encodes at stage-1 / 2 = 640×352. That fits one 768 tile at 9 frames.
    assert encode_tiling_for_canvas(1280, 704, 9, downscale_factor=2) is None


def test_video_conditionings_passes_desktop_encode_tiling(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_reference(chunk, **kwargs):  # noqa: ARG001
        captured["encode_tiling"] = kwargs.get("encode_tiling")
        captured["downscale"] = kwargs.get("downscale_factor")
        captured["temporal"] = kwargs.get("reference_temporal_scale_factor")
        captured["strength"] = kwargs.get("conditioning_attention_strength")
        captured["color_space"] = kwargs.get("color_space")
        return ["reference"]

    def fake_images(chunk, **kwargs):  # noqa: ARG001
        captured["image_color_space"] = kwargs.get("color_space")
        return ["image"]

    monkeypatch.setattr(
        "ltx_pipelines.chunks.conditionings.reference_video_conditionings_for_chunk",
        fake_reference,
    )
    monkeypatch.setattr("ltx_pipelines.chunks.conditionings.image_conditionings_for_chunk", fake_images)
    pipe = TiledICLoraPipeline.__new__(TiledICLoraPipeline)
    pipe.reference_downscale_factor = 1
    pipe.reference_temporal_scale_factor = 2
    pipe.image_conditioner = lambda build: build(object())
    color_space = object()
    ctx = SimpleNamespace(
        images=[],
        num_frames=121,
        video_conditioning=[("ref.mp4", 1.0)],
        conditioning_attention_strength=0.7,
        conditioning_attention_mask=None,
        color_space=color_space,
    )
    # 40 * 32 = 1280, 22 * 32 = 704. Exceeds one default tile at 121 frames.
    chunk = SimpleNamespace(
        video=torch.zeros(1, 4, 2, 40, 22),
        layout=SimpleNamespace(pixel_frames=121),
    )

    make = pipe._video_conditionings(ctx, SimpleNamespace(apply_ic_lora=True), {})
    items = make(chunk)

    # Image conditionings come first, then the reference video, in one encoder pass.
    assert items == ["image", "reference"]
    assert captured["downscale"] == 1
    assert captured["encode_tiling"] == TileSizeConfig.default()
    assert captured["temporal"] == 2
    assert captured["strength"] == 0.7
    assert captured["color_space"] is color_space
    assert captured["image_color_space"] is color_space
    # Audio-only chunks carry no video, so nothing is encoded.
    assert make(SimpleNamespace(video=None, layout=SimpleNamespace(pixel_frames=0))) == []


def test_stage_without_ic_lora_takes_upstream_closure(monkeypatch) -> None:
    """Stage 2 of the default recipe runs without the IC-LoRA, so no reference is encoded."""
    sentinel = object()
    seen: dict[str, object] = {}

    def fake_parent(self, ctx, stage_config, frame_sources):  # noqa: ARG001
        seen["stage_config"] = stage_config
        return sentinel

    monkeypatch.setattr(ICLoraPipeline, "_video_conditionings", fake_parent)
    pipe = TiledICLoraPipeline.__new__(TiledICLoraPipeline)
    stage_config = SimpleNamespace(apply_ic_lora=False)

    result = pipe._video_conditionings(SimpleNamespace(), stage_config, {})

    assert result is sentinel
    assert seen["stage_config"] is stage_config


def _still_ctx(*images: ImageConditioningInput) -> SimpleNamespace:
    return SimpleNamespace(
        images=list(images),
        num_frames=121,
        video_conditioning=[("ref.mp4", 1.0)],
        conditioning_attention_strength=1.0,
        conditioning_attention_mask=None,
        color_space=object(),
    )


def _still_pipeline(downscale: int = 1) -> TiledICLoraPipeline:
    pipe = TiledICLoraPipeline.__new__(TiledICLoraPipeline)
    pipe.reference_downscale_factor = downscale
    pipe.reference_temporal_scale_factor = 1
    pipe.image_conditioner = lambda build: build(object())
    return pipe


def test_negative_frame_still_is_appended_after_the_reference_video(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_images(chunk, **kwargs):  # noqa: ARG001
        seen["frame_images"] = kwargs["images"]
        return ["image"]

    def fake_reference(chunk, **kwargs):  # noqa: ARG001
        return ["reference"]

    def fake_stills(**kwargs):
        seen["stills"] = kwargs["images"]
        seen["size"] = (kwargs["height"], kwargs["width"])
        return ["still"]

    monkeypatch.setattr("ltx_pipelines.chunks.conditionings.image_conditionings_for_chunk", fake_images)
    monkeypatch.setattr(
        "ltx_pipelines.chunks.conditionings.reference_video_conditionings_for_chunk", fake_reference
    )
    monkeypatch.setattr(
        "services.ic_lora_pipeline.tiled_ic_lora_pipeline.combined_image_conditionings", fake_stills
    )
    first_frame = ImageConditioningInput("first.png", 0, 1.0)
    still = ImageConditioningInput("look.png", -1, 1.0)
    chunk = SimpleNamespace(
        video=torch.zeros(1, 4, 2, 16, 24),
        layout=SimpleNamespace(pixel_frames=9),
    )

    make = _still_pipeline()._video_conditionings(
        _still_ctx(first_frame, still), SimpleNamespace(apply_ic_lora=True), {}
    )

    assert make(chunk) == ["image", "reference", "still"]
    assert seen["frame_images"] == [first_frame]
    assert seen["stills"] == [still]
    assert seen["size"] == (16 * 32, 24 * 32)


def test_still_at_another_negative_index_is_rejected() -> None:
    ctx = _still_ctx(ImageConditioningInput("look.png", -2, 1.0))

    with pytest.raises(ValueError, match="frame_idx -1"):
        _still_pipeline()._video_conditionings(ctx, SimpleNamespace(apply_ic_lora=True), {})


def test_still_with_a_reference_downscale_is_rejected() -> None:
    ctx = _still_ctx(ImageConditioningInput("look.png", -1, 1.0))

    with pytest.raises(ValueError, match="reference_downscale_factor 1"):
        _still_pipeline(downscale=2)._video_conditionings(ctx, SimpleNamespace(apply_ic_lora=True), {})


def test_stage_without_ic_lora_rejects_a_negative_frame_still() -> None:
    """The upstream closure refuses a negative index, so a still never reaches a plain stage."""
    ctx = _still_ctx(ImageConditioningInput("look.png", -1, 1.0))

    with pytest.raises(ValueError, match="outside the planned stitched range"):
        _still_pipeline()._video_conditionings(ctx, SimpleNamespace(apply_ic_lora=False), {})


def test_failed_encode_closes_the_video_generator_so_the_reference_is_released(monkeypatch, tmp_path) -> None:
    state = {"closed": False}

    def windows():
        try:
            yield torch.zeros(1, 3, 9, 16, 16)
            yield torch.zeros(1, 3, 9, 16, 16)
        finally:
            state["closed"] = True

    def failing_encode(*, video, **kwargs):  # noqa: ARG001
        next(iter(video))
        raise RuntimeError("encoder died")

    monkeypatch.setattr("ltx_pipelines.utils.media_io.encode_video", failing_encode)
    video = windows()
    result = InferenceResult(
        video=video, audio=None, num_frames=18, tiling_config=None, streams=(video,)
    )

    with pytest.raises(RuntimeError, match="encoder died"):
        encode_result(
            result,
            output_path=str(tmp_path / "out.mp4"),
            num_frames=18,
            frame_rate=24.0,
            mute_audio=True,
            source_audio_path=None,
            streamed=True,
            device=torch.device("cpu"),
        )

    assert state["closed"] is True
