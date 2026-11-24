"""Stage-1-only Distilled decode stays at canvas/2 and does not upsample."""

from __future__ import annotations

import inspect

import pytest
import torch
from ltx_core.types import SpatioTemporalScaleFactors

import ltx_pipelines.distilled as distilled
from ltx_pipelines.utils.types import ImageConditioningInput, LoadedImageConditioning, PipelineOutput
from services.fast_video_pipeline.distilled_skip_stage_2 import DistilledPipelineWithSkipStage2


class _Encoding:
    video_encoding = None
    audio_encoding = None


class _Conditioner:
    def load_conditionings(self, images, color_space=None):  # noqa: ARG002
        return [LoadedImageConditioning(still=image.path, strength=image.strength, binding=image.binding) for image in images]

    def __call__(self, build):
        return build(object())


class _LatentState:
    def __init__(self, latent: torch.Tensor) -> None:
        self.latent = latent


_DECODER_SCALE = object()
_TRANSFORMER_SCALE = object()


def _pipeline(monkeypatch, *, real_latents: bool = False, decoder_scale: object = _DECODER_SCALE, stage_scale: object = _TRANSFORMER_SCALE):
    stage_pixels: list[tuple[int, int]] = []
    latent_scales: list[object] = []
    image_calls: list[dict] = []
    prompt_kwargs: list[dict] = []

    def fake_image_conditionings(**kwargs):
        image_calls.append(kwargs)
        return []

    def fake_latents(*, pixel_shape, video_scale_factors, **_kwargs):
        latent_scales.append(video_scale_factors)
        video = torch.zeros(1, 4, pixel_shape.frames, pixel_shape.height, pixel_shape.width)
        audio = torch.zeros(1, 8, pixel_shape.frames, 16)
        return video, audio

    monkeypatch.setattr(
        "services.fast_video_pipeline.distilled_skip_stage_2.image_conditionings",
        fake_image_conditionings,
    )
    if not real_latents:
        monkeypatch.setattr(
            "services.fast_video_pipeline.distilled_skip_stage_2.create_initial_av_latents",
            fake_latents,
        )
    monkeypatch.setattr("ltx_pipelines.utils.helpers.tiling_scale_factors_for_vae", lambda _path: decoder_scale)

    pipe = DistilledPipelineWithSkipStage2.__new__(DistilledPipelineWithSkipStage2)
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.bfloat16
    pipe.use_ancestral_sampler = False
    pipe.duration_predictor = None
    pipe.image_conditioner = _Conditioner()

    def prompt_encoder(*_args, **kwargs):
        prompt_kwargs.append(kwargs)
        return (_Encoding(),)

    def stage(**kwargs):
        video = kwargs["modalities"].video
        stage_pixels.append((video.latent.shape[-2], video.latent.shape[-1]))
        return _LatentState(video.latent), _LatentState(video.latent.clone())

    class _Decoder:
        checkpoint_path = "unused"
        diffvae_optimization = None

        def __call__(self, latent, _tiling, _generator, dtype=None):  # noqa: ARG002
            del latent, dtype
            height, width = stage_pixels[-1]
            return iter((torch.zeros(1, 3, 9, height, width),))

    def audio_decoder(latent):  # noqa: ARG001
        return None

    stage.video_scale_factors = stage_scale  # type: ignore[attr-defined]
    pipe.prompt_encoder = prompt_encoder
    pipe.image_calls = image_calls  # type: ignore[attr-defined]
    pipe.prompt_kwargs = prompt_kwargs  # type: ignore[attr-defined]
    pipe.stage = stage
    pipe.video_decoder = _Decoder()
    pipe.audio_decoder = audio_decoder
    return pipe, stage_pixels, latent_scales


# ltx-pipelines v1.6.0 DistilledPipeline.__call__.
_DISTILLED_CALL_PARAMS_V1_6 = (
    "self",
    "prompt",
    "seed",
    "height",
    "width",
    "frame_rate",
    "images",
    "num_frames",
    "vae_dtype",
    "tiling_config",
    "enhance_prompt",
    "enhance_static_cache",
    "stage_1_sigmas",
    "stage_2_sigmas",
    "color_space",
    "generated_keyframes",
    "decode_with_keyframes",
    "chunk_config",
)


def test_distilled_call_signature_is_v1_6() -> None:
    params = tuple(inspect.signature(distilled.DistilledPipeline.__call__).parameters)
    assert params == _DISTILLED_CALL_PARAMS_V1_6


def test_skip_stage_2_decodes_half_canvas_without_upsampling(monkeypatch) -> None:
    pipe, stage_pixels, latent_scales = _pipeline(monkeypatch)

    output = pipe(
        prompt="a clip",
        seed=1,
        height=576,
        width=1024,
        frame_rate=24,
        images=[],
        num_frames=9,
        tiling_config=None,
        skip_stage_2=True,
    )

    assert output.num_frames == 9
    assert stage_pixels == [(288, 512)]
    # Latents use the transformer's geometry. The decoder's tiling scale is a different
    # (smaller) factor on the 2.5 DiffVAE and would allocate an oversized latent.
    assert latent_scales == [_TRANSFORMER_SCALE]


def test_skip_stage_2_latent_shapes_follow_stage_geometry_not_decoder_geometry(monkeypatch) -> None:
    """Real ``create_initial_av_latents``, with a decoder scale unlike the transformer's.

    A 2.5 DiffVAE decoder reports 2x8x8 while the transformer latent is 8x32x32. Sizing from
    the decoder value would allocate an oversized latent and change these shapes.
    """
    decoder = SpatioTemporalScaleFactors(time=2, height=8, width=8)
    stage = SpatioTemporalScaleFactors(time=4, height=16, width=16)
    pipe, _, _ = _pipeline(monkeypatch, real_latents=True, decoder_scale=decoder, stage_scale=stage)
    seen: list[torch.Tensor] = []
    original_stage = pipe.stage

    def recording_stage(**kwargs):
        seen.append(kwargs["modalities"].video.latent)
        return original_stage(**kwargs)

    recording_stage.video_scale_factors = stage  # type: ignore[attr-defined]
    pipe.stage = recording_stage  # type: ignore[assignment]

    pipe(
        prompt="a clip",
        seed=1,
        height=576,
        width=1024,
        frame_rate=24,
        images=[],
        num_frames=9,
        tiling_config=None,
        skip_stage_2=True,
    )

    # Half canvas is 288x512; (9 - 1) / 4 + 1 = 3 latent frames.
    assert tuple(seen[0].shape[-3:]) == (3, 288 // stage.height, 512 // stage.width)


@pytest.mark.parametrize(("requested", "expected"), [(9, 9), (12, 9), (17, 17), (24, 17)])
def test_skip_stage_2_snaps_frames_to_the_vae_grid(monkeypatch, requested: int, expected: int) -> None:
    """Auto-duration can yield off-grid counts; upstream snaps them, so this path must too."""
    pipe, _, _ = _pipeline(monkeypatch)
    frames_seen: list[int] = []

    def recording_latents(*, pixel_shape, **_kwargs):
        frames_seen.append(pixel_shape.frames)
        return torch.zeros(1, 4, 1, pixel_shape.height, pixel_shape.width), torch.zeros(1, 8, 1, 16)

    monkeypatch.setattr(
        "services.fast_video_pipeline.distilled_skip_stage_2.create_initial_av_latents",
        recording_latents,
    )

    output = pipe(
        prompt="a clip",
        seed=1,
        height=576,
        width=1024,
        frame_rate=24,
        images=[],
        num_frames=requested,
        tiling_config=None,
        skip_stage_2=True,
    )

    assert frames_seen == [expected]
    assert output.num_frames == expected


def test_skip_stage_2_rejects_stage_2_and_chunking_arguments(monkeypatch) -> None:
    pipe, _, _ = _pipeline(monkeypatch)
    base = dict(
        prompt="a clip",
        seed=1,
        height=576,
        width=1024,
        frame_rate=24,
        images=[],
        num_frames=9,
        tiling_config=None,
        skip_stage_2=True,
    )

    for ignored in ({"stage_2_sigmas": None}, {"decode_with_keyframes": True}, {"chunk_config": None}):
        with pytest.raises(TypeError):
            pipe(**base, **ignored)


def test_skip_stage_2_false_forwards_to_upstream(monkeypatch) -> None:
    forwarded: dict[str, object] = {}

    def fake_call(self, *args, **kwargs):  # noqa: ARG001
        forwarded["kwargs"] = kwargs
        return PipelineOutput(torch.zeros(1, 3, 9, 8, 8), None, 9, None)

    monkeypatch.setattr(distilled.DistilledPipeline, "__call__", fake_call)
    pipe = DistilledPipelineWithSkipStage2.__new__(DistilledPipelineWithSkipStage2)

    output = pipe(
        prompt="a clip",
        seed=1,
        height=704,
        width=1280,
        frame_rate=24,
        images=[],
        num_frames=9,
        tiling_config=None,
        skip_stage_2=False,
    )

    assert "skip_stage_2" not in forwarded["kwargs"]
    assert forwarded["kwargs"]["prompt"] == "a clip"
    assert (forwarded["kwargs"]["height"], forwarded["kwargs"]["width"]) == (704, 1280)
    assert tuple(output.video.shape) == (1, 3, 9, 8, 8)


def test_skip_stage_2_encodes_loaded_stills_at_half_canvas_and_enhances_from_the_first_path(monkeypatch) -> None:
    """``LoadedImageConditioning`` has no ``path``: it must be read before loading, and the stage-1 encode must use the loaded objects."""
    pipe, _, _ = _pipeline(monkeypatch)
    still = ImageConditioningInput.at("first.png", 0, 1.0)

    pipe(
        prompt="a clip",
        seed=1,
        height=576,
        width=1024,
        frame_rate=24,
        images=[still],
        num_frames=9,
        tiling_config=None,
        skip_stage_2=True,
    )

    assert pipe.prompt_kwargs[0]["enhance_prompt_image"] == "first.png"
    (call,) = pipe.image_calls
    assert [(image.still, image.frame_idx) for image in call["images"]] == [("first.png", 0)]
    assert (call["height"], call["width"]) == (288, 512)


def test_skip_stage_2_rejects_a_still_outside_the_clip(monkeypatch) -> None:
    pipe, _, _ = _pipeline(monkeypatch)
    with pytest.raises(ValueError, match="outside the planned stitched range"):
        pipe(
            prompt="a clip",
            seed=1,
            height=576,
            width=1024,
            frame_rate=24,
            images=[ImageConditioningInput.guide("late.png", 9, 1.0)],
            num_frames=9,
            tiling_config=None,
            skip_stage_2=True,
        )
