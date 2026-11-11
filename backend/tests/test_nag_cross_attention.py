"""NAG text cross-attention patch: each test names the bug it catches."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from ltx_core.model.transformer.attention import AttentionFunction, AttentionOps, MaskedAttentionFunction
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import AdaLNTimestepEmbedding
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.types import LoadedImageConditioning

from api_types import ExtendParams, RetakeParams, TextToVideoParams
from runtime_config.nag_limits import NEGATIVE_PROMPT_MAX_LENGTH
from services.fast_video_pipeline.distilled_skip_stage_2 import DistilledPipelineWithSkipStage2
from services.fast_video_pipeline.nag_config import (
    NagConfig,
    NagOffReason,
    decide_nag,
    nag_config_for_request,
)
from services.fast_video_pipeline.nag_distilled import NagDistilledMixin
from services.features.video.extend import ExtendExecutor
from services.features.video.retake import RetakeExecutor
from services.ic_lora_pipeline.tiled_ic_lora_pipeline import TiledICLoraPipeline
from services.patches.nag_cross_attention import (
    NagState,
    active_state,
    nag_scope,
    normalized_attention_guidance,
)
from state.app_state_types import RetakePipelineState, TextEncodingResult

_VIDEO_DIM = 32
_AUDIO_DIM = 16
_POSITIVE_TOKENS = 5
_VIDEO_TOKENS = 6


# AUTOMATIC picks the MPS kernel on Apple hosts, which rejects CPU tensors.
_CPU_ATTENTION = AttentionOps(
    attention_function=AttentionFunction.PYTORCH.to_callable(),
    masked_attention_function=MaskedAttentionFunction.PYTORCH.to_callable(),
)


def _block(*, cross_attention_adaln: bool) -> BasicAVTransformerBlock:
    torch.manual_seed(0)
    block = BasicAVTransformerBlock(
        video=TransformerConfig(
            dim=_VIDEO_DIM, heads=2, d_head=16, context_dim=_VIDEO_DIM, cross_attention_adaln=cross_attention_adaln
        ),
        audio=TransformerConfig(
            dim=_AUDIO_DIM, heads=2, d_head=8, context_dim=_AUDIO_DIM, cross_attention_adaln=cross_attention_adaln
        ),
        attention_ops=_CPU_ATTENTION,
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    return block.eval()


def _state() -> NagState:
    return NagState(config=NagConfig(negative_prompt="blurry"), positive_length=_POSITIVE_TOKENS)


def _text_cross_attention(
    block: BasicAVTransformerBlock,
    attn: torch.nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    *,
    cross_attention_adaln: bool,
) -> torch.Tensor:
    dim = x.shape[-1]
    table = block.scale_shift_table if dim == _VIDEO_DIM else block.audio_scale_shift_table
    prompt_table = getattr(block, "prompt_scale_shift_table", None) if cross_attention_adaln else None
    modulation = torch.randn(1, 1, table.shape[0] * dim, generator=torch.Generator().manual_seed(1))
    prompt_timestep = torch.randn(1, 1, 2 * dim, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        return block._apply_text_cross_attention(  # pyright: ignore[reportPrivateUsage]
            x,
            context,
            attn,  # pyright: ignore[reportArgumentType]
            table,
            prompt_table,
            AdaLNTimestepEmbedding(modulation=modulation, embedding=modulation),
            prompt_timestep if cross_attention_adaln else None,
            None,
            cross_attention_adaln=cross_attention_adaln,
        )


def test_nag_off_leaves_video_text_cross_attention_unchanged() -> None:
    block = _block(cross_attention_adaln=False)
    x = torch.randn(1, _VIDEO_TOKENS, _VIDEO_DIM)
    context = torch.randn(1, _POSITIVE_TOKENS, _VIDEO_DIM)

    out = _text_cross_attention(block, block.attn2, x, context, cross_attention_adaln=False)

    with torch.no_grad():
        assert torch.equal(out, block.attn2(x, context=context, mask=None))


def test_negative_context_gets_the_same_adaln_modulation_as_the_positive_context() -> None:
    # With negative == positive, Z_neg == Z_pos and NAG must return the unguided output.
    # An unmodulated negative half would differ from the modulated positive half and break this.
    block = _block(cross_attention_adaln=True)
    x = torch.randn(1, _VIDEO_TOKENS, _VIDEO_DIM)
    positive = torch.randn(1, _POSITIVE_TOKENS, _VIDEO_DIM)
    unguided = _text_cross_attention(block, block.attn2, x, positive, cross_attention_adaln=True)

    with nag_scope(_state()):
        same = _text_cross_attention(
            block, block.attn2, x, torch.cat([positive, positive], dim=1), cross_attention_adaln=True
        )
        other = _text_cross_attention(
            block,
            block.attn2,
            x,
            torch.cat([positive, torch.randn(1, _POSITIVE_TOKENS, _VIDEO_DIM)], dim=1),
            cross_attention_adaln=True,
        )

    assert torch.allclose(same, unguided, atol=1e-5)
    assert not torch.allclose(other, unguided, atol=1e-3)


def test_audio_text_cross_attention_and_self_attention_are_not_guided() -> None:
    block = _block(cross_attention_adaln=False)
    video_x = torch.randn(1, _VIDEO_TOKENS, _VIDEO_DIM)
    video_context = torch.cat([torch.randn(1, _POSITIVE_TOKENS, _VIDEO_DIM)] * 2, dim=1)
    audio_x = torch.randn(1, 4, _AUDIO_DIM)
    audio_context = torch.randn(1, _POSITIVE_TOKENS + 2, _AUDIO_DIM)

    with nag_scope(_state()):
        audio_out = _text_cross_attention(block, block.audio_attn2, audio_x, audio_context, cross_attention_adaln=False)
        self_out = _text_cross_attention(block, block.attn1, video_x, video_context, cross_attention_adaln=False)

    with torch.no_grad():
        assert torch.equal(audio_out, block.audio_attn2(audio_x, context=audio_context, mask=None))
        assert torch.equal(self_out, block.attn1(video_x, context=video_context, mask=None))


def test_nag_scope_is_closed_after_a_failed_generation() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with nag_scope(_state()):
            raise RuntimeError("boom")

    assert active_state() is None


@pytest.mark.parametrize("negative_prompt", ["", "   \n\t"])
def test_blank_negative_prompt_does_not_turn_nag_on(negative_prompt: str) -> None:
    assert nag_config_for_request(negative_prompt, {}) is None


def test_zero_nag_scale_or_env_disabled_turns_nag_off() -> None:
    assert nag_config_for_request("blurry", {}, scale=0.0) is None
    assert nag_config_for_request("blurry", {}, default_scale=0.0) is None
    assert nag_config_for_request("blurry", {"LTX_NAG_ENABLED": "0"}) is None


def test_queued_generation_snapshots_nag_scale_so_later_settings_edits_do_not_change_rerenders(
    client, test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.nag_scale = 7.5
    created = client.post(
        "/api/generations/text-to-video",
        json={
            "params": {
                "prompt": "a cat",
                "negativePrompt": "subtitles, watermark",
                "model": "ltx-2.5-fast",
                "aspectRatio": "16:9",
                "resolution": "540p",
                "duration": 5,
                "fps": 24,
            }
        },
    )
    assert created.status_code == 200, created.text
    params = created.json()["spec"]["params"]
    assert params["negativePrompt"] == "subtitles, watermark"
    assert params["nagScale"] == 7.5
    assert params["nagAlpha"] == 0.25
    assert params["nagTau"] == 2.5

    # Changing Settings afterwards does not alter the persisted spec.
    test_state.state.app_settings.nag_scale = 15.0
    fetched = client.get(f"/api/generations/{created.json()['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["spec"]["params"]["nagScale"] == 7.5


def test_normalized_attention_guidance_clips_when_l1_ratio_exceeds_tau() -> None:
    z_pos = torch.ones(1, 3, 8)
    z_neg = -torch.ones(1, 3, 8)
    # Unclipped guidance is 11 * 1 - 10 * (-1) = 21 per element (L1 ratio = 21 > tau = 2.0).
    # With alpha=1.0, the output is the clipped guidance vector Z^, whose L1 norm must equal tau * ||z_pos||_1.
    out = normalized_attention_guidance(z_pos, z_neg, scale=11.0, alpha=1.0, tau=2.0)
    pos_l1 = torch.linalg.vector_norm(z_pos, ord=1, dim=-1)
    out_l1 = torch.linalg.vector_norm(out, ord=1, dim=-1)
    assert torch.allclose(out_l1, pos_l1 * 2.0, atol=1e-4)


@dataclass(frozen=True)
class _FakeRunCtx:
    video_context: torch.Tensor
    audio_context: torch.Tensor


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, prompts: list[str], **kwargs: object) -> list[EmbeddingsProcessorOutput]:
        self.calls.append((list(prompts), dict(kwargs)))
        return [
            EmbeddingsProcessorOutput(
                video_encoding=torch.full((1, 4, 8), float(index + 1)),
                audio_encoding=torch.zeros(1, 2, 4),
                attention_mask=torch.ones(1, 4),
            )
            for index, _ in enumerate(prompts)
        ]


class _FakeDistilledBase:
    def __init__(self) -> None:
        self.prompt_encoder = _FakeEncoder()
        self.active_during_next: list[NagState | None] = []

    def _prepare_run(self, **kwargs: object) -> _FakeRunCtx:
        (ctx_p,) = self.prompt_encoder([str(kwargs["prompt"])], enhance_first_prompt=True)
        assert ctx_p.audio_encoding is not None
        return _FakeRunCtx(video_context=ctx_p.video_encoding, audio_context=ctx_p.audio_encoding)

    def stream_chunks(self, **kwargs: object):
        ctx = self._prepare_run(**kwargs)

        def _gen():
            for chunk_idx in range(3):
                self.active_during_next.append(active_state())
                yield (chunk_idx, ctx.video_context.shape[1])

        return _gen(), 9, "tiling"


class _FakeNagPipeline(NagDistilledMixin, _FakeDistilledBase):
    pass


def test_prepare_run_appends_negative_embedding_and_sets_positive_length() -> None:
    pipe = _FakeNagPipeline()
    original_encoder = pipe.prompt_encoder
    nag = NagConfig(negative_prompt="watermark", scale=9.0)

    with pipe._nag_request(nag):  # pyright: ignore[reportPrivateUsage]
        ctx = pipe._prepare_run(prompt="a fox")  # pyright: ignore[reportPrivateUsage]

    assert pipe.prompt_encoder is original_encoder
    assert original_encoder.calls == [(["a fox", "watermark"], {"enhance_first_prompt": True})]
    assert ctx.video_context.shape == (1, 8, 8)
    assert torch.all(ctx.video_context[:, :4] == 1.0)
    assert torch.all(ctx.video_context[:, 4:] == 2.0)
    assert pipe._nag_state is not None  # pyright: ignore[reportPrivateUsage]
    assert pipe._nag_state.positive_length == 4  # pyright: ignore[reportPrivateUsage]


def test_scoped_chunks_activates_nag_only_inside_next_and_not_between_chunks() -> None:
    pipe = _FakeNagPipeline()
    nag = NagConfig(negative_prompt="flicker", scale=11.0)

    chunks, num_frames, _ = pipe.stream_chunks(prompt="a fox", nag=nag)
    assert num_frames == 9
    assert active_state() is None

    seen_between_chunks: list[NagState | None] = []
    yielded = []
    for item in chunks:
        seen_between_chunks.append(active_state())
        yielded.append(item)

    assert active_state() is None
    assert yielded == [(0, 8), (1, 8), (2, 8)]
    assert seen_between_chunks == [None, None, None]
    assert len(pipe.active_during_next) == 3
    assert all(state is not None and state.positive_length == 4 for state in pipe.active_during_next)


def test_skip_stage_2_activates_nag_denoise_scope_during_stage_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.fast_video_pipeline.distilled_skip_stage_2.image_conditionings",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "services.fast_video_pipeline.distilled_skip_stage_2.create_initial_av_latents",
        lambda *, pixel_shape, **_kwargs: (
            torch.zeros(1, 4, pixel_shape.frames, pixel_shape.height, pixel_shape.width),
            torch.zeros(1, 8, pixel_shape.frames, 16),
        ),
    )

    pipe = DistilledPipelineWithSkipStage2.__new__(DistilledPipelineWithSkipStage2)
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.bfloat16
    pipe.use_ancestral_sampler = False
    pipe.duration_predictor = None

    class _Conditioner:
        def load_conditionings(self, images, color_space=None):  # noqa: ARG002
            return [
                LoadedImageConditioning(still=img.path, strength=img.strength, binding=img.binding)
                for img in images
            ]

        def __call__(self, build):
            return build(object())

    class _LatentState:
        def __init__(self, latent: torch.Tensor) -> None:
            self.latent = latent

    class _Decoder:
        checkpoint_path = "unused"
        diffvae_optimization = None

        def __call__(self, _latent, _tiling, _generator, dtype=None):  # noqa: ARG002
            # Decode runs after stage 1 exits, so NAG scope must already be closed here.
            assert active_state() is None
            return torch.zeros(1, 3, 9, 288, 512)

    active_in_stage: list[NagState | None] = []

    def stage(**kwargs):
        active_in_stage.append(active_state())
        video = kwargs["modalities"].video
        return _LatentState(video.latent), _LatentState(video.latent.clone())

    stage.video_scale_factors = object()  # type: ignore[attr-defined]
    pipe.image_conditioner = _Conditioner()  # type: ignore[assignment]
    pipe.prompt_encoder = _FakeEncoder()
    pipe.stage = stage  # type: ignore[assignment]
    pipe.video_decoder = _Decoder()  # type: ignore[assignment]
    pipe.audio_decoder = lambda _latent: None  # type: ignore[assignment]

    nag = NagConfig(negative_prompt="subtitles", scale=11.0)
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
        nag=nag,
    )

    assert active_state() is None
    assert len(active_in_stage) == 1
    assert active_in_stage[0] is not None
    assert active_in_stage[0].config == nag
    assert active_in_stage[0].positive_length == 4


def test_handler_drops_nag_when_text_encoding_is_not_local_or_compile_is_enabled(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    # 1. API text encoding active (API key set, use_local_text_encoder=False): NAG is skipped.
    test_state.state.app_settings.ltx_api_key = "test-api-key"
    test_state.state.app_settings.use_local_text_encoder = False
    fake_services.text_encoder.encode_responses.append(
        TextEncodingResult(video_context=torch.zeros(1, 4, 8), audio_context=None)
    )

    resp_api = client.post(
        "/api/generate",
        json={
            "prompt": "a fox",
            "negativePrompt": "blurry",
            "model": "fast",
            "resolution": "540p",
            "duration": 5,
            "fps": 24,
            "aspectRatio": "16:9",
        },
    )
    assert resp_api.status_code == 200, resp_api.text
    assert fake_services.fast_video_pipeline.generate_calls[-1]["nag"] is None

    # 2. Local text encoding active, but use_torch_compile=True: NAG is skipped.
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.use_torch_compile = True

    resp_compile = client.post(
        "/api/generate",
        json={
            "prompt": "a fox",
            "negativePrompt": "blurry",
            "model": "fast",
            "resolution": "540p",
            "duration": 5,
            "fps": 24,
            "aspectRatio": "16:9",
        },
    )
    assert resp_compile.status_code == 200, resp_compile.text
    assert fake_services.fast_video_pipeline.generate_calls[-1]["nag"] is None

    queued_compile = client.post(
        "/api/generations/text-to-video",
        json={
            "params": {
                "prompt": "a fox",
                "negativePrompt": "blurry",
                "model": "ltx-2.5-fast",
                "aspectRatio": "16:9",
                "resolution": "540p",
                "duration": 5,
                "fps": 24,
            }
        },
    )
    assert queued_compile.status_code == 200, queued_compile.text
    assert "nagScale" not in queued_compile.json()["spec"]["params"]


def test_edit_executor_resolve_nag_returns_none_when_default_scale_is_zero_and_spec_omits_nag_scale(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.nag_scale = 0.0
    retake_executor = RetakeExecutor(
        test_state.pipelines,
        test_state.assets,  # type: ignore[arg-type]
        test_state.text,
        lambda: test_state.video_generation.models_dir,
        test_state.config.default_negative_prompt,
        test_state.generation,
        test_state.prompt_enhancement,
        default_nag_scale=lambda: test_state.settings.get_settings_snapshot().nag_scale,
    )
    extend_executor = ExtendExecutor(
        test_state.pipelines,
        test_state.assets,  # type: ignore[arg-type]
        test_state.text,
        lambda: test_state.video_generation.models_dir,
        test_state.config.default_negative_prompt,
        test_state.generation,
        test_state.prompt_enhancement,
        default_nag_scale=lambda: test_state.settings.get_settings_snapshot().nag_scale,
    )

    pipeline_state = RetakePipelineState(
        pipeline=object(),  # type: ignore[arg-type]
        distilled=True,
        quantized=False,
        ltx_model_id="ltx-2.5-22b-distilled",
        loading_mode="full_models_loading",
        gemma_root="/fake/gemma",
    )
    retake_params = RetakeParams(
        model="ltx-2.5-fast",
        prompt="replace with a cat",
        negativePrompt="blurry, flicker",
        startTime=0.0,
        duration=2.0,
    )
    extend_params = ExtendParams(
        model="ltx-2.5-fast",
        prompt="continue the walk",
        negativePrompt="blurry, flicker",
        duration=2.0,
    )

    assert retake_executor.resolve_nag(retake_params, pipeline_state) is None
    assert extend_executor.resolve_nag(extend_params, pipeline_state) is None

    test_state.state.app_settings.nag_scale = 8.5
    resolved_retake = retake_executor.resolve_nag(retake_params, pipeline_state)
    resolved_extend = extend_executor.resolve_nag(extend_params, pipeline_state)
    assert resolved_retake is not None and resolved_retake.scale == 8.5
    assert resolved_extend is not None and resolved_extend.scale == 8.5


def test_dump_fast_params_allows_nag_on_edit_and_a2v_when_torch_compile_is_enabled(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.nag_scale = 9.0
    test_state.state.app_settings.use_torch_compile = True

    fast_dumped = test_state.queued_generations._dump_fast_params(  # pyright: ignore[reportPrivateUsage]
        TextToVideoParams(
            prompt="a fox",
            negativePrompt="blurry",
            model="ltx-2.5-fast",
            aspectRatio="16:9",
            resolution="540p",
            duration=5,
            fps=24,
        )
    )
    assert "nagScale" not in fast_dumped

    edit_dumped = test_state.queued_generations._dump_fast_params(  # pyright: ignore[reportPrivateUsage]
        RetakeParams(
            model="ltx-2.5-fast",
            prompt="replace with a cat",
            negativePrompt="blurry",
            startTime=0.0,
            duration=2.0,
        ),
        uses_torch_compile=False,
    )
    assert edit_dumped["nagScale"] == 9.0


def test_tiled_ic_lora_pipeline_activates_nag_scope_inside_stream_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = TiledICLoraPipeline.__new__(TiledICLoraPipeline)
    encoder = _FakeEncoder()
    pipe.prompt_encoder = encoder  # type: ignore[assignment]
    active_during_next: list[NagState | None] = []

    def fake_parent_stream_chunks(self: object, **kwargs: object):
        ctx = self._prepare_run(**kwargs)  # type: ignore[attr-defined]

        def _gen():
            for chunk_idx in range(3):
                active_during_next.append(active_state())
                yield (chunk_idx, ctx.video_context.shape[1])

        return _gen(), 9, "tiling"

    monkeypatch.setattr(ICLoraPipeline, "_prepare_run", _FakeDistilledBase._prepare_run)
    monkeypatch.setattr(ICLoraPipeline, "stream_chunks", fake_parent_stream_chunks)
    nag = NagConfig(negative_prompt="watermark", scale=11.0)

    chunks, num_frames, _ = pipe.stream_chunks(prompt="a fox", nag=nag)
    assert num_frames == 9
    assert active_state() is None
    yielded = list(chunks)
    assert active_state() is None
    assert yielded == [(0, 8), (1, 8), (2, 8)]
    assert len(active_during_next) == 3
    assert all(
        state is not None and state.config == nag and state.positive_length == 4
        for state in active_during_next
    )


def test_decide_nag_returns_config_or_off_reason_across_inputs() -> None:
    assert decide_nag("", {}).reason == NagOffReason.NO_NEGATIVE
    assert decide_nag("blurry", {"LTX_NAG_ENABLED": "0"}).reason == NagOffReason.DISABLED_BY_ENV
    assert decide_nag("blurry", {}, scale=0.0).reason == NagOffReason.STRENGTH_ZERO
    assert decide_nag("blurry", {}, default_scale=0.0).reason == NagOffReason.STRENGTH_ZERO
    assert (
        decide_nag("blurry", {}, local_encoding=False).reason
        == NagOffReason.API_ENCODING
    )
    assert (
        decide_nag("blurry", {}, compiled=True).reason
        == NagOffReason.TORCH_COMPILE
    )
    active = decide_nag("blurry", {}, scale=7.5)
    assert active.reason is None
    assert active.config == NagConfig(negative_prompt="blurry", scale=7.5)
    assert active.enhancer_negative == "blurry"


def test_queued_generation_preserves_explicit_zero_nag_scale_so_execution_does_not_fall_back_to_default(
    client, test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.nag_scale = 11.0
    created = client.post(
        "/api/generations/text-to-video",
        json={
            "params": {
                "prompt": "a cat",
                "negativePrompt": "blurry",
                "nagScale": 0,
                "model": "ltx-2.5-fast",
                "aspectRatio": "16:9",
                "resolution": "540p",
                "duration": 5,
                "fps": 24,
            }
        },
    )
    assert created.status_code == 200, created.text
    params = created.json()["spec"]["params"]
    assert params["nagScale"] == 0.0
    assert "nagAlpha" not in params
    assert "nagTau" not in params


def test_negative_prompt_exceeding_max_length_is_rejected_with_422(client) -> None:
    too_long = "x" * (NEGATIVE_PROMPT_MAX_LENGTH + 1)
    resp = client.post(
        "/api/generations/text-to-video",
        json={
            "params": {
                "prompt": "a cat",
                "negativePrompt": too_long,
                "model": "ltx-2.5-fast",
                "aspectRatio": "16:9",
                "resolution": "540p",
                "duration": 5,
                "fps": 24,
            }
        },
    )
    assert resp.status_code == 422


def test_generate_forwards_negative_prompt_to_enhancer_and_nag(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files(include_prompt_enhancer=True)
    test_state.state.app_settings.use_local_text_encoder = True
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a vivid fox running"

    resp = client.post(
        "/api/generate",
        json={
            "prompt": "a fox",
            "negativePrompt": "blurry",
            "model": "fast",
            "resolution": "540p",
            "duration": 5,
            "fps": 24,
            "aspectRatio": "16:9",
        },
    )
    assert resp.status_code == 200, resp.text
    assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[-1]["negative_prompt"] == "blurry"
    nag = fake_services.fast_video_pipeline.generate_calls[-1]["nag"]
    assert nag is not None
    assert nag.negative_prompt == "blurry"


def test_generation_does_not_steer_prompt_enhancer_with_negative_prompt_when_nag_is_disabled(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files(include_prompt_enhancer=True)
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.nag_scale = 0.0
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a vivid fox running"

    resp = client.post(
        "/api/generate",
        json={
            "prompt": "a fox",
            "negativePrompt": "blurry, watermark",
            "model": "fast",
            "resolution": "540p",
            "duration": 5,
            "fps": 24,
            "aspectRatio": "16:9",
        },
    )
    assert resp.status_code == 200, resp.text
    assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[-1]["negative_prompt"] == ""
    assert fake_services.fast_video_pipeline.generate_calls[-1]["nag"] is None


def test_edit_executor_does_not_pass_negative_prompt_to_enhancer_when_local_encoding_is_unavailable(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    test_state.state.app_settings.ltx_api_key = "test-api-key"
    test_state.state.app_settings.use_local_text_encoder = False
    retake_executor = RetakeExecutor(
        test_state.pipelines,
        test_state.assets,  # type: ignore[arg-type]
        test_state.text,
        lambda: test_state.video_generation.models_dir,
        test_state.config.default_negative_prompt,
        test_state.generation,
        test_state.prompt_enhancement,
        default_nag_scale=lambda: test_state.settings.get_settings_snapshot().nag_scale,
    )
    params = RetakeParams(
        model="ltx-2.5-fast",
        prompt="replace with a cat",
        negativePrompt="blurry, flicker",
        startTime=0.0,
        duration=2.0,
    )
    decision = retake_executor.decide_nag(
        params,
        local_encoding=test_state.text.should_use_local_encoding("ltx-2.5-22b-distilled"),
    )
    assert decision.reason == NagOffReason.API_ENCODING
    assert decision.enhancer_negative == ""


