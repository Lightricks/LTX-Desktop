"""A Home IC-LoRA recipe run keeps every frame of the clip: pad up, run, cut back.

The executor runs with the real frame budget (``IcLoraHandler.recipe_run_frames``) and real
ffmpeg. Only the model is a stand-in, and it has the real frame snap: it writes the largest
8k+1 count that fits the clip it receives.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import JsonValue

from api_types import (
    IcLoraRecipeStoredParams,
    IcLoraSettings,
    LoraEntry,
    VideoAssetMetadata,
    VideoMeta,
)
from services.features.video.ic_lora_recipe import IcLoraRecipeExecutor
from services.generation_queue.types import OutputAllocation
from tests.fakes.generation import FakeJobSession
from services.media_probe import first_audio_codec, video_frame_count
from services.records import AssetRecord, CapabilityFailedError, GenerationRecord
from tests.lora_catalog_helpers import add_ic_lora
from tests.video_fixtures import write_pattern_clip

_ID = "executor-frames-test"
_FPS = 24
_GIB = 1024**3


class _GridModel:
    """Stands in for the model: it snaps the clip it receives DOWN to 8k+1 frames."""

    def __init__(self, ic_lora) -> None:
        self._ic_lora = ic_lora
        self.received_frames: int | None = None

    def recipe_run_frames(self, **kwargs: object):
        return self._ic_lora.recipe_run_frames(**kwargs)

    def generate_local_reserved(self, request: object, **kwargs: object) -> object:
        self.received_frames = video_frame_count(Path(str(getattr(request, "input_path"))))
        write_pattern_clip(
            Path(str(kwargs["output_path"])), frames=(self.received_frames - 1) // 8 * 8 + 1
        )
        return object()


@pytest.fixture
def catalog(test_state, fake_services, create_fake_model_files) -> None:
    create_fake_model_files()
    test_state.config.system = "Windows"
    test_state.config.vram_gb = 32
    test_state.config.vram_bytes = 32 * _GIB
    test_state.config.total_ram_bytes = 64 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    add_ic_lora(
        fake_services,
        id=_ID,
        default_settings=IcLoraSettings(skip_stage_2=True, resolution_factor=1.0, audio_mode="source"),
    )


def _stored() -> IcLoraRecipeStoredParams:
    return IcLoraRecipeStoredParams(
        prompt="p",
        model="ltx-2.5-fast",
        resolution="540p",
        audioMode="source",
        fps=_FPS,
        loras=[LoraEntry(ref="", scale=1.0, catalogId=_ID)],
    )


def _first_refused_count(test_state) -> int:
    """First on-grid count the gate refuses in standard mode, or runs with a risk."""

    def refused(clip_frames: int) -> bool:
        try:
            _, risk = test_state.ic_lora.recipe_run_frames(
                ic_lora_id=_ID, source_width=64, source_height=64, stored=_stored(),
                clip_frames=clip_frames,
            )
        except Exception:
            return True
        return risk is not None

    return next(count for count in range(9, 20000, 8) if refused(count))


def _run(
    test_state,
    tmp_path: Path,
    source: Path,
    *,
    frames: int,
    audio: bool,
    risk: dict[str, JsonValue] | None = None,
) -> tuple[Path, _GridModel]:
    asset = AssetRecord(
        id="clip",
        media_kind="video",
        origin="uploaded",
        path=str(source),
        mime_type="video/mp4",
        name="clip.mp4",
        metadata=VideoAssetMetadata(
            mediaType="video",
            metadata=VideoMeta(
                width=64,
                height=64,
                durationMs=round(frames / _FPS * 1000),
                sizeBytes=source.stat().st_size,
                audioStreamCount=1 if audio else 0,
                fps=_FPS,
            ),
        ),
        created_at=0,
    )
    params = _stored()
    spec: dict[str, JsonValue] = {
        "params": params.model_dump(mode="json"),
        "inputs": {"video": {"assetId": asset.id}},
    }
    if risk is not None:
        spec["_risk"] = risk
    generation = GenerationRecord(
        id="gen",
        feature=_ID,
        contract_version=1,
        status="running",
        spec=spec,
        created_at=0,
        queued_at=0,
        attempt_count=1,
        started_at=0,
        outputs=(),
    )

    class _Assets:
        def get_asset(self, asset_id: str) -> AssetRecord | None:
            return asset if asset_id == asset.id else None

    model = _GridModel(test_state.ic_lora)
    executor = IcLoraRecipeExecutor(
        model, _Assets(), lambda: test_state.config.default_models_dir, catalog_id=_ID
    )
    plan = executor.plan_outputs(generation)[0]
    output = tmp_path / "output.mp4"
    executor.execute(
        generation,
        (OutputAllocation(plan=plan, asset_id="asset", dest_path=str(output)),),
        FakeJobSession(),
    )
    return output, model


def test_an_off_grid_clip_is_padded_up_to_the_grid_and_cut_back_to_its_own_length(
    test_state, catalog, tmp_path: Path
) -> None:
    """The bug is the tail of a clip lost to the pipeline's round-down (87 in, 81 out)."""
    source = write_pattern_clip(tmp_path / "source.mp4", frames=87, audio=True)

    output, model = _run(test_state, tmp_path, source, frames=87, audio=True)

    assert model.received_frames == 89
    assert video_frame_count(output) == 87
    assert first_audio_codec(output) == "aac"


def test_a_clip_on_the_grid_is_given_to_the_model_as_it_is(
    test_state, catalog, tmp_path: Path
) -> None:
    source = write_pattern_clip(tmp_path / "source.mp4", frames=89)

    output, model = _run(test_state, tmp_path, source, frames=89, audio=False)

    assert model.received_frames == 89
    assert video_frame_count(output) == 89


def test_a_clip_whose_padded_count_is_over_budget_runs_the_snapped_down_count(
    test_state, catalog, tmp_path: Path
) -> None:
    """The create-time gate admitted it at the snapped-down count. The run must use that count."""
    limit = _first_refused_count(test_state)
    clip_frames = limit - 3
    source = write_pattern_clip(tmp_path / "source.mp4", frames=clip_frames)

    output, model = _run(test_state, tmp_path, source, frames=clip_frames, audio=False)

    assert model.received_frames == clip_frames
    assert video_frame_count(output) == limit - 8


_OVER_BET: dict[str, JsonValue] = {
    "tier": "over",
    "shortfall_kind": None,
    "over_gib": None,
    "over_ram_gb": None,
    "message": "Over budget.",
}


def test_a_stored_over_bet_runs_the_snapped_up_count_after_a_switch_to_cautious_mode(
    test_state, catalog, tmp_path: Path
) -> None:
    """The bug is the executor gating on the live mode: a queued job that create admitted fails."""
    limit = _first_refused_count(test_state)
    clip_frames = limit + 3  # both the snapped-down count and the snapped-up count are over
    source = write_pattern_clip(tmp_path / "source.mp4", frames=clip_frames)
    test_state.state.app_settings.budget_mode = "cautious"

    output, model = _run(
        test_state, tmp_path, source, frames=clip_frames, audio=False, risk=_OVER_BET
    )

    assert model.received_frames == limit + 8
    assert video_frame_count(output) == clip_frames


def test_a_job_stored_without_a_bet_is_refused_after_a_switch_to_brave_mode(
    test_state, catalog, tmp_path: Path
) -> None:
    """The bug is the executor gating on the live mode: it runs a job that create refused."""
    limit = _first_refused_count(test_state)
    clip_frames = limit + 3
    source = write_pattern_clip(tmp_path / "source.mp4", frames=clip_frames)
    test_state.state.app_settings.budget_mode = "brave"

    with pytest.raises(CapabilityFailedError):
        _run(test_state, tmp_path, source, frames=clip_frames, audio=False)
