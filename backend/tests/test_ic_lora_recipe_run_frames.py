"""Frame count a queued Home recipe runs, priced by the real job budget.

Each test finds the first frame count the real gate refuses, then checks the counts around it.
A change to the budget numbers moves that count and the tests follow it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from _routes._errors import HTTPError
from api_types import (
    IcLoraRecipeStoredParams,
    IcLoraSettings,
    LoraEntry,
    VideoAssetMetadata,
    VideoMeta,
)
from runtime_config.budget_admission import RiskAllowance
from runtime_config.video_job_budget import VIDEO_JOB_TOO_LARGE
from services.records import AssetRecord
from tests.fakes.services import FakeCapture
from tests.lora_catalog_helpers import add_ic_lora

_ID = "run-frames-test"
_GIB = 1024**3


@dataclass(frozen=True)
class Gate:
    """A catalog shape, and the risk tier Brave mode gives a count the gate refuses."""

    settings: IcLoraSettings
    resolution: str
    tier: str
    flags: dict[str, object]  # the stored allowance that tier leaves on the generation


# A single-stage job is refused for memory. A two-stage job is refused by a fence that moves
# with the frame count, and that Brave mode runs as an unmeasured bet.
_GATES = {
    "memory": Gate(
        IcLoraSettings(skip_stage_2=True, resolution_factor=1.0, audio_mode="source"),
        "540p", "over", {"allowance": RiskAllowance(over=True, unmeasured=False)},
    ),
    "fence": Gate(
        IcLoraSettings(skip_stage_2=False, resolution_factor=1.0, audio_mode="source"),
        "1080p", "unmeasured", {"allowance": RiskAllowance(over=False, unmeasured=True)},
    ),
}


def _stored(resolution: str) -> IcLoraRecipeStoredParams:
    return IcLoraRecipeStoredParams(
        prompt="p",
        model="ltx-2.5-fast",
        resolution=resolution,
        audioMode="generated",
        fps=24,
        loras=[LoraEntry(ref="", scale=1.0, catalogId=_ID)],
    )


def _configure_windows(test_state, *, vram_gib: float = 32, total_ram_gib: float = 64) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = int(vram_gib * _GIB)
    test_state.config.total_ram_bytes = int(total_ram_gib * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"


class Runner:
    def __init__(self, test_state, gate: Gate) -> None:
        self._state = test_state
        self.gate = gate

    def run(
        self, clip_frames: int, *, width: int = 1920, height: int = 1080, **kwargs: object
    ) -> tuple[int, object]:
        return self._state.ic_lora.recipe_run_frames(
            ic_lora_id=_ID,
            source_width=width,
            source_height=height,
            stored=_stored(self.gate.resolution),
            clip_frames=clip_frames,
            **kwargs,
        )

    def first_refused(self, *, width: int = 1920, height: int = 1080, cutout: bool = False) -> int:
        """First on-grid count the gate refuses, or runs with a risk."""
        for clip in range(9, 20000, 8):
            try:
                _, risk = self.run(clip, width=width, height=height, cutout=cutout)
            except HTTPError:
                return clip
            if risk is not None:
                return clip
        raise AssertionError("the gate never refused a count")

    def refused(self, clip_frames: int, mode: str, **kwargs: object) -> None:
        """The count is refused: a 422 in cautious mode, the gate's risk in brave mode."""
        if mode == "cautious":
            with pytest.raises(HTTPError) as exc_info:
                self.run(clip_frames, **kwargs)
            assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
        else:
            _, risk = self.run(clip_frames, **kwargs)
            assert risk is not None and risk.tier == self.gate.tier


@pytest.fixture(params=sorted(_GATES))
def runner(request, test_state, fake_services, create_fake_model_files) -> Runner:
    gate = _GATES[request.param]
    create_fake_model_files()
    _configure_windows(test_state)
    add_ic_lora(fake_services, id=_ID, default_settings=gate.settings)
    return Runner(test_state, gate)


@pytest.fixture(params=["cautious", "brave"])
def mode(request, test_state) -> str:
    test_state.state.app_settings.budget_mode = request.param
    return request.param


def test_a_count_that_fits_snaps_up_so_no_frame_is_dropped(runner: Runner, mode: str) -> None:
    last_fit = runner.first_refused() - 8

    assert runner.run(last_fit - 3) == (last_fit, None)
    assert runner.run(last_fit) == (last_fit, None)


def test_a_clip_below_the_floor_runs_the_floor_count(runner: Runner, mode: str) -> None:
    assert runner.run(5) == (9, None)


def test_a_snapped_up_count_over_budget_falls_back_to_the_clean_snapped_down_count(
    runner: Runner, mode: str
) -> None:
    """Brave returns a risk in place of the 422. The fallback must read that risk."""
    limit = runner.first_refused()

    assert runner.run(limit - 3) == (limit - 8, None)


def test_an_on_grid_count_over_budget_is_refused_or_runs_with_its_risk(
    runner: Runner, mode: str
) -> None:
    limit = runner.first_refused()

    runner.refused(limit, mode)


def test_when_both_counts_are_over_budget_the_snapped_up_count_is_refused_or_runs_with_its_risk(
    runner: Runner, mode: str
) -> None:
    limit = runner.first_refused()

    runner.refused(limit + 3, mode)
    if mode == "brave":
        assert runner.run(limit + 3)[0] == limit + 8


def test_a_recipe_that_always_pads_never_falls_back_to_snapped_down(
    runner: Runner, mode: str
) -> None:
    limit = runner.first_refused(cutout=True)

    runner.refused(limit - 3, mode, cutout=True)
    if mode == "brave":
        assert runner.run(limit - 3, cutout=True)[0] == limit
    assert runner.run(limit - 11, cutout=True) == (limit - 8, None)


def test_the_stored_allowance_and_not_the_live_mode_gates_the_run(
    runner: Runner, test_state
) -> None:
    """Create admitted the job under one mode. A flip must not change what execute runs."""
    limit = runner.first_refused()
    flags = runner.gate.flags

    test_state.state.app_settings.budget_mode = "brave"
    # Create picked the clean snapped-down count, so nothing is stored. Execute must too.
    assert runner.run(limit - 3, allowance=RiskAllowance.none()) == (limit - 8, None)

    test_state.state.app_settings.budget_mode = "cautious"
    # Create ran the over-budget count with a risk, and stored it. Execute must keep it.
    count, risk = runner.run(limit, **flags)
    assert count == limit
    assert risk is not None and risk.tier == runner.gate.tier


def _asset(path: Path, *, width: int, height: int, frames: int) -> AssetRecord:
    return AssetRecord(
        id="v1",
        media_kind="video",
        origin="uploaded",
        path=str(path),
        mime_type="video/mp4",
        name="clip.mp4",
        metadata=VideoAssetMetadata(
            mediaType="video",
            metadata=VideoMeta(
                width=width,
                height=height,
                durationMs=round(frames / 24 * 1000),
                sizeBytes=100,
                audioStreamCount=0,
                fps=24,
            ),
        ),
        created_at=0,
    )


@pytest.fixture
def stage_2_runner(test_state, fake_services, create_fake_model_files) -> Runner:
    """Stage-2 IC-LoRA: its canvas fits the clip size, so a crop changes the canvas."""
    create_fake_model_files()
    _configure_windows(test_state)
    gate = Gate(
        IcLoraSettings(skip_stage_2=False, stage_2_ic_lora=True, resolution_factor=1.0),
        "540p", "over", {},
    )
    add_ic_lora(fake_services, id=_ID, default_settings=gate.settings)
    return Runner(test_state, gate)


# 1920x1200 is cropped to 3:2 (1800x1200) before the run. The 1800x1200 canvas is smaller.
_WIDE = {"width": 1920, "height": 1200}
_CROPPED = {"width": 1800, "height": 1200}


def test_the_budget_prices_the_canvas_of_the_center_cropped_clip(stage_2_runner: Runner) -> None:
    cropped_limit = stage_2_runner.first_refused(**_CROPPED)
    uncropped_limit = stage_2_runner.first_refused(**_WIDE, cutout=True)
    assert uncropped_limit < cropped_limit, "the crop must change the canvas for this test"

    assert stage_2_runner.first_refused(**_WIDE) == cropped_limit
    assert stage_2_runner.run(uncropped_limit, **_WIDE) == (uncropped_limit, None)
    with pytest.raises(HTTPError):
        stage_2_runner.run(uncropped_limit, **_WIDE, cutout=True)


def test_the_create_preflight_prices_the_same_canvas_as_the_run(
    stage_2_runner: Runner, test_state, fake_services, tmp_path: Path
) -> None:
    uncropped_limit = stage_2_runner.first_refused(**_WIDE, cutout=True)
    item = fake_services.lora_catalog_provider._catalog.ic_loras[-1]
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(path), FakeCapture(frames=["f"] * uncropped_limit, fps=24, **_WIDE)
    )
    video = _asset(path, **_WIDE, frames=uncropped_limit)
    stored = _stored("540p")

    risk = test_state.ic_lora.preflight_queued_recipe_job(
        item=item, video=video, stored=stored, source_fps=24.0
    )
    assert risk is None
    with pytest.raises(HTTPError) as exc_info:
        test_state.ic_lora.preflight_queued_recipe_job(
            item=item, video=video, stored=stored, source_fps=24.0, cutout=True
        )
    assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
