"""Frame count of a queued Home recipe when snapped-up is unmeasured and snapped-down is over.

At 4K on a 32 GiB Windows card with a 73-frame window, one unchunked 73-frame pass fits decode
(32.73 GiB ask, ``over``), while an 81-frame clip spans two windows so ``choose_decode_window``
steps the window down to 65 frames (31.52 GiB ask, ``unmeasured`` for ``refine-details``). A clip
between 73 and 81 frames snaps down to the ``over`` count (73) and up to the ``unmeasured`` count (81).
"""

from __future__ import annotations

import pytest

from _routes._errors import HTTPError
from api_types import IcLoraRecipeStoredParams, IcLoraSettings, LoraEntry
from runtime_config.budget_admission import RiskAllowance
from runtime_config.ic_lora_tiling import IcLoraTiling
from tests.lora_catalog_helpers import add_ic_lora
from tests.test_ic_lora_recipe_run_frames import Gate, Runner, _configure_windows

_REFINE_ID = "refine-details"


class _UpscaleRunner(Runner):
    def run(
        self, clip_frames: int, *, width: int = 1920, height: int = 1080, **kwargs: object
    ) -> tuple[int, object]:
        return self._state.ic_lora.recipe_run_frames(
            ic_lora_id=_REFINE_ID,
            source_width=width,
            source_height=height,
            stored=IcLoraRecipeStoredParams(
                prompt="p",
                model="ltx-2.5-fast",
                resolution=self.gate.resolution,
                audioMode="generated",
                fps=24,
                loras=[LoraEntry(ref="", scale=1.0, catalogId=_REFINE_ID)],
            ),
            clip_frames=clip_frames,
            upscale=True,
            **kwargs,
        )


@pytest.fixture
def mac_runner(test_state, fake_services, create_fake_model_files) -> Runner:
    create_fake_model_files()
    _configure_windows(test_state, vram_gib=32, total_ram_gib=64)
    gate = Gate(
        IcLoraSettings(
            skip_stage_2=True,
            resolution_factor=1.0,
            tiling=IcLoraTiling(long_side=1920, short_side=1088, window_frames=73),
        ),
        "2160p",
        "over",
        {},
    )
    add_ic_lora(fake_services, id=_REFINE_ID, default_settings=gate.settings)
    return _UpscaleRunner(test_state, gate)


def _first_unmeasured_count(runner: Runner, test_state) -> int:
    test_state.state.app_settings.budget_mode = "brave"
    for clip in range(73, 2000, 8):
        _, risk = runner.run(clip)
        if risk is not None and risk.tier == "unmeasured":
            return clip
    raise AssertionError("the Mac cell never reached an unmeasured count")


@pytest.mark.parametrize("above_cap", [1, 5])
@pytest.mark.parametrize("live_mode", ["cautious", "brave"])
def test_an_admitted_unmeasured_job_still_runs_when_its_snapped_down_count_is_refused(
    mac_runner: Runner, test_state, above_cap: int, live_mode: str
) -> None:
    """The bug is the down probe raising: create stored only the unmeasured bet, which does not
    cover the over count, so execute failed a job that create admitted."""
    snapped_up = _first_unmeasured_count(mac_runner, test_state)
    snapped_down = snapped_up - 8
    clip = snapped_down + above_cap
    _, down_risk = mac_runner.run(snapped_down)
    assert down_risk is not None and down_risk.tier == "over", "the pool must make the cap over"

    created_count, created_risk = mac_runner.run(clip)
    assert created_count == snapped_up
    assert created_risk is not None and created_risk.tier == "unmeasured"

    test_state.state.app_settings.budget_mode = live_mode
    count, risk = mac_runner.run(clip, allowance=RiskAllowance(over=False, unmeasured=True))
    assert count == snapped_up
    assert risk is not None and risk.tier == "unmeasured"


def test_a_refused_snapped_down_count_does_not_hide_the_snapped_up_refusal(
    mac_runner: Runner, test_state
) -> None:
    """The bug is the double refusal carrying the message of the count the job did not ask for."""
    snapped_up = _first_unmeasured_count(mac_runner, test_state)
    snapped_down = snapped_up - 8
    test_state.state.app_settings.budget_mode = "cautious"

    def refusal(clip_frames: int) -> str:
        with pytest.raises(HTTPError) as exc_info:
            mac_runner.run(clip_frames)
        return exc_info.value.detail

    up_only, down_only = refusal(snapped_up), refusal(snapped_down)
    assert up_only != down_only, "the two counts must refuse for different reasons"
    assert refusal(snapped_down + 1) == up_only
