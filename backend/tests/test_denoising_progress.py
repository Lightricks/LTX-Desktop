"""Denoise-step totals and in-flight progress reporting.

Totals must come from the same sigma schedules the pipelines run, not a guessed
duration or a hardcoded Fast-only step count.
"""

from __future__ import annotations

from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

from services.denoising_progress import (
    DISTILLED_STAGE_1_STEPS,
    DISTILLED_STAGE_2_STEPS,
    DISTILLED_TOTAL_STEPS,
    distilled_total_steps,
    note_step,
    reset,
    track_denoising,
)
from state.app_state_types import GpuSlot, VideoPipelineState
from tests.fakes.services import FakeFastVideoPipeline


def test_two_stage_distilled_total_matches_sigma_schedules() -> None:
    expected = (len(DISTILLED_SIGMA_VALUES) - 1) + (len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1)
    assert DISTILLED_STAGE_1_STEPS == 8
    assert DISTILLED_STAGE_2_STEPS == 3
    assert DISTILLED_TOTAL_STEPS == 11
    assert distilled_total_steps() == expected
    assert distilled_total_steps() == DISTILLED_TOTAL_STEPS


def test_stage_1_only_total_omits_stage_2() -> None:
    assert distilled_total_steps(stage_2=False) == DISTILLED_STAGE_1_STEPS
    assert distilled_total_steps(stage_2=False) == len(DISTILLED_SIGMA_VALUES) - 1
    assert distilled_total_steps(stage_2=False) == distilled_total_steps() - DISTILLED_STAGE_2_STEPS


def test_note_step_reports_current_and_total_to_the_sink() -> None:
    seen: list[tuple[int, int]] = []
    from services import denoising_progress

    denoising_progress.configure_sink(lambda current, total: seen.append((current, total)))
    total = distilled_total_steps()
    with track_denoising(total):
        note_step()
        note_step()
    reset()
    assert seen[0] == (0, total)
    assert seen[1] == (1, total)
    assert seen[2] == (2, total)


def test_denoise_steps_fill_generation_progress(test_state) -> None:
    pipeline = FakeFastVideoPipeline()
    test_state.state.gpu_slot = GpuSlot(
        active_pipeline=VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
            ltx_model_id="ltx-2.5-22b-distilled",
            loading_mode="full_models_loading",
        ),
    )
    test_state.generation.start_generation("gen-steps")
    total = distilled_total_steps()
    with track_denoising(total):
        note_step()
        note_step()
    progress = test_state.generation.get_generation_progress()
    assert progress.phase == "inference"
    assert progress.currentStep == 2
    assert progress.totalSteps == total
    assert progress.progress == 15 + int(75 * 2 / total)
