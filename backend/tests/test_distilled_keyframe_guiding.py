"""CPU-only checks for the Distilled MKF guiding-helper swap."""

from __future__ import annotations

import pytest

import ltx_pipelines.chunks.conditionings as chunk_conditionings
import ltx_pipelines.utils.helpers as helpers
from ltx_pipelines.utils.helpers import image_conditionings_by_adding_guiding_latent
from services.fast_video_pipeline.distilled_keyframe_guiding import distilled_keyframe_guiding


def test_guiding_context_swaps_combined_helper() -> None:
    original_helpers = helpers.combined_image_conditionings
    original_chunks = chunk_conditionings.combined_image_conditionings
    with distilled_keyframe_guiding():
        assert helpers.combined_image_conditionings is image_conditionings_by_adding_guiding_latent
        assert chunk_conditionings.combined_image_conditionings is image_conditionings_by_adding_guiding_latent
    assert helpers.combined_image_conditionings is original_helpers
    assert chunk_conditionings.combined_image_conditionings is original_chunks


def test_guiding_context_restores_helper_after_exception() -> None:
    original_helpers = helpers.combined_image_conditionings
    original_chunks = chunk_conditionings.combined_image_conditionings
    with pytest.raises(RuntimeError, match="swap-failed"):
        with distilled_keyframe_guiding():
            assert helpers.combined_image_conditionings is image_conditionings_by_adding_guiding_latent
            assert chunk_conditionings.combined_image_conditionings is image_conditionings_by_adding_guiding_latent
            raise RuntimeError("swap-failed")
    assert helpers.combined_image_conditionings is original_helpers
    assert chunk_conditionings.combined_image_conditionings is original_chunks
