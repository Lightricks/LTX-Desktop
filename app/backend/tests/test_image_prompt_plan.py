"""Where the rewrite of an image prompt honors a cancel and gives back the GPU lease.

Each test names the bug it catches. The rewrite matrix is tested through the queue in
test_text_to_image.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from handlers.image_prompt_plan import plan_fal_image_prompt
from services.generation_interrupt import GenerationCancelledError


def _cancelled() -> None:
    raise GenerationCancelledError()


@pytest.mark.parametrize("gemma", [True, False], ids=["gemma", "gemini"])
def test_a_cancel_before_the_rewrite_stops_it_and_releases_the_lease(
    test_state, fake_services, create_fake_model_files, gemma: bool
) -> None:
    # Bug caught: a cancel that comes during planning still loads Gemma and holds the GPU lease.
    settings = test_state.state.app_settings
    settings.explore_auto_enhance_prompts = True
    if gemma:
        create_fake_model_files(include_prompt_enhancer=True)
    else:
        settings.prompt_enhancer_provider_preference = "api"
        settings.gemini_api_key = "gemini-key"
    leased = {"taken": False, "released": False}

    @contextmanager
    def lease() -> Iterator[bool]:
        leased["taken"] = True
        try:
            yield True
        finally:
            leased["released"] = True

    with pytest.raises(GenerationCancelledError):
        plan_fal_image_prompt(
            test_state.prompt_enhancement,
            "a fox",
            provenance="typed",
            generation_seed=1,
            explicit_generation_seed=False,
            lease_idle_gpu=lease,
            raise_if_cancelled=_cancelled,
        )

    assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
    assert test_state.http.calls == []
    assert leased["taken"] is gemma
    assert leased["released"] is gemma
