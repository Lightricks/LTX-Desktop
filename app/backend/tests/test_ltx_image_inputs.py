"""Desktop stills map to ltx-pipelines image bindings."""

from __future__ import annotations

from api_types import ImageConditioningInput
from ltx_pipelines.utils.types import ImageGuide, ImageReplace
from services.ltx_pipeline_common import to_ltx_images


def test_first_frame_replaces_and_later_frames_guide() -> None:
    images = to_ltx_images(
        [ImageConditioningInput("a.png", 0, 0.9), ImageConditioningInput("b.png", 48, 1.0)]
    )

    assert [image.binding for image in images] == [ImageReplace(0), ImageGuide(48)]
    assert [image.strength for image in images] == [0.9, 1.0]


def test_guide_all_keeps_the_first_frame_as_a_keyframe() -> None:
    """Multi-keyframe interpolation regressed to replace-at-0 when frame 0 was not a guide."""
    images = to_ltx_images(
        [ImageConditioningInput("a.png", 0, 1.0), ImageConditioningInput("b.png", 48, 1.0)], guide_all=True
    )

    assert [image.binding for image in images] == [ImageGuide(0), ImageGuide(48)]


def test_reference_still_at_minus_one_is_a_guide_not_a_replace() -> None:
    """A replace binding rejects a negative frame, which broke Layout To Render."""
    (image,) = to_ltx_images([ImageConditioningInput("look.png", -1, 1.0)])

    assert image.binding == ImageGuide(-1)


def test_fast_pipeline_passes_guide_all_to_the_ltx_bindings() -> None:
    """Multi-keyframe interpolation fell back to replace-at-0 when ``_run_inference`` ignored the flag."""
    from types import SimpleNamespace

    from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline

    seen: list[object] = []

    def pipeline(**kwargs: object) -> SimpleNamespace:
        seen.append(kwargs["images"])
        return SimpleNamespace(video=iter(()), audio=None, num_frames=9, tiling_config=None)

    runner = LTXFastVideoPipeline.__new__(LTXFastVideoPipeline)
    runner.pipeline = pipeline  # type: ignore[assignment]

    runner._run_inference(  # pyright: ignore[reportPrivateUsage]
        prompt="p", seed=1, height=64, width=64, num_frames=9, frame_rate=24.0,
        images=[ImageConditioningInput("a.png", 0, 1.0), ImageConditioningInput("b.png", 8, 1.0)],
        tiling_config=None, guide_all_images=True,  # type: ignore[arg-type]
    )

    assert [image.binding for image in seen[0]] == [ImageGuide(0), ImageGuide(8)]  # type: ignore[attr-defined]
