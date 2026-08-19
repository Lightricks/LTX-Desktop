"""CPU-only checks for distilled A2V image CRF resolution.

Image+audio generation failed because DistilledA2VPipeline built
ImageConditioningInput tuples with crf=None and never called
ImageConditioner.resolve_crf, which DistilledPipeline runs for image-only.
"""

from __future__ import annotations

from collections.abc import Sequence

from ltx_pipelines.utils.args import ImageConditioningInput as LtxImageInput
from services.a2v_pipeline.distilled_a2v_pipeline import resolve_image_conditionings

_CHECKPOINT_CRF = 29


class _FakeImageConditioner:
    def __init__(self) -> None:
        self.received: list[LtxImageInput] | None = None

    def resolve_crf(self, images: Sequence[LtxImageInput]) -> list[LtxImageInput]:
        self.received = list(images)
        return [
            image if image.crf is not None else image._replace(crf=_CHECKPOINT_CRF)
            for image in images
        ]


def test_resolve_image_conditionings_builds_inputs_and_calls_resolve_crf() -> None:
    conditioner = _FakeImageConditioner()

    result = resolve_image_conditionings(
        [("start.png", 0, 0.8), ("end.png", 24, 1.0)],
        conditioner,
    )

    assert conditioner.received is not None
    assert [(image.path, image.frame_idx, image.strength, image.crf) for image in conditioner.received] == [
        ("start.png", 0, 0.8, None),
        ("end.png", 24, 1.0, None),
    ]
    assert [(image.path, image.frame_idx, image.strength, image.crf) for image in result] == [
        ("start.png", 0, 0.8, _CHECKPOINT_CRF),
        ("end.png", 24, 1.0, _CHECKPOINT_CRF),
    ]
