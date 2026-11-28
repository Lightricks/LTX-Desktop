"""Queued Home Motion Track recipe (one image and a path).

Draws the control clip from the path at the denoise size, then runs the catalog IC-LoRA
path on it. The image is the first frame and the picture the path moves over.
"""

from __future__ import annotations

from pathlib import Path

from api_types import (
    IcLoraImageInput,
    IcLoraSynthesizedStoredParams,
    IcLoraTrackInputs,
)
from services.features.video.job_common import require_existing_asset
from services.features.video.motion_path import render_motion_track
from services.features.video.synthesized_ic_lora import (
    SynthesizedIcLoraExecutor,
    SynthesizedJob,
)


class IcLoraTrackRecipeExecutor(SynthesizedIcLoraExecutor[IcLoraTrackInputs]):
    inputs_model = IcLoraTrackInputs
    control_suffix = "track.mp4"

    def prepare(
        self,
        params: IcLoraSynthesizedStoredParams,
        inputs: IcLoraTrackInputs,
        canvas: tuple[int, int],
        control: Path,
    ) -> SynthesizedJob:
        image = require_existing_asset(
            self._assets,
            inputs.image.assetId,
            media_kind="image",
            unavailable="image is no longer available",
        )
        width, height = canvas
        render_motion_track(
            inputs.paths,
            width=width,
            height=height,
            duration=params.duration,
            fps=int(params.fps),
            dest=control,
        )
        return SynthesizedJob(
            self.generate_request(
                self._recipe,
                params,
                canvas,
                input_path=str(control),
                # No request frame. The handler uses the catalog pin (frame 0).
                images=[IcLoraImageInput(path=image.path)],
            )
        )
