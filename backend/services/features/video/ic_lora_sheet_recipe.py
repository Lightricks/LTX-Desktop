"""Queued Home image-list IC-LoRA recipe (Ingredients).

Stitches the panels into one reference sheet at the denoise size, then runs the
catalog IC-LoRA path on it. The catalog path loops the sheet for ``duration`` seconds.
"""

from __future__ import annotations

from pathlib import Path

from api_types import IcLoraSheetInputs, IcLoraSynthesizedStoredParams
from services.features.video.job_common import require_existing_asset
from services.features.video.reference_sheet import ensure_reference_sheet
from services.features.video.synthesized_ic_lora import (
    SynthesizedIcLoraExecutor,
    SynthesizedJob,
)
from services.records import CapabilityFailedError


class IcLoraSheetRecipeExecutor(SynthesizedIcLoraExecutor[IcLoraSheetInputs]):
    inputs_model = IcLoraSheetInputs
    control_suffix = "sheet.png"
    loops_control_file = True

    def prepare(
        self,
        params: IcLoraSynthesizedStoredParams,
        inputs: IcLoraSheetInputs,
        canvas: tuple[int, int],
        control: Path,
    ) -> SynthesizedJob:
        image_paths = [
            require_existing_asset(
                self._assets,
                ref.assetId,
                media_kind="image",
                unavailable="image is no longer available",
            ).path
            for ref in inputs.images
        ]
        # The sheet goes beside the job output, in the store's in-flight folder.
        in_flight_dir = control.parent
        try:
            sheet = ensure_reference_sheet(
                image_paths, *canvas, in_flight_dir=in_flight_dir
            )
        except (OSError, ValueError) as exc:
            # A corrupt or unreadable panel image (Pillow raises OSError subclasses).
            raise CapabilityFailedError(
                "image is not readable", code="INVALID_GENERATION_SPEC"
            ) from exc
        return SynthesizedJob(
            self.generate_request(self._recipe, params, canvas, input_path=str(sheet)),
            panel_paths=image_paths,
            control_path=sheet,
        )

