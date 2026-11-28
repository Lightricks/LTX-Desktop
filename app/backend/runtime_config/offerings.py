"""Offering ids (`ltx-2.5-fast`) vs local checkpoints and future API ids.

GenSpace/VideoEditor keep using pipeline ids (`fast` / `pro`). Desktop and Remote
create use these offering ids so a later API path can share the same `params.model` field.
"""

from __future__ import annotations

from pathlib import Path

from api_types import OfferingId, LTXLocalModelId, LTXVideoGenPipeline
from runtime_config.model_download_specs import ltx_generation_bundle_on_disk

# Newest first — same order as ALL_LTX_LOCAL_MODEL_IDS.
OFFERING_IDS: tuple[OfferingId, ...] = ("ltx-2.5-fast", "ltx-2.3-fast")

# Preferred checkpoint first within an offering (2.3 1.1 before 2.3 1.0).
OFFERING_LOCAL_MODEL_IDS: dict[OfferingId, tuple[LTXLocalModelId, ...]] = {
    "ltx-2.5-fast": ("ltx-2.5-22b-distilled",),
    "ltx-2.3-fast": ("ltx-2.3-22b-distilled-1.1", "ltx-2.3-22b-distilled"),
}

# ltxv-api ids. Unused in v1 (create stays local); keep next to the offering table
# so a cloud path does not invent a second mapping.
OFFERING_TO_API_MODEL: dict[OfferingId, str] = {
    "ltx-2.3-fast": "ltx-2-3-fast",
    "ltx-2.5-fast": "ltx-2-5-fast",
}

# Home/Remote create remaps offering → this pipeline id at execute. One owner so
# T2V/I2V/A2V cannot diverge when a third offering is added.
LOCAL_OFFERING_PIPELINE: LTXVideoGenPipeline = "fast"


def local_pipeline_for_offering(_offering: OfferingId) -> LTXVideoGenPipeline:
    return LOCAL_OFFERING_PIPELINE


def offering_id_for_local_model_id(model_id: LTXLocalModelId) -> OfferingId | None:
    for offering, ids in OFFERING_LOCAL_MODEL_IDS.items():
        if model_id in ids:
            return offering
    return None


def resolve_offering_local_model_id(
    models_dir: Path, offering: OfferingId
) -> LTXLocalModelId | None:
    for model_id in OFFERING_LOCAL_MODEL_IDS[offering]:
        if ltx_generation_bundle_on_disk(models_dir, model_id):
            return model_id
    return None
