"""Shared queued-video-job helpers (t2v / i2v / a2v / retake / extend)."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from api_types import (
    IcLoraRecipeResolution,
    IcLoraSynthesizedStoredParams,
    LTXLocalModelId,
    LTXVideoGenAspectRatio,
    OfferingId,
)
from _routes._errors import HTTPError
from runtime_config.budget_admission import (
    RiskAllowance,
    stored_generation_risk as stored_generation_risk,
)
from runtime_config.ic_lora_high_res import high_res_cell
from runtime_config.ltx_capabilities import LtxCapabilityFeature, budget_pixels, local_caps, supports
from runtime_config.reject_reasons import PassShape
from runtime_config.offerings import resolve_offering_local_model_id
from services.records import AssetRecord, CapabilityFailedError
from state.app_settings import MAX_GENERATION_SEED

INPUT_ASSET_NOT_FOUND = "INPUT_ASSET_NOT_FOUND"
# Private spec key beside ``_risk``: the Fast shape create resolved.
PASS_SHAPE_SPEC_KEY = "_passShape"

TModel = TypeVar("TModel", bound=BaseModel)


class AssetLookup(Protocol):
    def get_asset(self, asset_id: str) -> AssetRecord | None: ...


def is_existing_file(path: str) -> bool:
    try:
        return Path(path).is_file()
    except OSError:
        return False


def looks_like_filesystem_path(detail: str) -> bool:
    stripped = detail.strip()
    if not stripped:
        return False
    if Path(stripped).is_absolute():
        return True
    return "/" in stripped or "\\" in stripped


def capability_failed_from_http(
    exc: HTTPError, *, unavailable: str = "input media is no longer available"
) -> CapabilityFailedError:
    if looks_like_filesystem_path(exc.detail):
        return CapabilityFailedError(unavailable, code=INPUT_ASSET_NOT_FOUND)
    return CapabilityFailedError(exc.detail, code=exc.code)


def model_from_spec(
    spec: Mapping[str, object], key: str, model: type[TModel]
) -> TModel:
    value = spec.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"generation spec requires a {key} object")
    return model.model_validate(value)


def spec_risk_allowance(spec: Mapping[str, object]) -> RiskAllowance:
    """The stored bet as a RiskAllowance.

    Only ``over`` waives a later memory shortfall; ``unmeasured`` waives an
    untested fence without covering a VRAM or RAM miss. A missing field or junk
    fails closed. Executors pass this — never the live mode — so switching to
    Cautious does not cancel a job already in the queue.
    """
    return RiskAllowance.from_stored(spec)


def spec_pass_shape(spec: Mapping[str, object]) -> PassShape:
    """The shape create resolved, as stored.

    Executors pass this, never the live "Never split" setting, so a flip does
    not re-cut a queued job. A missing field (jobs queued before shapes were
    stored) or junk means ``auto``.
    """
    match spec.get(PASS_SHAPE_SPEC_KEY):
        case "one_pass":
            return "one_pass"
        case "parts":
            return "parts"
        case _:
            return "auto"


def resolve_generation_seed(seed: int | None) -> int:
    return seed if seed is not None else random.randint(0, MAX_GENERATION_SEED)


def resolve_job_local_model_id(
    models_dir: Path, offering: OfferingId
) -> LTXLocalModelId:
    resolved = resolve_offering_local_model_id(models_dir, offering)
    if resolved is None:
        raise HTTPError(409, "LTX_MODEL_NOT_INSTALLED")
    return resolved


def require_capable_local_model(
    models_dir: Path,
    offering: OfferingId,
    capability: LtxCapabilityFeature,
    *,
    unsupported_code: str,
    unsupported_message: str,
) -> LTXLocalModelId:
    local_id = resolve_job_local_model_id(models_dir, offering)
    if not supports(local_caps(local_id), capability):
        raise CapabilityFailedError(unsupported_message, code=unsupported_code)
    return local_id


def require_existing_asset(
    assets: AssetLookup,
    asset_id: str,
    *,
    media_kind: str,
    unavailable: str,
) -> AssetRecord:
    asset = assets.get_asset(asset_id)
    if asset is None or asset.media_kind != media_kind or not is_existing_file(asset.path):
        raise CapabilityFailedError(unavailable, code=INPUT_ASSET_NOT_FOUND)
    return asset


# The name of a recipe whose display name the caller cannot reach. The catalog owns the real name.
UNNAMED_RECIPE = "This recipe"


def high_res_unsupported_error(name: str) -> HTTPError:
    """The refusal of a recipe that cannot run as a tiled high cell. One wording for every recipe."""
    return HTTPError(
        422,
        f"{name} cannot render above 1080p. Pick a lower resolution.",
        code="IC_LORA_HIGH_RES_UNSUPPORTED",
    )


def denoise_canvas_for(
    resolution: IcLoraRecipeResolution,
    aspect_ratio: LTXVideoGenAspectRatio,
    *,
    name: str,
    local_model_id: LTXLocalModelId = "ltx-2.5-22b-distilled",
) -> tuple[int, int]:
    """The size the model denoises. The sheet and the control clip use it."""
    # A high cell runs as spatial tiles, and each tile sees only its own slice of the
    # sheet, so the panels would not stay whole. Queued or retried specs from before
    # the cells were removed reach this helper, so it owns the rejection.
    if high_res_cell(resolution) is not None:
        raise high_res_unsupported_error(name)
    try:
        return budget_pixels(
            local_caps(local_model_id), resolution, aspect_ratio, mode="video"
        )
    except KeyError as exc:
        raise HTTPError(400, "INVALID_LOCAL_RESOLUTION", code="INVALID_LOCAL_RESOLUTION") from exc


def denoise_canvas(
    local_model_id: LTXLocalModelId, params: IcLoraSynthesizedStoredParams, *, name: str
) -> tuple[int, int]:
    return denoise_canvas_for(
        params.resolution, params.aspectRatio, name=name, local_model_id=local_model_id
    )
