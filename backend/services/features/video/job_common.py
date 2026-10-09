"""Shared queued-video-job helpers (t2v / i2v / a2v / retake / extend)."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from api_types import LTXLocalModelId, OfferingId
from _routes._errors import HTTPError
from runtime_config.ltx_capabilities import LtxCapabilityFeature, local_caps, supports
from runtime_config.offerings import resolve_offering_local_model_id
from services.records import AssetRecord, CapabilityFailedError
from state.app_settings import MAX_GENERATION_SEED

INPUT_ASSET_NOT_FOUND = "INPUT_ASSET_NOT_FOUND"

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
