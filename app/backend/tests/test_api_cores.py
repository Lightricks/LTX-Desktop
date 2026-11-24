"""Desktop and Remote Asset/Generation cores share validation, not filesystem fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api_types import (
    Asset,
    AssetCore,
    Generation,
    GenerationCore,
    ImageAssetMetadata,
    ImageMeta,
    MediaKind,
)
from remote.dto import (
    RemoteAsset,
    RemoteAssetListItem,
    RemoteGeneration,
    RemoteGenerationSpec,
    to_remote_asset,
    to_remote_generation,
    to_remote_list_item,
)
from remote.media_urls import MediaUrlSigner

_SIGNER = MediaUrlSigner(b"s" * 32)
_DEVICE_ID = "dev-1"


_IMAGE_METADATA = ImageAssetMetadata(
    mediaType="image",
    metadata=ImageMeta(width=8, height=8),
)
_SECRET_PATH = "/Users/me/secret.png"
_SECRET_THUMB = "/Users/me/secret-thumb.png"


def _remote_asset(*, media_kind: MediaKind = "image") -> RemoteAsset:
    return RemoteAsset(
        id="asset-1",
        media_kind=media_kind,
        origin="uploaded",
        mime_type="image/png",
        name="x.png",
        metadata=_IMAGE_METADATA,
        created_at=1,
        bytes_url="/api/assets/asset-1/bytes?exp=1&did=dev-1&sig=abc",
    )


def _desktop_asset(*, media_kind: MediaKind = "image") -> Asset:
    return Asset(
        id="asset-1",
        media_kind=media_kind,
        origin="uploaded",
        path=_SECRET_PATH,
        thumbnail_path=_SECRET_THUMB,
        mime_type="image/png",
        name="x.png",
        metadata=_IMAGE_METADATA,
        created_at=1,
    )


def _desktop_generation(**overrides: object) -> Generation:
    values = {
        "id": "gen-1",
        "feature": "text-to-video",
        "contract_version": 1,
        "status": "queued",
        "error_code": None,
        "created_at": 1,
        "queued_at": 1,
        "attempt_count": 0,
        "started_at": None,
        "finished_at": None,
        "spec": {"params": {"prompt": "fox", "imagePath": _SECRET_PATH}},
        "outputs": [_desktop_asset()],
        **overrides,
    }
    return Generation.model_validate(values)


def _remote_generation(**overrides: object) -> RemoteGeneration:
    values = {
        "id": "gen-1",
        "feature": "text-to-video",
        "contract_version": 1,
        "status": "queued",
        "error_code": None,
        "created_at": 1,
        "queued_at": 1,
        "attempt_count": 0,
        "started_at": None,
        "finished_at": None,
        "spec": RemoteGenerationSpec(),
        "outputs": [_remote_asset()],
        **overrides,
    }
    return RemoteGeneration.model_validate(values)


def test_concrete_models_keep_distinct_spec_output_and_path_fields() -> None:
    assert set(Asset.model_fields) - set(AssetCore.model_fields) == {
        "path",
        "thumbnail_path",
    }
    assert set(RemoteAsset.model_fields) - set(AssetCore.model_fields) == {
        "bytes_url",
        "thumbnail_url",
    }
    assert "path" not in RemoteAsset.model_fields
    assert "thumbnail_path" not in RemoteAsset.model_fields
    assert set(RemoteAssetListItem.model_fields) - set(RemoteAsset.model_fields) == {
        "in_use",
        "has_thumbnail",
    }
    assert "path" not in RemoteAssetListItem.model_fields
    assert "thumbnail_path" not in RemoteAssetListItem.model_fields
    assert set(Generation.model_fields) - set(GenerationCore.model_fields) == {
        "spec",
        "outputs",
    }
    assert set(RemoteGeneration.model_fields) - set(GenerationCore.model_fields) == {
        "spec",
        "outputs",
    }


def test_asset_and_remote_asset_reject_media_kind_metadata_mismatch() -> None:
    with pytest.raises(ValidationError, match="media_kind"):
        _desktop_asset(media_kind="video")
    with pytest.raises(ValidationError, match="media_kind"):
        _remote_asset(media_kind="video")
    assert _desktop_asset().media_kind == "image"
    assert _remote_asset().media_kind == "image"


def test_generation_and_remote_generation_enforce_failed_error_code() -> None:
    with pytest.raises(ValidationError, match="failed generation requires error_code"):
        _desktop_generation(status="failed", error_code=None)
    with pytest.raises(ValidationError, match="failed generation requires error_code"):
        _remote_generation(status="failed", error_code=None)
    with pytest.raises(ValidationError, match="error_code is only valid when status is failed"):
        _desktop_generation(status="queued", error_code="EXECUTOR_FAILED")
    with pytest.raises(ValidationError, match="error_code is only valid when status is failed"):
        _remote_generation(status="queued", error_code="EXECUTOR_FAILED")
    assert (
        _desktop_generation(status="failed", error_code="EXECUTOR_FAILED").error_code
        == "EXECUTOR_FAILED"
    )
    assert (
        _remote_generation(status="failed", error_code="EXECUTOR_FAILED").error_code
        == "EXECUTOR_FAILED"
    )


def test_generation_and_remote_generation_allow_cancelling_status() -> None:
    assert _desktop_generation(status="cancelling").status == "cancelling"
    assert _remote_generation(status="cancelling").status == "cancelling"


def test_desktop_ignores_extra_fields_remote_forbids_them() -> None:
    asset_payload = {
        "id": "asset-1",
        "media_kind": "image",
        "origin": "uploaded",
        "path": _SECRET_PATH,
        "thumbnail_path": None,
        "mime_type": "image/png",
        "name": "x.png",
        "metadata": _IMAGE_METADATA,
        "created_at": 1,
        "unknown": "x",
    }
    assert Asset.model_validate(asset_payload).path == _SECRET_PATH
    remote_asset_payload = {
        key: value
        for key, value in asset_payload.items()
        if key not in {"path", "thumbnail_path"}
    }
    remote_asset_payload["bytes_url"] = "/api/assets/asset-1/bytes?exp=1&did=dev-1&sig=abc"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RemoteAsset.model_validate(remote_asset_payload)
    generation_payload = {
        "id": "gen-1",
        "feature": "text-to-video",
        "contract_version": 1,
        "status": "queued",
        "spec": {"params": {"prompt": "fox"}},
        "created_at": 1,
        "queued_at": 1,
        "attempt_count": 0,
        "outputs": [asset_payload],
        "unknown": "x",
    }
    assert Generation.model_validate(generation_payload).id == "gen-1"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RemoteGeneration.model_validate(
            {
                "id": "gen-1",
                "feature": "text-to-video",
                "contract_version": 1,
                "status": "queued",
                "spec": {"params": {}},
                "created_at": 1,
                "queued_at": 1,
                "attempt_count": 0,
                "outputs": [],
                "unknown": "x",
            }
        )


def test_remote_asset_rejects_filesystem_paths_as_extra_fields() -> None:
    payload = {
        "id": "asset-1",
        "media_kind": "image",
        "origin": "uploaded",
        "mime_type": "image/png",
        "name": "x.png",
        "metadata": _IMAGE_METADATA,
        "created_at": 1,
        "bytes_url": "/api/assets/asset-1/bytes?exp=1&did=dev-1&sig=abc",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RemoteAsset.model_validate({**payload, "path": _SECRET_PATH})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RemoteAsset.model_validate({**payload, "thumbnail_path": _SECRET_THUMB})


def test_projections_omit_filesystem_paths() -> None:
    dumped_asset = to_remote_asset(
        _desktop_asset(), signer=_SIGNER, device_id=_DEVICE_ID
    ).model_dump()
    assert "path" not in dumped_asset
    assert "thumbnail_path" not in dumped_asset
    assert dumped_asset["bytes_url"].startswith("/api/assets/asset-1/bytes?")
    assert dumped_asset["thumbnail_url"].startswith(
        "/api/assets/asset-1/thumbnail/bytes?"
    )
    assert _SECRET_PATH not in str(dumped_asset)
    assert _SECRET_THUMB not in str(dumped_asset)

    dumped_generation = to_remote_generation(
        _desktop_generation(), signer=_SIGNER, device_id=_DEVICE_ID
    ).model_dump(mode="json")
    assert "path" not in dumped_generation
    assert dumped_generation["outputs"][0]["id"] == "asset-1"
    assert "path" not in dumped_generation["outputs"][0]
    assert "thumbnail_path" not in dumped_generation["outputs"][0]
    assert _SECRET_PATH not in str(dumped_generation)
    assert "imagePath" not in dumped_generation["spec"]["params"]


def test_remote_generation_projects_prompt_provenance() -> None:
    dumped = to_remote_generation(
        _desktop_generation(
            spec={
                "params": {
                    "prompt": "fox",
                    "promptProvenance": "enhanced",
                    "imagePath": _SECRET_PATH,
                }
            }
        ),
        signer=_SIGNER,
        device_id=_DEVICE_ID,
    ).model_dump(mode="json")
    assert dumped["spec"]["params"]["prompt"] == "fox"
    assert dumped["spec"]["params"]["promptProvenance"] == "enhanced"
    assert "imagePath" not in dumped["spec"]["params"]
    assert _SECRET_PATH not in str(dumped)


def test_remote_params_keep_known_modes_and_drop_invalid_values() -> None:
    dumped = to_remote_generation(
        _desktop_generation(
            spec={
                "params": {
                    "prompt": "fox",
                    "mode": "replace_video",
                    "startTime": 1.5,
                }
            }
        ),
        signer=_SIGNER,
        device_id=_DEVICE_ID,
    ).model_dump(mode="json")
    assert dumped["spec"]["params"]["mode"] == "replace_video"
    assert dumped["spec"]["params"]["startTime"] == 1.5

    dropped = to_remote_generation(
        _desktop_generation(
            spec={
                "params": {
                    "prompt": "fox",
                    "mode": "not-a-mode",
                    "startTime": -1,
                }
            }
        ),
        signer=_SIGNER,
        device_id=_DEVICE_ID,
    ).model_dump(mode="json")
    assert dropped["spec"]["params"].get("mode") is None
    assert dropped["spec"]["params"].get("startTime") is None


def test_remote_resolution_keeps_tier_ids_and_target_sizes() -> None:
    tier = to_remote_generation(
        _desktop_generation(spec={"params": {"prompt": "fox", "resolution": "720p"}}),
        signer=_SIGNER,
        device_id=_DEVICE_ID,
    ).model_dump(mode="json")
    assert tier["spec"]["params"]["resolution"] == "720p"

    sized = to_remote_generation(
        _desktop_generation(
            spec={
                "params": {
                    "prompt": "fox",
                    "resolution": {"width": 1280, "height": 720, "path": _SECRET_PATH},
                }
            }
        ),
        signer=_SIGNER,
        device_id=_DEVICE_ID,
    ).model_dump(mode="json")
    assert sized["spec"]["params"]["resolution"] == {"width": 1280, "height": 720}
    assert _SECRET_PATH not in str(sized)

    dropped = to_remote_generation(
        _desktop_generation(
            spec={
                "params": {
                    "prompt": "fox",
                    "resolution": {"width": 0, "height": -1},
                }
            }
        ),
        signer=_SIGNER,
        device_id=_DEVICE_ID,
    ).model_dump(mode="json")
    assert dropped["spec"]["params"].get("resolution") is None


def test_remote_list_item_omits_paths_and_sets_has_thumbnail() -> None:
    dumped = to_remote_list_item(
        _desktop_asset(), in_use=True, signer=_SIGNER, device_id=_DEVICE_ID
    ).model_dump()
    assert dumped["in_use"] is True
    assert dumped["has_thumbnail"] is True
    assert dumped["bytes_url"].startswith("/api/assets/asset-1/bytes?")
    assert dumped["thumbnail_url"].startswith(
        "/api/assets/asset-1/thumbnail/bytes?"
    )
    assert "path" not in dumped
    assert "thumbnail_path" not in dumped
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RemoteAssetListItem.model_validate({**dumped, "path": _SECRET_PATH})
