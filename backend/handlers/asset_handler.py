from __future__ import annotations

from collections.abc import Callable

from api_types import (
    Asset,
    AssetListItem,
    AssetListQuery,
    AssetListResponse,
    IngestAssetRequest,
    TrimMediaRequest,
)
from _routes._errors import HTTPError
from services.audio_trim import extract_audio_asset, trim_audio_asset
from services.video_trim import trim_video_asset
from services.store import Store
from services.records import (
    AssetInUseError,
    AssetRecord,
    ListedAsset,
    MediaError,
    StatusError,
    UnavailableError,
)


def _asset_to_api(record: AssetRecord) -> Asset:
    return Asset.model_validate(record.model_dump(mode="json"))


def _listed_asset_to_api(item: ListedAsset) -> AssetListItem:
    return AssetListItem.model_validate(
        {**_asset_to_api(item.asset).model_dump(), "in_use": item.in_use}
    )


_MEDIA_ERROR_STATUS = {
    "UNSUPPORTED_MEDIA": 422,
    "INVALID_TRIM_RANGE": 422,
    "TRIM_BUSY": 503,
}


class AssetHandler:
    def __init__(self, store: Store) -> None:
        self._db = store

    def ingest_asset(self, req: IngestAssetRequest) -> Asset:
        try:
            record = self._db.ingest_upload(req.path)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except FileNotFoundError as exc:
            raise HTTPError(
                400, f"file not found: {req.path}", code="FILE_NOT_FOUND"
            ) from exc
        except MediaError as exc:
            raise HTTPError(400, str(exc), code=exc.code) from exc
        return _asset_to_api(record)

    def get_asset(
        self, asset_id: str, *, tombstone_missing: bool = True
    ) -> Asset:
        try:
            record = self._db.get_asset(
                asset_id, tombstone_missing=tombstone_missing
            )
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        if record is None:
            raise HTTPError(404, "ASSET_NOT_FOUND")
        return _asset_to_api(record)

    def list_assets(
        self, query: AssetListQuery, *, tombstone_missing: bool = True
    ) -> AssetListResponse:
        try:
            page = self._db.list_assets(
                media_kind=query.media_kind,
                sort=query.sort,
                q=query.q,
                cursor=query.cursor,
                limit=query.limit,
                tombstone_missing=tombstone_missing,
            )
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except ValueError as exc:
            raise HTTPError(400, str(exc)) from exc
        return AssetListResponse(
            items=[_listed_asset_to_api(item) for item in page.items],
            next_cursor=page.next_cursor,
        )

    def delete_asset(self, asset_id: str) -> None:
        try:
            deleted = self._db.delete_asset(asset_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except AssetInUseError as exc:
            raise HTTPError(409, "ASSET_IN_USE") from exc
        except StatusError as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        if not deleted:
            raise HTTPError(404, "ASSET_NOT_FOUND")

    def trim_audio(self, asset_id: str, req: TrimMediaRequest) -> Asset:
        if req.endSec <= req.startSec:
            raise HTTPError(422, "INVALID_TRIM_RANGE")
        return self._media_transform(
            lambda: trim_audio_asset(self._db, asset_id, req.startSec, req.endSec)
        )

    def trim_video(self, asset_id: str, req: TrimMediaRequest) -> Asset:
        if req.endSec <= req.startSec:
            raise HTTPError(422, "INVALID_TRIM_RANGE")
        return self._media_transform(
            lambda: trim_video_asset(self._db, asset_id, req.startSec, req.endSec)
        )

    def extract_audio(self, asset_id: str) -> Asset:
        return self._media_transform(lambda: extract_audio_asset(self._db, asset_id))

    def _media_transform(self, produce: Callable[[], AssetRecord]) -> Asset:
        try:
            record = produce()
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except FileNotFoundError as exc:
            raise HTTPError(404, "ASSET_NOT_FOUND") from exc
        except MediaError as exc:
            status = _MEDIA_ERROR_STATUS.get(exc.code, 400)
            raise HTTPError(status, str(exc), code=exc.code) from exc
        return _asset_to_api(record)
