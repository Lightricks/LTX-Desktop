from __future__ import annotations

import logging
from pathlib import Path

from api_types import MediaKind
from services.records import AssetRecord, MediaError
from services.store import Store

logger = logging.getLogger(__name__)


def require_source(
    store: Store, asset_id: str, expected_kind: MediaKind
) -> tuple[AssetRecord, Path]:
    record = store.get_asset(asset_id)
    if record is None:
        raise FileNotFoundError(asset_id)
    if record.media_kind != expected_kind:
        raise MediaError(
            "UNSUPPORTED_MEDIA",
            f"expected {expected_kind} asset, got {record.media_kind}",
        )
    try:
        source = Path(record.path).resolve()
    except OSError as exc:
        logger.warning("unreadable media file for asset %s", asset_id)
        raise MediaError("UNREADABLE_MEDIA", "unreadable media file") from exc
    try:
        is_file = source.is_file()
    except OSError:
        is_file = False
    if not is_file:
        logger.warning(
            "unreadable media file for asset %s: %s", asset_id, source
        )
        raise MediaError("UNREADABLE_MEDIA", "unreadable media file")
    return record, source
