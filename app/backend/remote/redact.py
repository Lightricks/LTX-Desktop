"""Defense-in-depth: strip leftover Asset-shaped filesystem paths from remote JSON.

Remote Asset/Generation payloads are projected in `remote.dto`. This response
class only rewrites `path` / `thumbnail_path` on objects that still look like a
Desktop Asset (`media_kind` + `id` + `path`). It is not the remote contract.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi.responses import JSONResponse

_ASSET_PATH_KEYS = frozenset({"path", "thumbnail_path"})


def redact_asset_paths(value: object) -> object:
    """Replace `path` / `thumbnail_path` on Asset-shaped objects; recurse otherwise."""
    if isinstance(value, dict):
        payload = cast(dict[str, Any], value)
        is_asset = "media_kind" in payload and "id" in payload and "path" in payload
        redacted: dict[str, object] = {}
        for key, inner in payload.items():
            if is_asset and key in _ASSET_PATH_KEYS:
                redacted[key] = None if key == "thumbnail_path" else ""
            else:
                redacted[key] = redact_asset_paths(inner)
        return redacted
    if isinstance(value, list):
        return [redact_asset_paths(item) for item in cast(list[object], value)]
    return value


class RedactingJSONResponse(JSONResponse):
    """Default JSON response for the remote app: never leak Desktop filesystem paths."""

    def render(self, content: Any) -> bytes:
        return super().render(redact_asset_paths(content))
