from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone

from services.media_probe import extension_for_mime

_MAX_SLUG = 40


def slug_prompt(prompt: object) -> str:
    if not isinstance(prompt, str) or prompt.strip() == "":
        return "generation"
    normalized = unicodedata.normalize("NFKD", prompt)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slugged = re.sub(r"[^a-z0-9]+", "-", ascii_only)
    slugged = re.sub(r"-{2,}", "-", slugged).strip("-")[:_MAX_SLUG].strip("-")
    return slugged or "generation"


def output_asset_name(
    *,
    prompt: object,
    created_at_ms: int,
    asset_id: str,
    mime_type: str,
) -> str:
    stamp = datetime.fromtimestamp(created_at_ms / 1000, tz=timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    ext = extension_for_mime(mime_type).lstrip(".")
    return f"{slug_prompt(prompt)}-{stamp}-{asset_id[:6]}.{ext}"
