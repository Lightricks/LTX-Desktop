"""Shared tail of the asset transforms: size-check the encode, ingest it."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from services.records import AssetRecord, MediaError
from services.store import Store

_UNSAFE_NAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
_MAX_STEM_CHARS = 100


def derived_file_name(source_name: str, suffix: str, extension: str) -> str:
    """``talk.mov`` -> ``talk-trim.mp4``; ingest names the asset after the file."""
    stem = _UNSAFE_NAME_CHARS.sub("_", Path(source_name).stem).strip(" .")
    return f"{stem[:_MAX_STEM_CHARS] or 'clip'}-{suffix}{extension}"


def check_output_size(path: Path, limit_bytes: int, label: str) -> None:
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise MediaError(
            "UNREADABLE_MEDIA", f"encoded {label} dest is unreadable"
        ) from exc
    # -fs truncates at the limit and still exits 0; a truncated clip must
    # never be ingested as success.
    if size_bytes >= limit_bytes:
        raise MediaError("FILE_TOO_LARGE", f"encoded {label} exceeds size limit")


def ingest_encoded(
    store: Store,
    write: Callable[[Path], Path],
    *,
    prefix: str,
    label: str,
    failure: str,
) -> AssetRecord:
    """``write`` encodes into the temp dir and returns the file to ingest."""
    try:
        # A failed cleanup must not turn an ingested asset into a 500.
        with tempfile.TemporaryDirectory(
            prefix=prefix, ignore_cleanup_errors=True
        ) as tmp:
            dest = write(Path(tmp))
            try:
                return store.ingest_upload(str(dest))
            except FileNotFoundError as exc:
                raise MediaError(
                    "UNREADABLE_MEDIA", f"encoded {label} dest is unreadable"
                ) from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError("UNREADABLE_MEDIA", failure) from exc
