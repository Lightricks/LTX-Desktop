"""Monkey-patch: ``GemmaAssets.from_single_file`` reads the packed text encoder without mmap.

Upstream opens the single-file Gemma text encoder (~26 GB for LTX 2.5) with
``safetensors.safe_open(..., framework="pt")`` and reads the tokenizer (~32 MB) and a few small
sidecars with ``get_tensor``. With the torch framework, ``get_tensor`` maps the whole file through
``torch.UntypedStorage.from_file(shared=False)``. On Windows that mapping can end in a native
access violation (exit code 3221225477 / 0xC0000005) inside ``torch/storage.py __getitem__``, which
kills the backend process and marks the running job INTERRUPTED
(https://github.com/Lightricks/LTX-Desktop/issues/203).

This patch parses the safetensors header with plain positional file reads and reads only the bytes
of the tensors it needs. Output is byte-identical to upstream for ``U8`` and ``I8`` tensors, which
is what the packer writes. It is installed on Windows only. Other platforms keep the upstream
read, which has no reported failure.

The file is user-supplied, so every read is bounded and checked before it is trusted:
- the header length is capped (the same 100 MB limit safetensors enforces),
- every ``data_offsets`` pair must lie inside the file and match the tensor shape,
- the total asset bytes read are capped.
Anything unexpected (other dtype, odd header, cap exceeded, short read) falls back to the original
implementation. Missing ``gemma_config`` metadata or a missing tokenizer tensor raise the same
``ValueError`` as upstream.

Not replaced: ``_gemma4_unified_is_comfy_flat`` in ``ltx_core`` opens the same file with
``safe_open`` only to list tensor names. It never calls ``get_tensor``, so it does not create the
torch storage that crashed in the report. Revisit it if a crash points there.

Remove once ``GemmaAssets.from_single_file`` reads without mapping the whole file, and a Windows
machine with the 26 GB text encoder generates without an access violation without this patch.

Usage:
    import inference.patches.gemma_assets_positional_read  # noqa: F401
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO, cast

from ltx_core.text_encoders.gemma import gemma_assets as _gemma_assets
from ltx_core.text_encoders.gemma.gemma_assets import GemmaAssets

logger = logging.getLogger(__name__)

# safetensors rejects headers above 100 MB. A real packed text encoder header is far smaller.
_MAX_HEADER_BYTES = 100_000_000
# tokenizer_json is ~32 MB for Gemma 4. All sidecars together stay far below this cap.
_MAX_ASSET_BYTES = 256 * 1024 * 1024
# Only one-byte integer dtypes round-trip to the same bytes as upstream ``_tensor_to_bytes``.
_BYTE_DTYPES = frozenset({"U8", "I8"})

_FROM_SINGLE_FILE = "from_single_file"
_original_from_single_file = cast(Any, GemmaAssets.__dict__.get(_FROM_SINGLE_FILE))
_logged_active = False


class _Unsupported(Exception):
    """The file is not in the narrow layout this patch handles. Use the upstream read."""


def _read_exact(fh: BinaryIO, size: int) -> bytes:
    data = fh.read(size)
    if len(data) != size:
        raise _Unsupported(f"short read: wanted {size} bytes, got {len(data)}")
    return data


def _read_header(fh: BinaryIO, file_size: int) -> tuple[dict[str, Any], int]:
    """Return the JSON header and the offset where tensor data starts."""
    (header_size,) = struct.unpack("<Q", _read_exact(fh, 8))
    if header_size > _MAX_HEADER_BYTES or 8 + header_size > file_size:
        raise _Unsupported(f"header size {header_size} is out of range for a {file_size}-byte file")
    try:
        header = json.loads(_read_exact(fh, header_size).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _Unsupported(f"header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise _Unsupported("header is not a JSON object")
    return cast(dict[str, Any], header), 8 + header_size


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _TensorBytes:
    """Bounded positional reader for the one-byte tensors of a safetensors file."""

    def __init__(self, fh: BinaryIO, header: dict[str, Any], data_start: int, file_size: int) -> None:
        self._fh = fh
        self._header = header
        self._data_start = data_start
        self._data_size = file_size - data_start
        self._budget = _MAX_ASSET_BYTES

    def read(self, key: str) -> bytes:
        info = self._header[key]
        if not isinstance(info, dict):
            raise _Unsupported(f"tensor {key!r} entry is not an object")
        entry = cast(dict[str, Any], info)
        if entry.get("dtype") not in _BYTE_DTYPES:
            raise _Unsupported(f"tensor {key!r} has dtype {entry.get('dtype')!r}")
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if not isinstance(shape, list) or not all(_is_int(d) and d >= 0 for d in cast(list[Any], shape)):
            raise _Unsupported(f"tensor {key!r} has an invalid shape")
        if not isinstance(offsets, list) or len(cast(list[Any], offsets)) != 2:
            raise _Unsupported(f"tensor {key!r} has invalid data_offsets")
        begin, end = cast(list[Any], offsets)
        if not (_is_int(begin) and _is_int(end)) or not 0 <= begin <= end <= self._data_size:
            raise _Unsupported(f"tensor {key!r} offsets {offsets} are outside the data section")
        if end - begin != math.prod(cast(list[int], shape)):
            raise _Unsupported(f"tensor {key!r} offsets do not match its shape")
        if end - begin > self._budget:
            raise _Unsupported(f"tensor {key!r} exceeds the asset size cap")
        self._budget -= end - begin
        self._fh.seek(self._data_start + begin)
        return _read_exact(self._fh, end - begin)


def _read_assets(path: Path) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    """Return ``(config_dict, tokenizer_json, sidecars)``, matching upstream's output."""
    with open(path, "rb") as fh:
        file_size = os.fstat(fh.fileno()).st_size
        header, data_start = _read_header(fh, file_size)
        raw_meta = header.pop("__metadata__", None) or {}
        if not isinstance(raw_meta, dict) or not all(isinstance(v, str) for v in cast(dict[Any, Any], raw_meta).values()):
            raise _Unsupported("__metadata__ is not a string map")
        meta = cast(dict[str, str], raw_meta)

        raw_config = meta.get(_gemma_assets.GEMMA_CONFIG_METADATA_KEY)
        if raw_config is None:
            raise ValueError(
                f"Safetensors text-encoder {path} is missing metadata key "
                f"{_gemma_assets.GEMMA_CONFIG_METADATA_KEY!r} (JSON-encoded HuggingFace config)."
            )
        config_dict = json.loads(raw_config)

        if _gemma_assets.TOKENIZER_JSON_TENSOR_KEY not in header:
            raise ValueError(
                f"Safetensors text-encoder {path} is missing tensor {_gemma_assets.TOKENIZER_JSON_TENSOR_KEY!r}."
            )
        reader = _TensorBytes(fh, header, data_start, file_size)
        tokenizer_json = reader.read(_gemma_assets.TOKENIZER_JSON_TENSOR_KEY)

        sidecars: dict[str, bytes] = {}
        for key in header:
            if key.startswith(_gemma_assets.HF_ASSET_TENSOR_PREFIX):
                sidecars[key.removeprefix(_gemma_assets.HF_ASSET_TENSOR_PREFIX)] = reader.read(key)

    for name in _gemma_assets._METADATA_FALLBACK_FILENAMES:  # pyright: ignore[reportPrivateUsage]
        if name in sidecars or name not in meta:
            continue
        sidecars[name] = meta[name].encode()
    return cast(dict[str, Any], config_dict), tokenizer_json, sidecars


def _patched_from_single_file(cls: type[GemmaAssets], path: Path) -> GemmaAssets:
    global _logged_active
    path = Path(path)
    try:
        config_dict, tokenizer_json, sidecars = _read_assets(path)
    except _Unsupported as exc:
        logger.warning(
            "Gemma assets in %s use a layout the positional read does not handle (%s). "
            "Falling back to the safetensors mmap read.",
            path,
            exc,
        )
        return _original_from_single_file.__func__(cls, path)
    if not _logged_active:
        logger.info("Reading single-file Gemma assets with positional reads instead of mmap (LTX-Desktop#203).")
        _logged_active = True

    assets = cls(
        source=str(path),
        config_dict=config_dict,
        tokenizer_json=tokenizer_json,
        sidecars=sidecars,
        weight_paths=(str(path.resolve()),),
    )
    assets._require_sidecars(*_gemma_assets._REQUIRED_ASSET_FILENAMES)  # pyright: ignore[reportPrivateUsage]
    return assets


assert isinstance(_original_from_single_file, classmethod), (
    "GemmaAssets.from_single_file is not a classmethod — patch needs updating."
)
for _name in (
    "GEMMA_CONFIG_METADATA_KEY",
    "TOKENIZER_JSON_TENSOR_KEY",
    "HF_ASSET_TENSOR_PREFIX",
    "_METADATA_FALLBACK_FILENAMES",
    "_REQUIRED_ASSET_FILENAMES",
):
    assert hasattr(_gemma_assets, _name), f"ltx_core gemma_assets.{_name} not found — patch needs updating."

if sys.platform == "win32":
    # GemmaAssets is a frozen dataclass: that only blocks instance writes, not class attributes.
    setattr(GemmaAssets, _FROM_SINGLE_FILE, classmethod(_patched_from_single_file))
