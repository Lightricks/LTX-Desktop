"""Monkey-patch: replace safe_open metadata reads with direct file reads.

safetensors' safe_open uses torch.UntypedStorage.from_file(shared=False) which
reserves copy-on-write commit charge equal to the file size. For a 22GB
checkpoint, this reserves 22GB of commit charge just to read a small JSON
header. Under memory pressure, this causes "paging file too small" errors.

This patch replaces all metadata-only safe_open calls with direct file reads
that parse the safetensors header without mmap or commit charge reservation.

Remove this patch once safetensors supports read-only file mapping.

Usage:
    import services.patches.safetensors_metadata_fix  # noqa: F401
"""

from __future__ import annotations

import json
import struct


def _read_safetensors_metadata(path: str) -> dict[str, str] | None:
    """Read metadata from a safetensors file header without mmap."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
    return header.get("__metadata__")


# --- Patch 1: SafetensorsModelStateDictLoader.metadata ---

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader


def _patched_model_metadata(self: SafetensorsModelStateDictLoader, path: str) -> dict:
    """Full ``__metadata__`` dict with JSON-encoded values parsed, mirroring upstream.

    Callers index into it themselves (``config``, ``model_version``,
    ``gemma_source_checkpoint``), so returning only ``config`` silently hides the
    sibling keys.
    """
    meta = _read_safetensors_metadata(path)
    if meta is None:
        return {}
    parsed: dict[str, object] = {}
    for key, value in meta.items():
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


assert hasattr(SafetensorsModelStateDictLoader, "metadata") and callable(
    getattr(SafetensorsModelStateDictLoader, "metadata")
), "SafetensorsModelStateDictLoader.metadata not found — patch needs updating."
SafetensorsModelStateDictLoader.metadata = _patched_model_metadata  # type: ignore[assignment]


# --- Patch 2: ltx_pipelines read_lora_reference_downscale_factor ---
# The upstream MPS-support work moved this helper to ltx_pipelines.iclora_utils and renamed it
# (dropped the leading underscore); ic_lora re-imports it, so both bindings are patched.
# The upstream version still uses safetensors safe_open (Windows commit-charge concern), so
# this mmap-free replacement is still worth applying.

import ltx_pipelines.ic_lora as _ic_lora_module
import ltx_pipelines.iclora_utils as _iclora_utils_module

_DOWNSCALE_FN = "read_lora_reference_downscale_factor"


def _patched_read_lora_reference_downscale_factor(lora_path: str) -> int:
    try:
        meta = _read_safetensors_metadata(lora_path) or {}
        return int(meta.get("reference_downscale_factor", 1))
    except Exception:
        import logging
        logging.warning(f"Failed to read metadata from LoRA file '{lora_path}'")
        return 1


assert hasattr(_iclora_utils_module, _DOWNSCALE_FN), (
    f"ltx_pipelines.iclora_utils.{_DOWNSCALE_FN} not found — patch needs updating."
)
setattr(_iclora_utils_module, _DOWNSCALE_FN, _patched_read_lora_reference_downscale_factor)
# ic_lora binds the name via `from ...iclora_utils import ...`, so its module-local reference
# (the actual call site) must be patched too.
if hasattr(_ic_lora_module, _DOWNSCALE_FN):
    setattr(_ic_lora_module, _DOWNSCALE_FN, _patched_read_lora_reference_downscale_factor)


# --- Patch 3: ltx_pipelines.utils.constants.detect_model_version ---
# Only the metadata read is replaced; the version -> params mapping stays upstream's
# (``detect_params`` calls this by module global), so new generations keep their own defaults.

import ltx_pipelines.distilled as _distilled_module
import ltx_pipelines.utils.constants as _constants_module

from ltx_core.loader.helpers import parse_model_version

_DETECT_VERSION_FN = "detect_model_version"


def _patched_detect_model_version(checkpoint_path: str) -> tuple[int, ...]:
    import logging
    logger = logging.getLogger(__name__)

    try:
        meta = _read_safetensors_metadata(checkpoint_path) or {}
        version = meta.get("model_version", "")
    except Exception:
        logger.warning("Could not read checkpoint metadata from %s, treating it as unversioned", checkpoint_path)
        return ()

    # Pre-release tags come both dot- and hyphen-separated ("2.3.rc1", "2.4-rc2").
    parsed = parse_model_version(version.replace("-", "."))
    logger.info("Checkpoint declares model_version=%s (parsed as %s)", version or "unknown", parsed)
    return parsed


assert hasattr(_constants_module, _DETECT_VERSION_FN), (
    f"ltx_pipelines.utils.constants.{_DETECT_VERSION_FN} not found — patch needs updating."
)
setattr(_constants_module, _DETECT_VERSION_FN, _patched_detect_model_version)
# distilled.py binds the name via `from ...constants import ...`, so its module-local
# reference (the sampler-selection call site) must be patched too.
if hasattr(_distilled_module, _DETECT_VERSION_FN):
    setattr(_distilled_module, _DETECT_VERSION_FN, _patched_detect_model_version)


# --- Patch 4: services.text_encoder.ltx_text_encoder.TextHandler.get_model_id_from_checkpoint ---

from services.text_encoder.ltx_text_encoder import LTXTextEncoder


def _patched_get_model_id_from_checkpoint(self: LTXTextEncoder, checkpoint_path: str) -> str | None:
    try:
        meta = _read_safetensors_metadata(checkpoint_path) or {}
        if "encrypted_wandb_properties" in meta:
            return meta["encrypted_wandb_properties"]
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Could not extract model_id from checkpoint: %s", exc, exc_info=True)
    return None


assert hasattr(LTXTextEncoder, "get_model_id_from_checkpoint"), (
    "LTXTextEncoder.get_model_id_from_checkpoint not found — patch needs updating."
)
LTXTextEncoder.get_model_id_from_checkpoint = _patched_get_model_id_from_checkpoint  # type: ignore[assignment]
