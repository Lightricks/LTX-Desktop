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
    meta = _read_safetensors_metadata(path)
    if meta is None or "config" not in meta:
        return {}
    return json.loads(meta["config"])


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


# --- Patch 3: ltx_pipelines.utils.constants.detect_params ---

import ltx_pipelines.utils.constants as _constants_module


_original_detect_params = _constants_module.detect_params


def _patched_detect_params(checkpoint_path: str) -> object:
    import logging
    logger = logging.getLogger(__name__)

    try:
        meta = _read_safetensors_metadata(checkpoint_path) or {}
        version = meta.get("model_version", "")
    except Exception:
        logger.warning("Could not read checkpoint metadata from %s, using defaults", checkpoint_path)
        return _constants_module.LTX_2_PARAMS

    if version.startswith(_constants_module._LTX_2_3_MODEL_VERSION_PREFIX):
        return _constants_module.LTX_2_3_PARAMS

    logger.info("Using LTX_2_PARAMS for checkpoint (version=%s)", version or "unknown")
    return _constants_module.LTX_2_PARAMS


assert hasattr(_constants_module, "detect_params"), (
    "ltx_pipelines.utils.constants.detect_params not found — patch needs updating."
)
_constants_module.detect_params = _patched_detect_params


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
