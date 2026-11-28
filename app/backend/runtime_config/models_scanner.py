"""Scan a models directory for installed LoRA adapters.

Uses a fast filename/directory heuristic for LoRA detection (no header parsing):
a ``.safetensors`` file is treated as a LoRA when it lives under a ``loras``/``lora``
directory, or its filename contains ``lora``.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

# Filename / directory patterns that suggest a LoRA adapter.
_LORA_NAME_RE = re.compile(r"lora", re.IGNORECASE)
_LORA_DIR_NAME_RE = re.compile(r"loras?", re.IGNORECASE)

# Set by the v2v training strategy on every IC-LoRA; never on regular LoRAs.
# It's the same key the inference pipeline reads for the reference-video scale factor.
_IC_LORA_METADATA_KEY = "reference_downscale_factor"

# A safetensors header is JSON metadata only — kilobytes in practice. Cap the
# declared size so a corrupt/crafted file can't make us allocate gigabytes.
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class InstalledModel:
    path: Path
    name: str
    kind: str  # "safetensors"
    size_bytes: int
    is_lora: bool
    is_ic_lora: bool


def _is_lora_heuristic(relative_path: Path) -> bool:
    """True if the filename or any parent directory suggests a LoRA adapter."""
    if _LORA_NAME_RE.search(relative_path.stem):
        return True
    if any(_LORA_DIR_NAME_RE.fullmatch(part) for part in relative_path.parts):
        return True
    # Catalog IC-LoRA downloads land under ic-loras/<id>/; those filenames often omit "lora"
    # (e.g. 3DREAL-light.safetensors) so the directory is the signal.
    return _under_ic_loras_dir(relative_path)


def _under_ic_loras_dir(relative_path: Path) -> bool:
    return any(part.lower() == "ic-loras" for part in relative_path.parts)


def is_ic_lora_file(path: Path) -> bool:
    """True if the safetensors header carries the IC-LoRA reference marker.

    Best-effort: reads only the header (8-byte length + JSON), no mmap or weight load.
    Returns False on any non-safetensors / truncated / unreadable file.
    """
    try:
        with open(path, "rb") as f:
            (header_size,) = struct.unpack("<Q", f.read(8))
            if header_size > _MAX_SAFETENSORS_HEADER_BYTES:
                return False
            header = json.loads(f.read(header_size).decode("utf-8"))
        metadata = header.get("__metadata__")
        return isinstance(metadata, dict) and _IC_LORA_METADATA_KEY in metadata
    except Exception:
        return False


def resolve_lora_ref(models_dir: Path, ref: str) -> Path:
    """Resolve a LoRA *ref* (from the request) to an existing file inside *models_dir*.

    Refs come straight off the UI's model picker; containing them to models_dir keeps a
    stale/crafted path from pointing the loader at an arbitrary readable file. Raises
    ValueError on escape or miss so callers can map it to a clean 400.
    """
    base = models_dir.resolve()
    path = Path(ref).resolve() if Path(ref).is_absolute() else (base / ref).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"LoRA not found in models dir: {ref}")
    return path


def scan_models_dir(models_dir: Path, *, lora_only: bool = False) -> list[InstalledModel]:
    """Return ``.safetensors`` model files under *models_dir* (scanned recursively).

    When *lora_only* is True, return only entries detected as LoRA adapters.
    """
    if not models_dir.exists():
        return []

    results: list[InstalledModel] = []
    for entry in sorted(models_dir.rglob("*.safetensors")):
        if not entry.is_file():
            continue
        rel = entry.relative_to(models_dir)
        # ".tmp" is the download staging dir (see lora_catalog_handler._download); an
        # in-flight or crashed-before-cleanup download must never look installed.
        if ".tmp" in rel.parts:
            continue
        is_lora = _is_lora_heuristic(rel)
        if lora_only and not is_lora:
            continue
        # Catalog downloads under ic-loras/ are IC-LoRAs even when the file lacks the
        # training-strategy metadata marker (common for community checkpoints).
        is_ic_lora = is_lora and (_under_ic_loras_dir(rel) or is_ic_lora_file(entry))
        results.append(
            InstalledModel(
                path=entry,
                name=entry.name,
                kind="safetensors",
                size_bytes=entry.stat().st_size,
                is_lora=is_lora,
                is_ic_lora=is_ic_lora,
            )
        )
    return results
