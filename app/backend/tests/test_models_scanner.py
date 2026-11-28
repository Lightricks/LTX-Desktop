"""Unit tests for the installed-models / LoRA scanner."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from runtime_config.models_scanner import scan_models_dir


def _touch(path: Path, size: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def _write_safetensors(path: Path, metadata: dict[str, str] | None = None) -> None:
    """Write a header-only (no tensors) safetensors file, optional __metadata__."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def test_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert scan_models_dir(tmp_path / "nope") == []


def test_detects_lora_by_folder(tmp_path: Path) -> None:
    _touch(tmp_path / "loras" / "cinematic.safetensors")
    entries = scan_models_dir(tmp_path)
    assert len(entries) == 1
    assert entries[0].is_lora is True
    assert entries[0].name == "cinematic.safetensors"
    assert entries[0].kind == "safetensors"


def test_detects_lora_by_filename(tmp_path: Path) -> None:
    _touch(tmp_path / "my_lora_v2.safetensors")
    [entry] = scan_models_dir(tmp_path)
    assert entry.is_lora is True


def test_non_lora_checkpoint_not_flagged(tmp_path: Path) -> None:
    _touch(tmp_path / "transformer.safetensors")
    [entry] = scan_models_dir(tmp_path)
    assert entry.is_lora is False


def test_lora_only_filters(tmp_path: Path) -> None:
    _touch(tmp_path / "transformer.safetensors")
    _touch(tmp_path / "loras" / "style.safetensors")
    only = scan_models_dir(tmp_path, lora_only=True)
    assert [e.name for e in only] == ["style.safetensors"]


def test_ignores_non_safetensors(tmp_path: Path) -> None:
    _touch(tmp_path / "loras" / "notes.txt")
    _touch(tmp_path / "loras" / "weights.gguf")
    assert scan_models_dir(tmp_path) == []


def test_detects_ic_lora_by_metadata(tmp_path: Path) -> None:
    _write_safetensors(
        tmp_path / "loras" / "my-ic-lora.safetensors",
        metadata={"reference_downscale_factor": "2"},
    )
    [entry] = scan_models_dir(tmp_path)
    assert entry.is_lora is True
    assert entry.is_ic_lora is True


def test_regular_lora_is_not_ic_lora(tmp_path: Path) -> None:
    # Valid safetensors, LoRA metadata but no reference marker -> regular LoRA.
    _write_safetensors(
        tmp_path / "loras" / "style.safetensors",
        metadata={"modelspec.architecture": "LTX2/lora"},
    )
    [entry] = scan_models_dir(tmp_path)
    assert entry.is_lora is True
    assert entry.is_ic_lora is False


def test_unparsable_file_is_not_ic_lora(tmp_path: Path) -> None:
    # Existing callers write raw bytes; the header read must not raise.
    _touch(tmp_path / "loras" / "garbage.safetensors")
    [entry] = scan_models_dir(tmp_path)
    assert entry.is_lora is True
    assert entry.is_ic_lora is False


def test_catalog_ic_loras_dir_counts_without_metadata(tmp_path: Path) -> None:
    # Community catalog downloads often omit the training metadata marker and the word
    # "lora" in the filename (e.g. 3DREAL-light.safetensors under ic-loras/<id>/).
    _touch(tmp_path / "ic-loras" / "3d-render-to-real" / "3DREAL-light.safetensors", size=8)
    [entry] = scan_models_dir(tmp_path, lora_only=True)
    assert entry.is_lora is True
    assert entry.is_ic_lora is True
    assert entry.name == "3DREAL-light.safetensors"
