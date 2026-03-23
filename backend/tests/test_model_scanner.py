"""Tests for ModelScannerImpl."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from services.model_scanner.model_scanner_impl import ModelScannerImpl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_gguf(path: Path, version: int = 3) -> None:
    """Write a minimal valid GGUF file (magic + version only)."""
    path.write_bytes(b"GGUF" + struct.pack("<I", version))


def _write_minimal_safetensors(path: Path) -> None:
    """Write a minimal valid safetensors file (8-byte header length + empty JSON header)."""
    header = b"{}"
    header_len = struct.pack("<Q", len(header))
    path.write_bytes(header_len + header)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scanner() -> ModelScannerImpl:
    return ModelScannerImpl()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_folder_returns_empty_list(scanner: ModelScannerImpl, tmp_path: Path) -> None:
    result = scanner.scan_video_models(tmp_path)
    assert result == []


def test_nonexistent_folder_returns_empty_list(scanner: ModelScannerImpl, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    result = scanner.scan_video_models(missing)
    assert result == []


def test_detects_gguf_file(scanner: ModelScannerImpl, tmp_path: Path) -> None:
    gguf_path = tmp_path / "ltx-video-Q5_K_M.gguf"
    _write_minimal_gguf(gguf_path)

    result = scanner.scan_video_models(tmp_path)

    assert len(result) == 1
    model = result[0]
    assert model.model_format == "gguf"
    assert model.quant_type == "Q5_K_M"
    assert model.filename == "ltx-video-Q5_K_M.gguf"
    assert model.path == str(gguf_path)
    assert model.size_bytes == gguf_path.stat().st_size
    assert model.is_distilled is False


def test_detects_safetensors_file(scanner: ModelScannerImpl, tmp_path: Path) -> None:
    st_path = tmp_path / "ltx-video.safetensors"
    _write_minimal_safetensors(st_path)

    result = scanner.scan_video_models(tmp_path)

    assert len(result) == 1
    model = result[0]
    assert model.model_format == "bf16"
    assert model.filename == "ltx-video.safetensors"
    assert model.path == str(st_path)
    assert model.size_bytes == st_path.stat().st_size
    assert model.is_distilled is False


def test_detects_nf4_folder(scanner: ModelScannerImpl, tmp_path: Path) -> None:
    nf4_dir = tmp_path / "ltx-video-nf4"
    nf4_dir.mkdir()
    (nf4_dir / "quantize_config.json").write_text(
        json.dumps({"quant_type": "nf4", "bits": 4}), encoding="utf-8"
    )
    (nf4_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    result = scanner.scan_video_models(tmp_path)

    assert len(result) == 1
    model = result[0]
    assert model.model_format == "nf4"
    assert model.quant_type == "nf4"
    assert model.filename == "ltx-video-nf4"
    assert model.path == str(nf4_dir)
    assert model.size_bytes > 0
    assert model.is_distilled is False


def test_skips_corrupt_gguf_file(scanner: ModelScannerImpl, tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.gguf"
    corrupt.write_bytes(b"NOTGGUF")

    result = scanner.scan_video_models(tmp_path)
    assert result == []


def test_skips_non_model_files(scanner: ModelScannerImpl, tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    result = scanner.scan_video_models(tmp_path)
    assert result == []


def test_multiple_models_in_folder(scanner: ModelScannerImpl, tmp_path: Path) -> None:
    # GGUF
    gguf_path = tmp_path / "ltx-Q8_0.gguf"
    _write_minimal_gguf(gguf_path)

    # Safetensors
    st_path = tmp_path / "ltx-bf16.safetensors"
    _write_minimal_safetensors(st_path)

    # NF4 folder
    nf4_dir = tmp_path / "ltx-nf4"
    nf4_dir.mkdir()
    (nf4_dir / "quantize_config.json").write_text(
        json.dumps({"quant_type": "nf4"}), encoding="utf-8"
    )

    result = scanner.scan_video_models(tmp_path)

    assert len(result) == 3
    formats = {m.model_format for m in result}
    assert formats == {"gguf", "bf16", "nf4"}
