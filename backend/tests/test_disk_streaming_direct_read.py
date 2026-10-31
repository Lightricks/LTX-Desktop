"""Streamed block reads with os.preadv on Darwin give the same slot values as get_tensor."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from safetensors.torch import save_file

import services.patches.disk_streaming_direct_read as direct_read

PREFIX = "blocks"
needs_preadv = pytest.mark.skipif(not hasattr(os, "preadv"), reason="os.preadv and /dev/fd are POSIX-only")


@pytest.fixture(autouse=True)
def _restore_readers() -> Iterator[None]:
    init, close, read_into = DiskTensorReader.__init__, DiskTensorReader.close, DiskBlockReader.read_into
    yield
    DiskTensorReader.__init__ = init  # type: ignore[method-assign]
    DiskTensorReader.close = close  # type: ignore[method-assign]
    DiskBlockReader.read_into = read_into  # type: ignore[method-assign]
    direct_read._installed = False
    direct_read._checked = False


@pytest.fixture
def get_tensor_keys(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Keys that went through the original get_tensor path."""
    keys: list[str] = []
    original = DiskTensorReader.get_tensor

    def _record(self: DiskTensorReader, key: str) -> torch.Tensor:
        keys.append(key)
        return original(self, key)

    monkeypatch.setattr(DiskTensorReader, "get_tensor", _record)
    return keys


def _checkpoint(tmp_path: Path) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Two files; mixed dtypes, so every tensor starts at a different offset."""
    torch.manual_seed(0)
    tensors = {
        f"{PREFIX}.0.attn.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        f"{PREFIX}.0.attn.bias": torch.randn(3, dtype=torch.float32),
        f"{PREFIX}.0.ff.weight": torch.randn(5, 7, dtype=torch.bfloat16),
        f"{PREFIX}.0.ff.scale": torch.randn(6, dtype=torch.float32),
    }
    names = list(tensors)
    first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({k: tensors[k] for k in names[:2]}, str(first))
    save_file({k: tensors[k] for k in names[2:]}, str(second))
    return [str(first), str(second)], tensors


def _key_map(tensors: dict[str, torch.Tensor]) -> dict[int, list[tuple[str, str]]]:
    return {0: [(key, key.removeprefix(f"{PREFIX}.0.")) for key in tensors]}


def _slots(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix(f"{PREFIX}.0."): torch.zeros_like(t) for key, t in tensors.items()}


def _install_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    direct_read.install()


def _read_block(paths: list[str], key_map: dict[int, list[tuple[str, str]]], slots: dict[str, torch.Tensor], sd_ops: SDOps | None = None) -> None:
    reader = DiskTensorReader(paths)
    try:
        DiskBlockReader(reader, key_map, sd_ops=sd_ops, blocks_prefix=PREFIX).read_into(slots, 0)
    finally:
        reader.close()


@needs_preadv
def test_direct_read_offset_error_corrupts_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, get_tensor_keys: list[str]
) -> None:
    paths, tensors = _checkpoint(tmp_path)
    expected = _slots(tensors)
    _read_block(paths, _key_map(tensors), expected)

    _install_on_darwin(monkeypatch)
    get_tensor_keys.clear()
    actual = _slots(tensors)
    _read_block(paths, _key_map(tensors), actual)

    assert get_tensor_keys == []
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


@needs_preadv
def test_value_operation_key_gets_raw_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, get_tensor_keys: list[str]
) -> None:
    paths, tensors = _checkpoint(tmp_path)
    doubled = f"{PREFIX}.0.ff.scale"
    sd_ops = SDOps("double_scale").with_matching().with_kv_operation(
        lambda key, value: [KeyValueOperationResult(key, value * 2)], key_suffix=".ff.scale"
    )
    _install_on_darwin(monkeypatch)
    slots = _slots(tensors)
    _read_block(paths, _key_map(tensors), slots, sd_ops=sd_ops)

    assert get_tensor_keys == [doubled]
    assert torch.equal(slots["ff.scale"], tensors[doubled] * 2)
    assert torch.equal(slots["attn.weight"], tensors[f"{PREFIX}.0.attn.weight"])


@needs_preadv
def test_dtype_mismatch_slot_gets_unconverted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, get_tensor_keys: list[str]
) -> None:
    paths, tensors = _checkpoint(tmp_path)
    _install_on_darwin(monkeypatch)
    slots = _slots(tensors)
    slots["attn.bias"] = torch.zeros(3, dtype=torch.bfloat16)  # file holds f32
    _read_block(paths, _key_map(tensors), slots)

    assert get_tensor_keys == [f"{PREFIX}.0.attn.bias"]
    assert torch.equal(slots["attn.bias"], tensors[f"{PREFIX}.0.attn.bias"].to(torch.bfloat16))


@needs_preadv
def test_block_key_map_not_restored_after_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, tensors = _checkpoint(tmp_path)
    _install_on_darwin(monkeypatch)
    key_map = _key_map(tensors)
    full = list(key_map[0])

    slots = _slots(tensors)
    slots["attn.bias"] = torch.zeros(3, dtype=torch.bfloat16)  # fallback that succeeds
    _read_block(paths, key_map, slots)
    assert key_map[0] == full

    slots = _slots(tensors)
    slots["attn.bias"] = torch.zeros(4, dtype=torch.float32)  # fallback copy_ fails on shape
    with pytest.raises(RuntimeError):
        _read_block(paths, key_map, slots)
    assert key_map[0] == full


@needs_preadv
def test_close_leaks_file_descriptors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _checkpoint(tmp_path)
    _install_on_darwin(monkeypatch)
    before = len(os.listdir("/dev/fd"))
    reader = DiskTensorReader(paths)
    assert len(os.listdir("/dev/fd")) > before
    reader.close()
    assert len(os.listdir("/dev/fd")) == before


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_install_patches_reader_off_darwin(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    before = DiskBlockReader.read_into
    direct_read.install()
    assert DiskBlockReader.read_into is before
    assert direct_read._installed is False
