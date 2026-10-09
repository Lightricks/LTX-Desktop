"""Patched SafetensorsStateDictLoader.load must match safetensors.torch.load_file."""

from __future__ import annotations

from pathlib import Path

import torch
from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
from safetensors.torch import load_file, save_file

import services.patches.safetensors_loader_fix as patch


def test_patch_rebinds_loader() -> None:
    assert SafetensorsStateDictLoader.load is patch._patched_load


def test_patched_load_matches_safetensors_load_file(tmp_path: Path) -> None:
    path = tmp_path / "tiny.safetensors"
    expected = {
        "weight": torch.randn(8, 4, dtype=torch.float32),
        "bias": torch.ones(4, dtype=torch.bfloat16),
    }
    save_file(expected, str(path))

    loaded = patch._patched_load(SafetensorsStateDictLoader(), str(path), sd_ops=None)  # type: ignore[arg-type]
    reference = load_file(str(path))

    assert set(loaded.sd) == set(reference) == set(expected)
    for key, tensor in expected.items():
        torch.testing.assert_close(loaded.sd[key], tensor, rtol=0, atol=0)
        torch.testing.assert_close(loaded.sd[key], reference[key], rtol=0, atol=0)
        assert loaded.sd[key].dtype == tensor.dtype
        assert loaded.sd[key].shape == tensor.shape
