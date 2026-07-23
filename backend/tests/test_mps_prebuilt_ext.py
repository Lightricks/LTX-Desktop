"""Tests for the mps_prebuilt_ext path resolution and no-op guards.

Deliberately does not exercise the "prebuilt .so found" patch-application branch:
that mutates torch.utils.cpp_extension.load process-wide, and a fake .so file can't
stand in for a real compiled extension. Platform is monkeypatched (not the real
sys.platform) so these run deterministically on any CI OS.
"""

from __future__ import annotations

from pathlib import Path

import mps_prebuilt_ext


def test_gib_converts_bytes_to_gibibytes() -> None:
    assert mps_prebuilt_ext.gib(1024 ** 3) == 1.0


def test_prebuilt_so_none_when_env_var_unset(monkeypatch) -> None:
    monkeypatch.delenv("LTX_MPS_EXT_PREBUILT_DIR", raising=False)
    assert mps_prebuilt_ext._prebuilt_so() is None


def test_prebuilt_so_none_when_file_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LTX_MPS_EXT_PREBUILT_DIR", str(tmp_path))
    assert mps_prebuilt_ext._prebuilt_so() is None


def test_prebuilt_so_found_when_file_present(tmp_path: Path, monkeypatch) -> None:
    so_path = tmp_path / "mps_sdpa_zc_ext.so"
    so_path.write_bytes(b"not a real extension, just a placeholder")
    monkeypatch.setenv("LTX_MPS_EXT_PREBUILT_DIR", str(tmp_path))
    assert mps_prebuilt_ext._prebuilt_so() == so_path


def test_setup_prebuilt_mps_extension_noop_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "linux")
    monkeypatch.setenv("LTX_MPS_EXT_PREBUILT_DIR", "/nonexistent")
    mps_prebuilt_ext.setup_prebuilt_mps_extension()  # must not raise


def test_setup_prebuilt_mps_extension_noop_on_darwin_without_prebuilt(monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "darwin")
    monkeypatch.delenv("LTX_MPS_EXT_PREBUILT_DIR", raising=False)
    mps_prebuilt_ext.setup_prebuilt_mps_extension()  # must not raise; no .so to load


def test_mps_memory_sample_none_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "linux")
    assert mps_prebuilt_ext.mps_memory_sample() is None


def test_reset_mps_sdpa_stats_noop_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "linux")
    mps_prebuilt_ext.reset_mps_sdpa_stats()  # must not raise
