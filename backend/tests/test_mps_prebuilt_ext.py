"""Tests for the mps_prebuilt_ext path resolution and no-op guards.

Deliberately does not exercise the "prebuilt .so found" patch-application branch:
that mutates torch.utils.cpp_extension.load process-wide, and a fake .so file can't
stand in for a real compiled extension. Platform is monkeypatched (not the real
sys.platform) so these run deterministically on any CI OS.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import types

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


def test_setup_prebuilt_mps_extension_noop_on_darwin_without_prebuilt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "darwin")
    monkeypatch.delenv("LTX_MPS_EXT_PREBUILT_DIR", raising=False)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    mps_prebuilt_ext.setup_prebuilt_mps_extension()  # must not raise; no .so to load


def test_clear_stale_mps_extension_locks_removes_jit_lock_files(tmp_path: Path) -> None:
    ext_dir = tmp_path / "py313_cpu" / "mps_sdpa_zc_ext"
    ext_dir.mkdir(parents=True)
    lock = ext_dir / "lock"
    ninja_lock = ext_dir / ".ninja_lock"
    keep = ext_dir / "mps_sdpa_zc_ext.so"
    lock.write_bytes(b"")
    ninja_lock.write_bytes(b"")
    keep.write_bytes(b"so")

    removed = mps_prebuilt_ext.clear_stale_mps_extension_locks(tmp_path)

    assert sorted(path.name for path in removed) == [".ninja_lock", "lock"]
    assert not lock.exists()
    assert not ninja_lock.exists()
    assert keep.exists()


def test_clear_stale_mps_extension_locks_removes_override_layout(tmp_path: Path) -> None:
    ext_dir = tmp_path / "mps_sdpa_zc_ext"
    ext_dir.mkdir()
    lock = ext_dir / "lock"
    ninja_lock = ext_dir / ".ninja_lock"
    lock.write_bytes(b"")
    ninja_lock.write_bytes(b"")

    removed = mps_prebuilt_ext.clear_stale_mps_extension_locks(tmp_path)

    assert sorted(path.name for path in removed) == [".ninja_lock", "lock"]
    assert not lock.exists()
    assert not ninja_lock.exists()


def test_clear_stale_mps_extension_locks_skips_held_ninja_lock(tmp_path: Path) -> None:
    if sys.platform == "win32":
        return
    import fcntl

    ext_dir = tmp_path / "py313_cpu" / "mps_sdpa_zc_ext"
    ext_dir.mkdir(parents=True)
    ninja_lock = ext_dir / ".ninja_lock"
    ninja_lock.write_bytes(b"")
    fd = os.open(ninja_lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        removed = mps_prebuilt_ext.clear_stale_mps_extension_locks(tmp_path)
        assert removed == []
        assert ninja_lock.exists()
    finally:
        os.close(fd)


def test_clear_stale_mps_extension_locks_noop_when_cache_missing(tmp_path: Path) -> None:
    assert mps_prebuilt_ext.clear_stale_mps_extension_locks(tmp_path / "missing") == []


def test_mps_memory_sample_none_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "linux")
    assert mps_prebuilt_ext.mps_memory_sample() is None


def test_reset_mps_sdpa_stats_noop_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "linux")
    mps_prebuilt_ext.reset_mps_sdpa_stats()  # must not raise


def test_coerce_replaces_none_fused_min_with_defaults() -> None:
    raw: dict = {"fused_min_bytes": {"bf16": None, "fp16": 1, "fp32": None}, "calibrated": True}
    out = mps_prebuilt_ext.coerce_mps_sdpa_thresholds(raw)
    assert out["fused_min_bytes"]["bf16"] == 4 * 1024**2
    assert out["fused_min_bytes"]["fp16"] == 1
    assert out["fused_min_bytes"]["fp32"] == 8 * 1024**2
    assert raw["fused_min_bytes"]["bf16"] is None


def test_coerce_is_noop_when_all_fused_min_are_set() -> None:
    raw: dict = {"fused_min_bytes": {"bf16": 8, "fp16": 8, "fp32": 16}, "calibrated": True}
    assert mps_prebuilt_ext.coerce_mps_sdpa_thresholds(raw) is raw


def test_coerce_passthrough_when_fused_min_missing() -> None:
    raw: dict = {"calibrated": False}
    assert mps_prebuilt_ext.coerce_mps_sdpa_thresholds(raw) is raw


def test_install_threshold_guard_noop_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "linux")
    mps_prebuilt_ext.install_mps_sdpa_threshold_guard()  # must not raise


def test_threshold_guard_coerces_none_on_get_thresholds(monkeypatch) -> None:
    monkeypatch.setattr(mps_prebuilt_ext.sys, "platform", "darwin")
    unsafe: dict = {"fused_min_bytes": {"bf16": None, "fp16": None, "fp32": None}, "calibrated": True}

    fake_cal = types.SimpleNamespace(get_thresholds=lambda: unsafe, _cached_thresholds=unsafe)
    fake_backends = types.ModuleType("mps_sdpa.backends")
    fake_backends._calibrate = fake_cal  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mps_sdpa", types.ModuleType("mps_sdpa"))
    monkeypatch.setitem(sys.modules, "mps_sdpa.backends", fake_backends)

    mps_prebuilt_ext.install_mps_sdpa_threshold_guard()
    out = fake_cal.get_thresholds()
    assert out["fused_min_bytes"]["bf16"] == 4 * 1024**2
    assert fake_cal._cached_thresholds["fused_min_bytes"]["bf16"] == 4 * 1024**2


def test_mps_sdpa_call_stats_includes_fallback_reasons(monkeypatch) -> None:
    fake_api = types.ModuleType("mps_sdpa.api")
    fake_api.get_call_stats = lambda: {"stock_fallback": 12}  # type: ignore[attr-defined]
    fake_api.get_fallback_stats = lambda: {"short-seq": 12}  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("mps_sdpa")
    fake_pkg.api = fake_api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mps_sdpa", fake_pkg)
    monkeypatch.setitem(sys.modules, "mps_sdpa.api", fake_api)

    text = mps_prebuilt_ext._mps_sdpa_call_stats()
    assert "stock_fallback=12" in text
    assert "fb:short-seq=12" in text
