"""Published matrix is the 181 generate projection, not a second Yes table."""

from __future__ import annotations

from pathlib import Path

from runtime_config.local_compat_matrix import MATRIX_PATH, render_compat_matrix, write_compat_matrix


def test_cuda_stream_t2v_yes_but_not_a_full_yes_table() -> None:
    md = render_compat_matrix()
    assert "| T2V | No | Yes | Yes | No | Yes | No |" in md
    assert "CUDA stream (16 GB) still runs T2V" in md
    assert "hides 720p/20s" in md
    assert "1080p/10s" in md


def test_darwin_keeps_ceiling_and_ic_lora_footnote() -> None:
    md = render_compat_matrix()
    assert "| IC-LoRA | No | Yes | Yes | No | Yes | No |" in md
    assert "Darwin IC-LoRA duration is unbounded" in md
    assert "no token 422" in md
    assert "Do not invent a CUDA-curve ceiling" in md


def test_intel_mac_and_cuda_unsupported_are_no() -> None:
    md = render_compat_matrix()
    assert "| 2.5 download | No | Yes | Yes | No | Yes | No |" in md
    assert "integer **15** GiB VRAM" in md
    assert "Darwin **32** GiB total RAM" in md
    assert "~72 GB" in md


def test_retake_extend_follow_the_default_local_model() -> None:
    """2.5 advertises Retake/Extend, so hardware-viable columns are a plain Yes."""
    md = render_compat_matrix()
    assert "| Retake | No | Yes | Yes | No | Yes | No |" in md
    assert "| Extend | No | Yes | Yes | No | Yes | No |" in md


def test_committed_matrix_matches_renderer() -> None:
    assert MATRIX_PATH.read_text(encoding="utf-8") == render_compat_matrix()


def test_write_compat_matrix_roundtrip(tmp_path: Path) -> None:
    path = write_compat_matrix(tmp_path / "local-compat-matrix.md")
    assert path.read_text(encoding="utf-8") == render_compat_matrix()
