"""Published matrix is the 181 generate projection, not a second Yes table."""

from __future__ import annotations

from pathlib import Path

from runtime_config.local_compat_matrix import MATRIX_PATH, render_compat_matrix, write_compat_matrix


def test_cuda_stream_t2v_yes_but_not_a_full_yes_table() -> None:
    md = render_compat_matrix()
    assert "| T2V | No | Yes | Yes | Yes | No | No | No | Yes | No |" in md
    assert "CUDA stream (16 GB) still runs T2V" in md
    assert "disables 720p/20s and 1080p at 8s and up (greyed out with the reason)" in md
    assert "1080p/10s" in md


def test_windows_bands_cover_stream_and_ram_gate() -> None:
    md = render_compat_matrix()
    assert "| I2V | No | Yes | Yes | Yes | No | No | No | Yes | No |" in md
    assert "| A2V | No | Yes | Yes | Yes | No | No | No | Yes | No |" in md
    # The gated-band No pins the §3.8 fix: without system/model/vram/ram
    # threaded into advertised_fast_durations, the RAM gate never applies
    # and these flip back to Yes.
    assert "Windows stream (16 GB VRAM, 40 GB RAM)" in md
    assert "Windows streamed Fast needs **40** GiB total RAM" in md
    assert "ram_shortfall" in md
    assert "free VRAM" in md
    assert "reads weights from disk and the RAM floor is **24** GiB" in md


def test_darwin_keeps_ceiling_and_ic_lora_footnote() -> None:
    md = render_compat_matrix()
    # Windows-stream column is No: the 40 GB probe RAM trips the L5 64 GB
    # floor (both paths). RAM-gated column stays No with it.
    assert "| IC-LoRA | No | Yes | Yes | No | No | No | No | Yes | No |" in md
    assert "Darwin IC-LoRA gates each duration against the sampled MPS" in md
    assert "I1/I2 model" in md
    assert "keeps 20s portrait" in md
    assert "Darwin stream gates each Fast cell against the sampled MPS" in md


def test_intel_mac_and_cuda_unsupported_are_no() -> None:
    md = render_compat_matrix()
    assert "| 2.5 download | No | Yes | Yes | Yes | Yes | No | No | Yes | No |" in md
    assert "Windows **9** (fp8 cards; **15** without fp8) / Linux **15** GiB total VRAM (exact bytes, integer fallback)" in md
    assert "Darwin **32** GiB total RAM" in md
    assert "~72 GB" in md


def test_macos_below_floor_is_no_everywhere() -> None:
    """macOS 14 and 15 on a 48 GB Apple Silicon Mac runs nothing locally."""
    md = render_compat_matrix()
    assert "| macOS below 26 |" in md
    for row in ("T2V", "I2V", "A2V", "LoRAs", "IC-LoRA", "Retake", "Extend", "local text encoder", "2.5 download"):
        line = next(line for line in md.splitlines() if line.startswith(f"| {row} |"))
        # Columns: Tool + 9 bands. Band 7 is "macOS below 26".
        assert line.split(" | ")[7] == "No"
    assert "Darwin **32** GiB total RAM and macOS **26** or later" in md
    assert "macOS 14 and 15 run the app but not local generation" in md


def test_retake_extend_gate_on_ram_floor_with_input_dependent_vram() -> None:
    """RAM-gated says Yes, runs nothing: 32 GB trips the 64 GB CPU-path floor.

    2.5 advertises Retake/Extend, so bands that clear viability + the floor
    are Yes — with an asterisk, since VRAM depends on the source length and
    no canonical input exists for a representative cell. Bands carry no
    drive class, so both Windows bands take the CPU path (64 GB).
    """
    md = render_compat_matrix()
    assert "| Retake | No | Yes* | Yes* | No | No | No | No | Yes* | No |" in md
    assert "| Extend | No | Yes* | Yes* | No | No | No | No | Yes* | No |" in md
    assert "VRAM depends on\nyour source length" in md
    assert "Windows bands take the CPU path (64 GB floor" in md


def test_committed_matrix_matches_renderer() -> None:
    assert MATRIX_PATH.read_text(encoding="utf-8") == render_compat_matrix()


def test_write_compat_matrix_roundtrip(tmp_path: Path) -> None:
    path = write_compat_matrix(tmp_path / "local-compat-matrix.md")
    assert path.read_text(encoding="utf-8") == render_compat_matrix()
