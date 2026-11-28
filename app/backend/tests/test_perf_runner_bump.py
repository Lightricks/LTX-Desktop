"""Unit tests for performance_runner bump-gate wiring (no live backend)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PERF = Path(__file__).resolve().parents[1] / "performance_runner"
if str(_PERF) not in sys.path:
    sys.path.insert(0, str(_PERF))

import scenarios as scn  # noqa: E402


def test_bump_tag_covers_gate_scenarios() -> None:
    keys = {s.key for s in scn.all_scenarios() if "bump" in s.tags}
    assert keys >= {
        "t2v_540p_8s",
        "t2v_1080p_5s",
        "mkf_interpolation",
        "iclora_day_to_night",
        "iclora_stage2_on",
        "cancel_mid_denoise",
    }


def test_cancel_scenario_does_not_expect_output() -> None:
    s = scn.SCENARIOS["cancel_mid_denoise"]
    assert s.cancel_after_s == 4.0
    assert s.checks == []


def test_last_frames_not_black_missing_file() -> None:
    ok, msg = scn.last_frames_not_black(None)
    assert ok is False
    assert "no file" in msg


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_last_frames_not_black_detects_black_tail(tmp_path: Path) -> None:
    # Bright body + black tail: sampling from t-1s without reversing would still
    # see white and miss E2. 8 fps, 1s: first 4 frames white, last 4 black.
    mixed = tmp_path / "white_then_black.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=white:s=64x64:d=0.5:r=8",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.5:r=8",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-pix_fmt", "yuv420p", str(mixed),
        ],
        check=True,
        capture_output=True,
    )
    ok, msg = scn.last_frames_not_black(str(mixed), n_frames=4)
    assert ok is False, msg
    assert "BLACK" in msg

    white = tmp_path / "white.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=64x64:d=1:r=8",
            "-pix_fmt", "yuv420p", str(white),
        ],
        check=True,
        capture_output=True,
    )
    ok, msg = scn.last_frames_not_black(str(white), n_frames=4)
    assert ok is True, msg


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_last_frames_not_black_fails_partial_black_tail(tmp_path: Path) -> None:
    # 8 fps, 1s: first 5 white, last 3 black. Mean-of-tail would still pass;
    # per-frame min must fail.
    mixed = tmp_path / "mostly_white_black_tail.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=white:s=64x64:d=0.625:r=8",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.375:r=8",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-pix_fmt", "yuv420p", str(mixed),
        ],
        check=True,
        capture_output=True,
    )
    ok, msg = scn.last_frames_not_black(str(mixed), n_frames=8)
    assert ok is False, msg
    assert "BLACK" in msg


def test_last_frames_not_black_fails_without_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not-a-real-video")

    def boom(*_a, **_k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(scn.subprocess, "run", boom)
    ok, msg = scn.last_frames_not_black(str(clip), n_frames=4)
    assert ok is False
    assert "ffmpeg not on PATH" in msg
