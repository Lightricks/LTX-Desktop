"""The live-suite oracles must tell good output from broken output (synthetic media, no backend)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PERF = Path(__file__).resolve().parents[1] / "performance_runner"
if str(_PERF) not in sys.path:
    sys.path.insert(0, str(_PERF))

import checks  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe"
)

_MOVING = "mandelbrot=size=64x64:rate=24"
_STILL = "color=c=gray:s=64x64:r=24"
_BLACK = "color=c=black:s=64x64:r=24"
_WHITE = "color=c=white:s=64x64:r=24"
_OTHER = "smptebars=size=64x64:rate=24"


@pytest.fixture(autouse=True)
def _fresh_references():
    checks.reset()
    yield
    checks.reset()


def _make(tmp_path: Path, name: str, source: str, *, seconds: float = 1.0, audio: str | None = None) -> str:
    out = tmp_path / name
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-t", str(seconds), "-i", source]
    if audio:
        cmd += ["-f", "lavfi", "-t", str(seconds), "-i", audio, "-c:a", "aac"]
    if name.endswith(".mp4"):
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-frames:v", "1"]
    subprocess.run([*cmd, str(out)], check=True)
    return str(out)


def test_media_ok_accepts_a_good_clip_and_enforces_audio_and_duration(tmp_path: Path) -> None:
    silent = _make(tmp_path, "silent.mp4", _MOVING)
    voiced = _make(tmp_path, "voiced.mp4", _MOVING, audio="sine=frequency=440")

    assert checks.media_ok()(silent)[0]
    assert checks.media_ok(audio=True)(voiced)[0]
    assert not checks.media_ok(audio=True)(silent)[0]
    assert not checks.media_ok(duration=(3.0, 4.0))(silent)[0]


def test_media_ok_rejects_a_truncated_file(tmp_path: Path) -> None:
    good = Path(_make(tmp_path, "good.mp4", _MOVING, seconds=2))
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(good.read_bytes()[: good.stat().st_size // 2])

    assert not checks.media_ok()(str(bad))[0]
    assert not checks.media_ok()(str(tmp_path / "missing.mp4"))[0]


def test_not_dead_passes_motion_and_flags_black_white_and_frozen(tmp_path: Path) -> None:
    assert checks.not_dead()(_make(tmp_path, "moving.mp4", _MOVING))[0]
    assert "black" in checks.not_dead()(_make(tmp_path, "black.mp4", _BLACK))[1]
    assert "blown-out" in checks.not_dead()(_make(tmp_path, "white.mp4", _WHITE))[1]
    assert "frozen" in checks.not_dead()(_make(tmp_path, "still.mp4", _STILL))[1]


def test_frame_matches_finds_the_conditioning_image_and_rejects_another(tmp_path: Path) -> None:
    clip = _make(tmp_path, "clip.mp4", _MOVING)
    first = _make(tmp_path, "first.png", _MOVING)
    other = _make(tmp_path, "other.png", _OTHER)

    assert checks.frame_matches(first)(clip)[0]
    assert not checks.frame_matches(other)(clip)[0]
    assert not checks.frame_matches(first)(_make(tmp_path, "black.mp4", _BLACK))[0]


def test_frame_matches_can_compare_a_later_frame_to_a_source_clip(tmp_path: Path) -> None:
    clip = _make(tmp_path, "clip.mp4", _MOVING, seconds=2)

    assert checks.frame_matches(clip, at="last", reference_at="last")(clip)[0]
    assert not checks.frame_matches(clip, at="last", reference_at="first", min_corr=0.999)(clip)[0]


_TONE_A = "aevalsrc='sin(2*PI*440*t)*(0.5+0.5*sin(2*PI*1.3*t))':d=4"
_TONE_B = "aevalsrc='sin(2*PI*440*t)*(0.5+0.5*sin(2*PI*0.45*t+2))':d=4"


def test_audio_matches_follows_the_reference_envelope_and_rejects_silence(tmp_path: Path) -> None:
    source = _make(tmp_path, "src.mp4", _MOVING, seconds=4, audio=_TONE_A)
    same = _make(tmp_path, "same.mp4", _MOVING, seconds=4, audio=_TONE_A)
    different = _make(tmp_path, "different.mp4", _MOVING, seconds=4, audio=_TONE_B)
    silent = _make(tmp_path, "silent.mp4", _MOVING, seconds=4)

    assert checks.audio_matches(source)(same)[0]
    assert not checks.audio_matches(source)(different)[0]
    assert not checks.audio_matches(source)(silent)[0]


def test_reproduces_and_differs_from_compare_against_the_remembered_run(tmp_path: Path) -> None:
    ref = _make(tmp_path, "ref.mp4", _MOVING)
    again = _make(tmp_path, "again.mp4", _MOVING)
    other = _make(tmp_path, "other.mp4", _OTHER)
    assert checks.remember("t")(ref)[0]

    assert checks.reproduces("t")(again)[0]
    assert not checks.reproduces("t")(other)[0]
    assert checks.differs_from("t")(other)[0]
    assert not checks.differs_from("t")(again)[0]


def test_comparison_without_a_reference_fails_rather_than_passing_unverified(tmp_path: Path) -> None:
    out = _make(tmp_path, "x.mp4", _MOVING)

    for check in (checks.reproduces("never-saved"), checks.differs_from("never-saved")):
        ok, detail = check(out)
        assert not ok and "no reference" in detail


def test_reset_forgets_remembered_references(tmp_path: Path) -> None:
    out = _make(tmp_path, "x.mp4", _MOVING)
    checks.remember("k")(out)
    assert checks.reproduces("k")(out)[0]

    checks.reset()

    assert not checks.reproduces("k")(out)[0]


def test_seed_batch_needs_matching_first_and_last_and_a_different_middle(tmp_path: Path) -> None:
    a = _make(tmp_path, "a.mp4", _MOVING)
    a_again = _make(tmp_path, "a2.mp4", _MOVING)
    b = _make(tmp_path, "b.mp4", _OTHER)

    assert checks.seed_batch()([a, b, a_again])[0]
    assert not checks.seed_batch()([a, a_again, a])[0]  # seed had no effect
    assert not checks.seed_batch()([a, b, b])[0]  # third job diverged from the first
    assert not checks.seed_batch()([a, b])[0]  # a job went missing


def test_picking_a_timed_frame_without_a_known_duration_raises_instead_of_guessing() -> None:
    frames = [b"a", b"b", b"c"]

    assert checks._pick(frames, "last", None) == b"c"
    with pytest.raises(RuntimeError, match="duration unknown"):
        checks._pick(frames, 1.5, None)
