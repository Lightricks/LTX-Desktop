"""Output oracles for the live regression suite.

The original checks only prove that a file exists. These prove the output behaves like
the request: it decodes, is not blank or frozen, honours its image/audio/video
conditioning, and is reproducible per seed. They need only ffmpeg and the stdlib.

Every factory returns a ``check(path) -> (ok, detail)`` compatible with
``Scenario.checks``. A check that cannot run (ffmpeg missing, unreadable file) FAILS:
a gate that silently skips is a false pass.

Thresholds are deliberately loose heuristics. The first hardware runs (Windows RTX 5090 and
macOS M5 Max, ltx-core 1.4.1) cleared them with margin: identical same-seed runs score inf dB
against the 30 dB floor, a different seed 13.7 dB, a LoRA against the plain run 16.4 dB, and
frames kept from a source video correlate at 1.00 against the 0.85 floor. Tighten them from
real runs if a regression slips through (the detail string reports the measured value).
"""

from __future__ import annotations

import array
import re
import statistics
import subprocess
from collections.abc import Callable, Sequence
from functools import wraps

import perf_config

Check = Callable[[str | None], tuple[bool, str]]

FRAME_MATCH_MIN_CORR = 0.85      # 32x18 grayscale thumbnail correlation vs a reference frame
AUDIO_MATCH_MIN_CORR = 0.80      # 0.1 s RMS-envelope correlation vs a reference soundtrack
SAME_SEED_MIN_PSNR_DB = 30.0     # same request twice must be near-identical
BLACK_LUMA_MAX = 8.0             # per-frame mean luma (0-255) at or below counts as black
WHITE_LUMA_MIN = 247.0
FROZEN_MIN_MEAN_DIFF = 0.2       # mean abs frame-to-frame luma change below this = frozen
_THUMB_W, _THUMB_H = 32, 18
_AUDIO_RATE, _AUDIO_WINDOW = 8000, 800  # 0.1 s windows

_REMEMBERED: dict[str, str] = {}


def reset() -> None:
    """Forget every remembered reference (a sweep starts with none)."""
    _REMEMBERED.clear()


def _run(args: list[str], timeout: float = 180) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, capture_output=True, timeout=timeout)


def _guarded(name: str) -> Callable[[Callable[..., tuple[bool, str]]], Callable[..., tuple[bool, str]]]:
    """Name the check for the report and turn tool failures into a FAIL."""

    def deco(fn: Callable[..., tuple[bool, str]]) -> Callable[..., tuple[bool, str]]:
        @wraps(fn)
        def wrapper(path: str | None) -> tuple[bool, str]:
            if not path:
                return False, "no output path"
            try:
                return fn(path)
            except FileNotFoundError:
                return False, "ffmpeg/ffprobe not on PATH"
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                return False, f"{type(exc).__name__}: {exc}"

        wrapper.__name__ = name
        return wrapper

    return deco


def _thumbs(path: str) -> list[bytes]:
    proc = _run([
        # passthrough: real frames only. A variable-frame-rate output (the queued retake
        # asset) would otherwise be resampled to a constant rate with duplicated frames.
        "ffmpeg", "-v", "error", "-i", path, "-an", "-fps_mode", "passthrough",
        "-vf", f"scale={_THUMB_W}:{_THUMB_H},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ])
    size = _THUMB_W * _THUMB_H
    if proc.returncode != 0 or len(proc.stdout) < size:
        raise RuntimeError((proc.stderr or b"").decode("utf-8", "replace")[:200] or "no frames decoded")
    raw = proc.stdout
    return [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        return statistics.correlation(a, b)
    except statistics.StatisticsError:  # a constant signal (black frame, silence) matches nothing
        return 0.0


def _pick(frames: list[bytes], at: str | float, duration: float | None) -> bytes:
    if at == "first":
        return frames[0]
    if at == "last":
        return frames[-1]
    if not duration:
        raise RuntimeError(f"duration unknown, cannot pick the frame at {at}s")
    return frames[min(len(frames) - 1, round(float(at) / duration * (len(frames) - 1)))]


def _duration(path: str) -> float | None:
    return (perf_config.ffprobe_info(path) or {}).get("duration")


def _envelope(path: str) -> list[float]:
    proc = _run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(_AUDIO_RATE),
                 "-f", "f32le", "pipe:1"])
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or b"").decode("utf-8", "replace")[:200])
    samples = array.array("f")
    samples.frombytes(proc.stdout[: len(proc.stdout) // 4 * 4])
    return [
        (sum(x * x for x in samples[i:i + _AUDIO_WINDOW]) / _AUDIO_WINDOW) ** 0.5
        for i in range(0, len(samples) - _AUDIO_WINDOW + 1, _AUDIO_WINDOW)
    ]


def media_ok(*, audio: bool | None = None, duration: tuple[float, float] | None = None) -> Check:
    """Decodes end to end with no errors; optional audio stream and duration window (seconds)."""

    @_guarded("media_ok")
    def check(path: str) -> tuple[bool, str]:
        proc = _run(["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"])
        if proc.returncode != 0 or proc.stderr.strip():
            return False, "decode errors: " + proc.stderr.decode("utf-8", "replace")[:160]
        probe = perf_config.ffprobe_info(path) or {}
        problems = []
        dur, fps, frames = probe.get("duration"), probe.get("fps"), probe.get("nb_frames")
        if duration and not (dur and duration[0] <= dur <= duration[1]):
            problems.append(f"duration {dur} not in {duration}")
        if dur and fps and frames and abs(frames - dur * fps) > 3:
            problems.append(f"{frames} frames != {dur:.2f}s*{fps:g}fps")
        if audio and not probe.get("has_audio"):
            problems.append("audio stream missing")
        return not problems, "; ".join(problems) or f"{dur}s {fps}fps {frames}f audio={probe.get('has_audio')}"

    return check


def not_dead() -> Check:
    """No black, blown-out, or frozen video (every frame is inspected, not just the tail)."""

    @_guarded("not_dead")
    def check(path: str) -> tuple[bool, str]:
        frames = _thumbs(path)
        if len(frames) < 2:
            return False, f"only {len(frames)} frame(s)"
        means = [sum(f) / len(f) for f in frames]
        dark = sum(m <= BLACK_LUMA_MAX for m in means)
        bright = sum(m >= WHITE_LUMA_MIN for m in means)
        motion = statistics.fmean(
            sum(abs(x - y) for x, y in zip(a, b)) / len(a) for a, b in zip(frames, frames[1:])
        )
        problems = []
        if dark:
            problems.append(f"{dark} black frame(s)")
        if bright:
            problems.append(f"{bright} blown-out frame(s)")
        if motion < FROZEN_MIN_MEAN_DIFF:
            problems.append(f"frozen (motion {motion:.2f})")
        return not problems, "; ".join(problems) or f"{len(frames)} frames, motion {motion:.2f}"

    return check


def frame_matches(
    reference: str,
    *,
    at: str | float = "first",
    reference_at: str | float = "first",
    min_corr: float = FRAME_MATCH_MIN_CORR,
) -> Check:
    """A frame of the output (``first``/``last``/seconds) resembles a frame of ``reference``
    (an image or a video). Used for keyframes and for regions a retake/extend must keep."""

    @_guarded(f"frame_matches[{at}]")
    def check(path: str) -> tuple[bool, str]:
        got = _pick(_thumbs(path), at, _duration(path))
        want = _pick(_thumbs(reference), reference_at, _duration(reference))
        corr = _corr(list(got), list(want))
        return corr >= min_corr, f"corr={corr:.2f} (min {min_corr})"

    return check


def audio_matches(reference: str, *, min_corr: float = AUDIO_MATCH_MIN_CORR) -> Check:
    """The output soundtrack follows ``reference``'s loudness envelope (A2V returns the
    source audio; a video-only retake keeps it). Silence never matches."""

    @_guarded("audio_matches")
    def check(path: str) -> tuple[bool, str]:
        got, want = _envelope(path), _envelope(reference)
        n = min(len(got), len(want))
        if n < 5:
            return False, f"too little audio to compare ({len(got)} vs {len(want)} windows)"
        corr = _corr(got[:n], want[:n])
        return corr >= min_corr, f"corr={corr:.2f} over {n / 10:.1f}s (min {min_corr})"

    return check


def _psnr(a: str, b: str) -> float:
    proc = _run(["ffmpeg", "-v", "info", "-i", a, "-i", b, "-lavfi", "psnr", "-f", "null", "-"])
    match = re.search(r"average:(inf|[0-9.]+)", proc.stderr.decode("utf-8", "replace"))
    if not match:
        raise RuntimeError("psnr unavailable (different sizes?)")
    return float("inf") if match.group(1) == "inf" else float(match.group(1))


def remember(key: str) -> Check:
    """Record this output as the reference ``key`` for a later ``reproduces``/``differs_from``."""

    @_guarded(f"remember[{key}]")
    def check(path: str) -> tuple[bool, str]:
        _REMEMBERED[key] = path
        return True, f"saved as '{key}'"

    return check


def _compare(key: str, want_same: bool) -> Check:
    @_guarded(f"{'reproduces' if want_same else 'differs_from'}[{key}]")
    def check(path: str) -> tuple[bool, str]:
        ref = _REMEMBERED.get(key)
        if ref is None:  # never pass on nothing: an unverified comparison is not a green one
            return False, f"no reference '{key}' recorded: run its scenario first, in the same sweep"
        psnr = _psnr(path, ref)
        same = psnr >= SAME_SEED_MIN_PSNR_DB
        return same == want_same, f"psnr={psnr:.1f}dB (same-seed threshold {SAME_SEED_MIN_PSNR_DB})"

    return check


def reproduces(key: str) -> Check:
    """Same request as the remembered run: near-identical output (seed, cache and adapter hygiene)."""
    return _compare(key, want_same=True)


def differs_from(key: str) -> Check:
    """A visibly different output than the remembered run: another seed is honoured, or an
    adapter that was requested on an otherwise identical request is really applied."""
    return _compare(key, want_same=False)


def seed_batch() -> Callable[[list[str | None]], tuple[bool, str]]:
    """For three jobs submitted together with seeds A, B, A: the two A outputs match and B
    differs. Proves back-to-back queued jobs neither leak into nor drop each other."""

    def check(outs: list[str | None]) -> tuple[bool, str]:
        if len(outs) != 3 or not all(outs):
            return False, f"expected 3 outputs, got {outs}"
        try:
            same, other = _psnr(str(outs[0]), str(outs[2])), _psnr(str(outs[0]), str(outs[1]))
        except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        ok = same >= SAME_SEED_MIN_PSNR_DB > other
        return ok, f"same-seed psnr={same:.1f}dB, other-seed psnr={other:.1f}dB (threshold {SAME_SEED_MIN_PSNR_DB})"

    check.__name__ = "seed_batch"
    return check
