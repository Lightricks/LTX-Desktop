"""Voice bridge for Short context Extend: pick the speech, build the clip, fail soft.

The planner is pure. ``find_voice_bridge`` and ``write_voice_bridge`` run real ffmpeg on small
clips whose "speech" is a tone, so no GPU or model is needed.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from inference.ffmpeg import run_ffmpeg
from inference.retake_pipeline.speech_analysis import ANALYSIS_SAMPLE_RATE, Span, decode_mono, speech_spans
from inference.retake_pipeline.voice_context import (
    VoiceBridge,
    _write_bridge_clip,
    find_voice_bridge,
    plan_voice_bridge,
    write_voice_bridge,
)
from inference.retake_pipeline.window import (
    CONTINUATION_CONTEXT_FRAMES,
    MAX_INPUT_FRAMES,
    ExtendEncodeWindow,
    continuation_context_frames,
    window_for_extend,
)
from tests.video_fixtures import write_clip_with_tone_spans, write_mp4

FPS = 24.0
SOURCE_FRAMES = 144  # 6 s


def _window(mode: str) -> ExtendEncodeWindow:
    return window_for_extend(
        source_frames=SOURCE_FRAMES,
        extend_frames=48,
        mode=mode,  # type: ignore[arg-type]
        max_input_frames=MAX_INPUT_FRAMES,
        max_context_frames=CONTINUATION_CONTEXT_FRAMES,
    )


def _plan(spans: list[Span], mode: str = "end") -> VoiceBridge | None:
    return plan_voice_bridge(
        spans, window=_window(mode), fps=FPS, source_frames=SOURCE_FRAMES, mode=mode  # type: ignore[arg-type]
    )


def test_end_bridge_fills_the_window_and_the_seam_piece_ends_at_the_source_end() -> None:
    # Bug: a bridge longer or shorter than the window shifts every latent the model reads.
    window = _window("end")
    bridge = _plan([Span(0.2, 2.5)])
    assert bridge is not None
    assert bridge.total_frames == window.context_frames
    assert bridge.seam_start_frame + bridge.seam_frames == SOURCE_FRAMES
    assert bridge.speech_start_frame + bridge.speech_frames == round(2.5 * FPS)


def test_start_bridge_puts_the_seam_piece_first_and_takes_the_next_speech() -> None:
    window = _window("start")
    bridge = _plan([Span(3.0, 5.0)], mode="start")
    assert bridge is not None
    assert bridge.total_frames == window.context_frames
    assert bridge.seam_start_frame == window.encode_start_frame
    assert bridge.speech_start_frame == round(3.0 * FPS)


def test_no_bridge_when_the_far_part_of_the_window_already_has_a_voice() -> None:
    # Bug: the bridge replaces a window that already holds the voice, and the real picture is lost.
    window_start = _window("end").start_time(FPS)
    in_window = Span(window_start + 0.05, window_start + 0.65)
    assert _plan([Span(0.2, 2.5), in_window]) is None


def test_no_bridge_without_a_usable_earlier_speech() -> None:
    # Bug: negative start frame, or a click is taken as the voice sample.
    assert _plan([]) is None
    assert _plan([Span(1.0, 1.3)]) is None


def test_speech_piece_start_is_clamped_to_the_first_frame() -> None:
    bridge = _plan([Span(0.0, 0.5)])
    assert bridge is not None
    assert bridge.speech_start_frame == 0


def test_a_short_pause_stays_inside_one_span() -> None:
    # Bug: a breath between words counts as a gap and wrongly triggers a bridge.
    rate = ANALYSIS_SAMPLE_RATE
    tone = np.sin(np.linspace(0, 2000 * np.pi, rate * 3, dtype=np.float32)) * 0.5
    short_pause, long_pause = tone.copy(), tone.copy()
    short_pause[rate : rate + int(rate * 0.15)] = 0
    long_pause[rate : rate + int(rate * 0.4)] = 0
    assert len(speech_spans(short_pause, rate)) == 1
    assert len(speech_spans(long_pause, rate)) == 2


def _frames(path: str) -> list[np.ndarray]:
    with av.open(path) as container:
        return [f.to_ndarray(format="rgb24").astype(np.float32) for f in container.decode(video=0)]


def _nearest_source_frame(frame: np.ndarray, source: list[np.ndarray]) -> int:
    return int(np.argmin([np.abs(frame - other).mean() for other in source]))


def test_bridge_clip_plays_the_speech_piece_then_the_seam_piece(tmp_path: Path) -> None:
    # Bug: the concat order is swapped, or the sound and the picture come from different pieces.
    source = write_clip_with_tone_spans(tmp_path / "source.mp4", frames=SOURCE_FRAMES, spans=[(0.2, 2.5)])
    dest = tmp_path / "bridge.mp4"
    window = _window("end")
    bridge = find_voice_bridge(str(source), window=window, fps=FPS, source_frames=SOURCE_FRAMES, mode="end")
    assert bridge is not None
    assert write_voice_bridge(str(source), str(dest), bridge, FPS)
    frames = _frames(str(dest))
    source_frames = _frames(str(source))
    assert len(frames) == window.context_frames
    assert abs(_nearest_source_frame(frames[0], source_frames) - bridge.speech_start_frame) <= 1
    assert abs(_nearest_source_frame(frames[bridge.speech_frames], source_frames) - bridge.seam_start_frame) <= 1
    audio = decode_mono(str(dest), window.context_frames / FPS)
    split = int(bridge.speech_frames / FPS * ANALYSIS_SAMPLE_RATE)
    assert np.abs(audio[: split - 800]).max() > 0.05
    assert np.abs(audio[split + 800 :]).max() < 0.01


def test_no_bridge_for_a_source_without_audio_or_one_that_cannot_be_read(tmp_path: Path) -> None:
    # Bug: the extend job fails because the bridge is only an aid.
    silent_source = write_mp4(tmp_path / "silent.mp4", seconds=6, fps=24)
    broken_source = tmp_path / "broken.mp4"
    broken_source.write_bytes(b"not a video")
    for source in (silent_source, broken_source):
        assert (
            find_voice_bridge(str(source), window=_window("end"), fps=FPS, source_frames=SOURCE_FRAMES, mode="end")
            is None
        )


def test_a_bridge_clip_that_cannot_be_written_leaves_no_file(tmp_path: Path) -> None:
    # Bug: a half-written file stays in the outputs folder, or the extend job fails.
    broken_source = tmp_path / "broken.mp4"
    broken_source.write_bytes(b"not a video")
    dest = tmp_path / "bridge.mp4"
    bridge = VoiceBridge(speech_start_frame=0, speech_frames=24, seam_start_frame=48, seam_frames=24, mode="end")
    assert write_voice_bridge(str(broken_source), str(dest), bridge, FPS) is False
    assert not dest.exists()


def test_end_bridge_uses_speech_that_runs_into_the_window() -> None:
    # Bug: a speaker who talks on into the window was dropped, because the span did not end
    # before the window, so there was no speech to build the bridge from.
    window = _window("end")
    window_start = window.start_time(FPS)
    bridge = _plan([Span(0.5, window_start + 0.2)])
    assert bridge is not None
    # The piece ends where the window starts, not inside it.
    assert bridge.speech_start_frame + bridge.speech_frames == round(window_start * FPS)


def test_start_bridge_uses_speech_that_runs_out_of_the_window() -> None:
    window = _window("start")
    window_end = window.context_frames / FPS
    bridge = _plan([Span(window_end - 0.2, window_end + 3.0)], mode="start")
    assert bridge is not None
    assert bridge.speech_start_frame == window.context_frames


def test_decode_mono_keeps_a_late_audio_track_on_the_clip_timeline(tmp_path: Path) -> None:
    # Bug: a track that starts 0.5 s late was read from time 0, so every speech span came out
    # 0.5 s early and the bridge cut the wrong frames (phone and screen recordings do this).
    clip = tmp_path / "late.mp4"
    run_ffmpeg(
        [
            "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24",
            "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip),
        ]
    )
    samples = decode_mono(str(clip), 3.0)
    first_loud = int(np.argmax(np.abs(samples) > 0.05)) / ANALYSIS_SAMPLE_RATE
    assert abs(first_loud - 0.5) < 0.05


def _late_track_clip(path: Path) -> Path:
    run_ffmpeg(
        [
            "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24",
            "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ]
    )
    return path


def test_decode_mono_reads_only_the_asked_stretch_on_the_clip_timeline(tmp_path: Path) -> None:
    # Bug: the level match decoded the whole track to measure ten seconds of it.
    clip = _late_track_clip(tmp_path / "late.mp4")
    first_second = decode_mono(str(clip), 3.0, end=1.0)
    assert abs(len(first_second) / ANALYSIS_SAMPLE_RATE - 1.0) < 0.05
    assert abs(int(np.argmax(np.abs(first_second) > 0.05)) / ANALYSIS_SAMPLE_RATE - 0.5) < 0.05
    last_second = decode_mono(str(clip), 3.0, start=2.0)
    assert abs(len(last_second) / ANALYSIS_SAMPLE_RATE - 1.0) < 0.1
    assert np.abs(last_second[: ANALYSIS_SAMPLE_RATE // 2]).max() > 0.05


def test_bridge_clip_keeps_a_late_audio_track_under_its_own_frames(tmp_path: Path) -> None:
    # Bug: a seam piece starting at frame 0 of a clip whose audio starts 0.5 s late had its
    # trim drop the missing lead-in, so the voice played before the lips moved.
    clip = _late_track_clip(tmp_path / "late.mp4")
    dest = tmp_path / "bridge.mp4"
    bridge = VoiceBridge(speech_start_frame=48, speech_frames=24, seam_start_frame=0, seam_frames=24, mode="start")
    _write_bridge_clip(str(clip), str(dest), bridge, FPS)
    audio = decode_mono(str(dest), 2.0)
    assert np.abs(audio[: int(0.4 * ANALYSIS_SAMPLE_RATE)]).max() < 0.01
    assert np.abs(audio[int(0.6 * ANALYSIS_SAMPLE_RATE) : int(0.9 * ANALYSIS_SAMPLE_RATE)]).max() > 0.05


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0])
def test_a_short_context_window_holds_a_voice_bridge_at_any_frame_rate(fps: float) -> None:
    # Bug: the fixed 49-frame window left under 0.4 s for speech once the 1.3 s seam piece took its
    # share at 30 fps, and nothing at 60 fps, so no bridge was ever built and nothing said why.
    source_frames = int(6 * fps)
    window = window_for_extend(
        source_frames=source_frames,
        extend_frames=48,
        mode="end",
        max_input_frames=MAX_INPUT_FRAMES,
        max_context_frames=continuation_context_frames(fps),
    )
    bridge = plan_voice_bridge(
        [Span(0.2, 2.5)], window=window, fps=fps, source_frames=source_frames, mode="end"
    )
    assert bridge is not None
    assert bridge.total_frames == window.context_frames
