"""CPU-only checks for stitching original pixels back onto Home retake."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from services.retake_pipeline.retake_stitch import (
    fade_window_for_retake,
    reencode_span,
    retake_audio_graph,
    stitch_retake_output,
)
from services.ffmpeg import (
    FFMPEG_TIMEOUT_SECONDS,
    STITCH_TIMEOUT_CAP_SECONDS,
    timeout_for_source_duration,
)
from services.retake_pipeline.rgb_stitch import iter_span
from services.retake_pipeline.window import (
    MAX_FADE_FRAMES,
    denoised_frame_span,
    feathered_mask_times,
    window_for_retake,
)


NTSC_FILM = 24000 / 1001


def _ted_retake_window():
    """The 30.6s–35.6s retake on a 60s 23.976 fps clip that jumped at the seam."""
    return window_for_retake(
        source_frames=1438,
        mask_start_frame=734,
        mask_end_frame=854,
        max_input_frames=505,
        fps=NTSC_FILM,
    )


def test_fade_sits_on_the_user_mask_for_at_most_16_frames() -> None:
    window = _ted_retake_window()
    fade = fade_window_for_retake(window)
    fade_in = min(MAX_FADE_FRAMES, fade.edit_start - fade.context_start)
    fade_out = min(MAX_FADE_FRAMES, fade.context_end - fade.edit_end)
    denoised_start, denoised_end = denoised_frame_span(window, NTSC_FILM)

    assert fade.edit_start == window.context_before_frames
    assert fade.edit_end - fade.edit_start == 120
    assert fade_in == MAX_FADE_FRAMES
    assert fade_out == MAX_FADE_FRAMES
    # Fully generated frames are the user mask. The denoised latents run wider;
    # those extra frames are blended or replaced with the original.
    assert denoised_start < fade.edit_start < fade.edit_end <= denoised_end


def test_denoised_span_snaps_out_to_whole_latents() -> None:
    window = _ted_retake_window()
    start, end = denoised_frame_span(window, NTSC_FILM)
    region_start, region_end = feathered_mask_times(window, NTSC_FILM)
    assert start / NTSC_FILM <= region_start and end / NTSC_FILM >= region_end
    assert (start - 1) % 8 == 0 and (end - 1) % 8 == 0


def test_mask_at_start_has_no_fade_in() -> None:
    window = window_for_retake(
        source_frames=241,
        mask_start_frame=0,
        mask_end_frame=48,
        max_input_frames=505,
        fps=25.0,
    )
    fade = fade_window_for_retake(window)
    assert fade.edit_start == 0
    assert min(MAX_FADE_FRAMES, fade.context_end - fade.edit_end) == MAX_FADE_FRAMES


def test_audio_cuts_at_the_mask_while_video_fades_wider() -> None:
    window = _ted_retake_window()
    graph = retake_audio_graph(window, NTSC_FILM, source_label="[1:a]")
    edit_start = f"{window.mask_start_frame / NTSC_FILM:.6f}"
    edit_end = f"{window.mask_end_frame / NTSC_FILM:.6f}"
    assert f"atrim=end={edit_start}" in graph
    assert f"atrim=start={edit_end}" in graph
    assert "concat=n=3:v=0:a=1" in graph
    fade = fade_window_for_retake(window)
    fade_in_start = window.encode_start_frame + fade.edit_start - MAX_FADE_FRAMES
    assert f"{fade_in_start / NTSC_FILM:.6f}" not in graph


def test_full_mask_stitch_copies_the_windowed_file(tmp_path: Path) -> None:
    windowed = tmp_path / "window.mp4"
    dest = tmp_path / "out.mp4"
    windowed.write_bytes(b"window-bytes")
    window = window_for_retake(
        source_frames=97,
        mask_start_frame=0,
        mask_end_frame=97,
        max_input_frames=505,
        fps=25.0,
    )
    stitch_retake_output(
        source_path=str(tmp_path / "source.mp4"),
        windowed_path=str(windowed),
        dest_path=str(dest),
        window=window,
        fps=25.0,
        target_width=64,
        target_height=64,
    )
    assert dest.read_bytes() == b"window-bytes"


def _solid_video(
    path: Path,
    *,
    rgb: tuple[int, int, int],
    frames: int,
    fps: int,
    height: int = 64,
    width: int = 64,
) -> str:
    import imageio.v2 as imageio

    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None)
    frame = np.full((height, width, 3), rgb, dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return str(path)


def _barred_video(path: Path, *, frames: int, fps: int) -> str:
    """64x64 canvas with an 8px black bar on the top and bottom around a 48px picture."""
    import imageio.v2 as imageio

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[8:56] = 180
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return str(path)


def test_stitch_keeps_source_outside_a_16_frame_lerp(tmp_path: Path) -> None:
    fps = 25.0
    window = window_for_retake(
        source_frames=81,
        mask_start_frame=32,
        mask_end_frame=56,
        max_input_frames=505,
        fps=fps,
    )
    source = _solid_video(tmp_path / "source.mp4", rgb=(0, 0, 0), frames=81, fps=25)
    generated = _solid_video(
        tmp_path / "generated.mp4",
        rgb=(30, 30, 30),
        frames=window.encode_frames,
        fps=25,
    )
    dest = tmp_path / "out.mp4"
    stitch_retake_output(
        source_path=source,
        windowed_path=generated,
        dest_path=str(dest),
        window=window,
        fps=fps,
        target_width=64,
        target_height=64,
    )
    fade = fade_window_for_retake(window)
    fade_in = min(MAX_FADE_FRAMES, fade.edit_start)
    with __import__("av").open(str(dest)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]

    assert len(frames) == 81
    assert frames[0].mean() == pytest.approx(0, abs=8)
    assert frames[fade.edit_start - fade_in - 1].mean() == pytest.approx(0, abs=8)
    mid = fade.edit_start - fade_in // 2
    assert 8 < frames[mid].mean() < 22
    assert frames[fade.edit_start].mean() == pytest.approx(30, abs=8)
    assert frames[fade.edit_end - 1].mean() == pytest.approx(30, abs=8)
    assert frames[-1].mean() == pytest.approx(0, abs=8)


def test_stitch_crops_letterbox_bars_instead_of_stretching_them(tmp_path: Path) -> None:
    fps = 25.0
    window = window_for_retake(
        source_frames=81,
        mask_start_frame=32,
        mask_end_frame=56,
        max_input_frames=505,
        fps=fps,
    )
    source = _solid_video(
        tmp_path / "source.mp4", rgb=(180, 180, 180), frames=81, fps=25, height=48
    )
    generated = _barred_video(
        tmp_path / "generated.mp4", frames=window.encode_frames, fps=25
    )
    dest = tmp_path / "out.mp4"
    stitch_retake_output(
        source_path=source,
        windowed_path=generated,
        dest_path=str(dest),
        window=window,
        fps=fps,
        target_width=64,
        target_height=48,
    )
    fade = fade_window_for_retake(window)
    with __import__("av").open(str(dest)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    edited = frames[fade.edit_start]
    assert edited.shape == (48, 64, 3)
    assert edited[0].mean() == pytest.approx(180, abs=15)


def test_reencode_span_is_the_mask_plus_the_fades() -> None:
    window = _ted_retake_window()
    span = reencode_span(window)
    assert span.fade_out_end - span.fade_in_start == 120 + 2 * MAX_FADE_FRAMES
    assert 0 < span.prefix_end < span.suffix_start < window.source_frames


def test_span_past_the_end_of_the_file_raises_instead_of_yielding_short(tmp_path: Path) -> None:
    path = tmp_path / "indexed.mp4"
    _indexed_clip(path, 20, 24)
    with pytest.raises(ValueError, match="expected 10"):
        list(iter_span(str(path), 15, 25, 16, 16))


def test_python_writer_receives_only_the_blend_and_the_mask(monkeypatch, tmp_path: Path) -> None:
    fps = 25.0
    window = window_for_retake(
        source_frames=81,
        mask_start_frame=32,
        mask_end_frame=56,
        max_input_frames=505,
        fps=fps,
    )
    _solid_video(tmp_path / "source.mp4", rgb=(0, 0, 0), frames=81, fps=25)
    _solid_video(
        tmp_path / "generated.mp4", rgb=(30, 30, 30), frames=window.encode_frames, fps=25
    )
    seen: list[int] = []

    def _capture(dest: Path, frames: object, **_kwargs: object) -> None:
        seen.append(sum(1 for _ in frames))  # type: ignore[arg-type]
        Path(dest).write_bytes(b"middle")

    monkeypatch.setattr("services.retake_pipeline.retake_stitch.write_rgb_video", _capture)
    monkeypatch.setattr("services.retake_pipeline.retake_stitch.run_ffmpeg", lambda *_a, **_k: None)
    monkeypatch.setattr("services.retake_pipeline.retake_stitch._has_audio", lambda _path: False)
    stitch_retake_output(
        source_path=str(tmp_path / "source.mp4"),
        windowed_path=str(tmp_path / "generated.mp4"),
        dest_path=str(tmp_path / "out.mp4"),
        window=window,
        fps=fps,
        target_width=64,
        target_height=64,
    )
    span = reencode_span(window)
    assert seen == [span.fade_out_end - span.fade_in_start]
    assert seen[0] < window.source_frames


def test_short_generated_span_raises(tmp_path: Path) -> None:
    fps = 25.0
    window = window_for_retake(
        source_frames=81,
        mask_start_frame=32,
        mask_end_frame=56,
        max_input_frames=505,
        fps=fps,
    )
    source = _solid_video(tmp_path / "source.mp4", rgb=(0, 0, 0), frames=81, fps=25)
    generated = _solid_video(tmp_path / "generated.mp4", rgb=(30, 30, 30), frames=1, fps=25)
    with pytest.raises(ValueError, match="Span"):
        stitch_retake_output(
            source_path=source,
            windowed_path=generated,
            dest_path=str(tmp_path / "out.mp4"),
            window=window,
            fps=fps,
            target_width=64,
            target_height=64,
        )


def test_silent_source_keeps_generated_audio(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def _capture(args: list[str], **_kwargs: object) -> None:
        captured.append(args)

    monkeypatch.setattr("services.retake_pipeline.retake_stitch.run_ffmpeg", _capture)
    monkeypatch.setattr(
        "services.retake_pipeline.retake_stitch.write_rgb_video",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "services.retake_pipeline.retake_stitch._has_audio",
        lambda path: path == "/window.mp4",
    )
    window = window_for_retake(
        source_frames=241,
        mask_start_frame=96,
        mask_end_frame=144,
        max_input_frames=505,
        fps=25.0,
    )
    dest = tmp_path / "out.mp4"
    stitch_retake_output(
        source_path="/source.mp4",
        windowed_path="/window.mp4",
        dest_path=str(dest),
        window=window,
        fps=25.0,
        target_width=64,
        target_height=64,
    )
    assert len(captured) == 1
    args = captured[0]
    assert "-f" in args and args[args.index("-f") + 1] == "lavfi"
    assert "anullsrc=" in args[args.index("-i", args.index("-f")) + 1]
    assert args.index("-shortest") > max(i for i, arg in enumerate(args) if arg == "-i")
    graph = args[args.index("-filter_complex") + 1]
    span = reencode_span(window)
    assert f"trim=end_frame={span.prefix_end}" in graph
    assert f"trim=start_frame={span.suffix_start}" in graph
    assert "[1:v]" in graph
    assert "[2:a]" in graph
    assert "[3:a]" in graph
    assert "[0:a]" not in graph
    assert "[1:a]" not in graph


def test_stitch_timeout_grows_with_the_source_and_stays_bounded() -> None:
    assert timeout_for_source_duration(10) == FFMPEG_TIMEOUT_SECONDS
    assert timeout_for_source_duration(60) == 480
    assert timeout_for_source_duration(10_000) == STITCH_TIMEOUT_CAP_SECONDS


def _indexed_clip(path: Path, frames: int, rate) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 8
        stream.codec_context.color_range = 1
        for index in range(frames):
            pixels = np.full((16, 16, 3), (index * 40) % 256, dtype=np.uint8)
            video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            video_frame.pts = index
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_late_rgb_span_matches_a_decode_from_the_start(tmp_path: Path) -> None:
    from fractions import Fraction

    path = tmp_path / "indexed.mp4"
    _indexed_clip(path, 160, Fraction(24000, 1001))
    full = list(iter_span(str(path), 0, 160, 16, 16))
    late = list(iter_span(str(path), 100, 108, 16, 16))
    assert np.array_equal(late, full[100:108])
    assert not np.array_equal(late[0], late[1])


def _shift_start_time(source: Path, dest: Path, offset: str) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(source),
            "-c",
            "copy",
            "-output_ts_offset",
            offset,
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


def test_late_rgb_span_matches_when_timestamps_do_not_start_at_zero(tmp_path: Path) -> None:
    path = tmp_path / "indexed.mp4"
    shifted = tmp_path / "shifted.mp4"
    _indexed_clip(path, 180, 24)
    _shift_start_time(path, shifted, "0.125")
    full = list(iter_span(str(shifted), 0, 180, 16, 16))
    late = list(iter_span(str(shifted), 100, 108, 16, 16))
    assert np.array_equal(late, full[100:108])
