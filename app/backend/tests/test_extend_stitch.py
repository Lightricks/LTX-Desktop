"""CPU-only checks for stitching original pixels back onto Home/Remote extend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.retake_pipeline.window import MAX_FADE_FRAMES, MAX_INPUT_FRAMES, window_for_extend
from services.retake_pipeline.extend_stitch import fade_window_for_extend, stitch_extend_output
from services.retake_pipeline.rgb_stitch import FadeWindow, blend_rgb, generated_weight


def test_generated_weight_matches_ltxv_api_full_fade_bands() -> None:
    window = FadeWindow(context_start=0, context_end=12, edit_start=5, edit_end=8)
    expected = [0, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 1, 1, 1, 3 / 4, 2 / 4, 1 / 4, 0]
    assert [generated_weight(index, window) for index in range(12)] == pytest.approx(expected)


def test_generated_weight_clamps_when_context_gap_is_smaller_than_max_fade() -> None:
    window = FadeWindow(context_start=0, context_end=8, edit_start=2, edit_end=5)
    expected = [0, 1 / 2, 1, 1, 1, 2 / 3, 1 / 3, 0]
    assert [generated_weight(index, window) for index in range(8)] == pytest.approx(expected)


def test_blend_rgb_is_identity_at_the_endpoints() -> None:
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    generated = np.full((2, 2, 3), 30, dtype=np.uint8)
    assert np.array_equal(blend_rgb(source, generated, 0.0), source)
    assert np.array_equal(blend_rgb(source, generated, 1.0), generated)
    assert blend_rgb(source, generated, 0.5)[0, 0, 0] == 15


def test_end_extend_fade_window_starts_mask_delta_before_the_new_region() -> None:
    window = window_for_extend(
        source_frames=129,
        extend_frames=104,
        mode="end",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    fade = fade_window_for_extend(window, mode="end", extend_frames=104, fps=25.0)
    assert fade.edit_start == 129 - 12
    assert fade.edit_end == 129 + 104
    assert min(MAX_FADE_FRAMES, fade.edit_start - fade.context_start) == MAX_FADE_FRAMES


def test_start_extend_fade_window_has_no_fade_in() -> None:
    window = window_for_extend(
        source_frames=129,
        extend_frames=104,
        mode="start",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    fade = fade_window_for_extend(window, mode="start", extend_frames=104, fps=25.0)
    assert fade.edit_start == 0
    assert generated_weight(0, fade) == 1.0
    assert generated_weight(fade.edit_end, fade) < 1.0


def _solid_video(path: Path, *, rgb: tuple[int, int, int], frames: int, fps: int) -> str:
    import imageio.v2 as imageio

    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None)
    frame = np.full((64, 64, 3), rgb, dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return str(path)


def test_stitch_extend_output_lerps_in_rgb_not_lutrgb(tmp_path: Path) -> None:
    source = _solid_video(tmp_path / "source.mp4", rgb=(0, 0, 0), frames=25, fps=25)
    generated = _solid_video(tmp_path / "generated.mp4", rgb=(30, 30, 30), frames=41, fps=25)
    window = window_for_extend(
        source_frames=25,
        extend_frames=16,
        mode="end",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    dest = tmp_path / "out.mp4"
    stitch_extend_output(
        source_path=source,
        windowed_path=generated,
        dest_path=str(dest),
        window=window,
        fps=25.0,
        mode="end",
        extend_frames=16,
        target_width=64,
        target_height=64,
    )
    fade = fade_window_for_extend(window, mode="end", extend_frames=16, fps=25.0)
    fade_in = min(MAX_FADE_FRAMES, fade.edit_start - fade.context_start)
    fade_in_start = fade.edit_start - fade_in
    with __import__("av").open(str(dest)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    # First fade frame is pure source (weight 0). A later fade frame is in between.
    assert frames[fade_in_start].mean() == pytest.approx(0, abs=8)
    mid = fade_in_start + fade_in // 2
    assert 8 < frames[mid].mean() < 22
    assert frames[fade.edit_start].mean() == pytest.approx(30, abs=8)


def test_start_extend_lerps_after_the_generated_prefix(tmp_path: Path) -> None:
    source = _solid_video(tmp_path / "source.mp4", rgb=(0, 0, 0), frames=25, fps=25)
    generated = _solid_video(tmp_path / "generated.mp4", rgb=(30, 30, 30), frames=41, fps=25)
    window = window_for_extend(
        source_frames=25,
        extend_frames=16,
        mode="start",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    dest = tmp_path / "out.mp4"
    stitch_extend_output(
        source_path=source,
        windowed_path=generated,
        dest_path=str(dest),
        window=window,
        fps=25.0,
        mode="start",
        extend_frames=16,
        target_width=64,
        target_height=64,
    )
    fade = fade_window_for_extend(window, mode="start", extend_frames=16, fps=25.0)
    with __import__("av").open(str(dest)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert frames[0].mean() == pytest.approx(30, abs=8)
    assert frames[fade.edit_end - 1].mean() == pytest.approx(30, abs=8)
    assert frames[-1].mean() == pytest.approx(0, abs=8)


def test_stitch_extend_output_keeps_generated_audio_when_source_is_silent(
    monkeypatch, tmp_path: Path
) -> None:
    captured: list[list[str]] = []

    def _capture(args: list[str], **_kwargs: object) -> None:
        captured.append(args)

    monkeypatch.setattr("services.retake_pipeline.extend_stitch.run_ffmpeg", _capture)
    monkeypatch.setattr(
        "services.retake_pipeline.extend_stitch._has_audio",
        lambda path: "generated" in path or "window" in path,
    )
    source = _solid_video(tmp_path / "source.mp4", rgb=(0, 0, 0), frames=25, fps=25)
    generated = _solid_video(tmp_path / "window.mp4", rgb=(30, 30, 30), frames=41, fps=25)
    window = window_for_extend(
        source_frames=25,
        extend_frames=16,
        mode="end",
        max_input_frames=MAX_INPUT_FRAMES,
    )
    stitch_extend_output(
        source_path=source,
        windowed_path=generated,
        dest_path=str(tmp_path / "out.mp4"),
        window=window,
        fps=25.0,
        mode="end",
        extend_frames=16,
        target_width=64,
        target_height=64,
    )
    assert len(captured) == 1
    args = captured[0]
    assert "-c:v" in args and args[args.index("-c:v") + 1] == "copy"
    assert "anullsrc=" in args[args.index("-i", args.index("-f")) + 1]
    # anullsrc never ends: -shortest is an output option, so it must follow the last -i.
    assert args.index("-shortest") > max(i for i, arg in enumerate(args) if arg == "-i")
    assert "[3:a]" in args[args.index("-filter_complex") + 1]
    assert "lutrgb=" not in args[args.index("-filter_complex") + 1]
    assert "xfade=" not in args[args.index("-filter_complex") + 1]
