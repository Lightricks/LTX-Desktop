"""Put original pixels back so Home/Remote extend is not a VAE round-trip of the source.

Ports ltxv-api ``stitch_frames`` / ``stitch_audio``
(``inference/pipelines/src/pipelines/retake/pipeline.py``): RGB ``lerp`` up to
``MAX_FADE_FRAMES``, audio cut at the edit span with a short fade on each side of the cut.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from inference.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, run_ffmpeg, timeout_for_source_duration
from inference.media import media_has_audio
from inference.retake_pipeline.audio_level import generated_gain_db
from inference.retake_pipeline.letterbox import SpatialLetterbox, generation_letterbox
from inference.types import ExtendMode
from inference.retake_pipeline.rgb_stitch import (
    FadeWindow,
    blend_fade,
    iter_generated,
    iter_source,
    write_rgb_video,
)
from inference.retake_pipeline.stitch_audio import AudioPiece, audio_cut_graph
from inference.retake_pipeline.stitch_silence import append_silence_input
from inference.retake_pipeline.window import (
    MASK_DELTA_SECONDS,
    MAX_FADE_FRAMES,
    ExtendEncodeWindow,
)

_has_audio = media_has_audio


def _mask_delta_frames(fps: float, context_frames: int) -> int:
    frames = round(MASK_DELTA_SECONDS * fps)
    return max(0, min(frames, max(0, context_frames - 1)))


def fade_window_for_extend(
    window: ExtendEncodeWindow,
    *,
    mode: ExtendMode,
    extend_frames: int,
    fps: float,
    seam_frames: int | None = None,
) -> FadeWindow:
    """``seam_frames``: real source frames next to the seam, when the window holds a voice
    bridge. The rest of the window is other moments of the source, so the lerp must not
    reach into it."""
    delta = _mask_delta_frames(fps, window.context_frames)
    context_end = window.context_frames + extend_frames
    real = window.context_frames if seam_frames is None else seam_frames
    if mode == "end":
        return FadeWindow(
            window.context_frames - real, context_end, window.context_frames - delta, context_end
        )
    return FadeWindow(0, extend_frames + real, 0, extend_frames + delta)


def _stitched_frames(
    *,
    source_path: str,
    generated_path: str,
    window: ExtendEncodeWindow,
    fade: FadeWindow,
    mode: ExtendMode,
    extend_frames: int,
    box: SpatialLetterbox,
) -> Iterator[np.ndarray]:
    origin = window.dropped_prefix_frames
    if mode == "end":
        fade_in = min(MAX_FADE_FRAMES, fade.edit_start - fade.context_start)
        fade_in_start = fade.edit_start - fade_in
        yield from iter_source(source_path, 0, origin + fade_in_start, box)
        yield from blend_fade(
            source_path=source_path,
            generated_path=generated_path,
            source_start=origin + fade_in_start,
            generated_start=fade_in_start,
            count=fade_in,
            fade=fade,
            box=box,
        )
        yield from iter_generated(generated_path, fade.edit_start, fade.context_end, box)
        return
    fade_out = min(MAX_FADE_FRAMES, fade.context_end - fade.edit_end)
    source_fade_start = fade.edit_end - extend_frames
    source_end = (
        window.dropped_prefix_frames + window.context_frames + window.dropped_suffix_frames
    )
    yield from iter_generated(generated_path, 0, fade.edit_end, box)
    yield from blend_fade(
        source_path=source_path,
        generated_path=generated_path,
        source_start=source_fade_start,
        generated_start=fade.edit_end,
        count=fade_out,
        fade=fade,
        box=box,
    )
    yield from iter_source(source_path, source_fade_start + fade_out, source_end, box)


def _mux_audio(
    *,
    video_path: str,
    source_path: str,
    generated_path: str,
    dest_path: str,
    mode: ExtendMode,
    source_cut: float,
    generated_cut: float,
    silence_original: bool,
    generated_gain_db: float,
    timeout: float,
) -> None:
    args: list[str] = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        video_path,
        "-i",
        source_path,
        "-i",
        generated_path,
    ]
    source_audio = "[1:a]"
    if silence_original:
        append_silence_input(args)
        source_audio = "[3:a]"
    generated_audio = "[2:a]"
    if mode == "end":
        graph = audio_cut_graph(
            [
                AudioPiece(source_audio, end=source_cut),
                AudioPiece(generated_audio, start=generated_cut, gain_db=generated_gain_db),
            ]
        )
    else:
        graph = audio_cut_graph(
            [
                AudioPiece(generated_audio, end=generated_cut, gain_db=generated_gain_db),
                AudioPiece(source_audio, start=source_cut),
            ]
        )
    args.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "0:v",
            "-c:v",
            "copy",
            "-map",
            "[a]",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            dest_path,
        ]
    )
    run_ffmpeg(args, timeout=timeout)


def stitch_extend_output(
    *,
    source_path: str,
    windowed_path: str,
    dest_path: str,
    window: ExtendEncodeWindow,
    fps: float,
    mode: ExtendMode,
    extend_frames: int,
    target_width: int,
    target_height: int,
    seam_frames: int | None = None,
) -> None:
    box = generation_letterbox(target_width, target_height)
    fade = fade_window_for_extend(
        window, mode=mode, extend_frames=extend_frames, fps=fps, seam_frames=seam_frames
    )
    dest = Path(dest_path)
    generated_has_audio = _has_audio(windowed_path)
    silence_original = generated_has_audio and not _has_audio(source_path)
    video_dest = (
        dest if not generated_has_audio else dest.with_name(f"{dest.stem}.video{dest.suffix}")
    )
    write_rgb_video(
        video_dest,
        _stitched_frames(
            source_path=source_path,
            generated_path=windowed_path,
            window=window,
            fade=fade,
            mode=mode,
            extend_frames=extend_frames,
            box=box,
        ),
        fps=fps,
        width=box.content_width,
        height=box.content_height,
    )
    if not generated_has_audio:
        return
    if mode == "end":
        source_cut = (window.dropped_prefix_frames + fade.edit_start) / fps
        generated_cut = fade.edit_start / fps
    else:
        source_cut = (fade.edit_end - extend_frames) / fps
        generated_cut = fade.edit_end / fps
    source_frames = (
        window.dropped_prefix_frames + window.context_frames + window.dropped_suffix_frames
    )
    duration = (source_frames + extend_frames) / fps
    gain_db = (
        0.0
        if silence_original
        else generated_gain_db(
            source_path=source_path,
            generated_path=windowed_path,
            mode=mode,
            generated_cut=generated_cut,
            source_seconds=source_frames / fps,
            generated_seconds=(window.context_frames + extend_frames) / fps,
        )
    )
    try:
        _mux_audio(
            video_path=str(video_dest),
            source_path=source_path,
            generated_path=windowed_path,
            dest_path=str(dest),
            mode=mode,
            source_cut=source_cut,
            generated_cut=generated_cut,
            silence_original=silence_original,
            generated_gain_db=gain_db,
            timeout=timeout_for_source_duration(duration),
        )
    finally:
        if video_dest != dest:
            video_dest.unlink(missing_ok=True)
