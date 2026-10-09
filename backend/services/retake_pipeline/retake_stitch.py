"""Put original pixels back so Home retake is not a VAE round-trip outside the edit.

Same stitch as ltxv-api ``stitch_frames`` / ``stitch_audio``
(``inference/pipelines/src/pipelines/retake/pipeline.py``): original frames
outside a short RGB lerp, generated frames for the user mask, audio cut on
that mask. The lerp is at most ``MAX_FADE_FRAMES`` (16).

Only that band is re-encoded in Python. The prefix and suffix are frame-trimmed
in ffmpeg. A bitstream copy cannot cut on a frame index, so those wings are
encoded once by that trim, not decoded in Python first. GenSpace
``POST /api/retake`` does not call this.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from services.ffmpeg import (
    FFMPEG_PROTOCOL_WHITELIST,
    run_ffmpeg,
    timeout_for_source_duration,
)
from services.media_probe import media_has_audio
from handlers.video_resolution import SpatialLetterbox, generation_letterbox
from services.retake_pipeline.rgb_stitch import (
    FadeWindow,
    blend_fade,
    exact_rate,
    iter_generated,
    write_rgb_video,
)
from services.retake_pipeline.stitch_audio import AudioPiece, audio_cut_graph
from services.retake_pipeline.stitch_silence import append_silence_input
from services.retake_pipeline.window import MAX_FADE_FRAMES, RetakeEncodeWindow

_has_audio = media_has_audio


@dataclass(frozen=True)
class ReencodeSpan:
    """Source-absolute prefix/suffix, and the window-relative band Python encodes."""

    prefix_end: int
    suffix_start: int
    fade_in_start: int
    edit_start: int
    edit_end: int
    fade_out_end: int


def fade_window_for_retake(window: RetakeEncodeWindow) -> FadeWindow:
    """User mask in generated-file frames. Context outside it stays original."""
    mask_frames = window.mask_end_frame - window.mask_start_frame
    return FadeWindow(
        0,
        window.encode_frames,
        window.context_before_frames,
        window.context_before_frames + mask_frames,
    )


def reencode_span(window: RetakeEncodeWindow) -> ReencodeSpan:
    fade = fade_window_for_retake(window)
    origin = window.encode_start_frame
    fade_in = min(MAX_FADE_FRAMES, fade.edit_start - fade.context_start)
    fade_out = min(MAX_FADE_FRAMES, fade.context_end - fade.edit_end)
    fade_in_start = fade.edit_start - fade_in
    fade_out_end = fade.edit_end + fade_out
    return ReencodeSpan(
        prefix_end=origin + fade_in_start,
        suffix_start=origin + fade_out_end,
        fade_in_start=fade_in_start,
        edit_start=fade.edit_start,
        edit_end=fade.edit_end,
        fade_out_end=fade_out_end,
    )


def retake_audio_graph(
    window: RetakeEncodeWindow, fps: float, *, source_label: str, generated_label: str = "[2:a]"
) -> str:
    """Hard-cut audio at the user mask. Video fades; audio does not."""
    mask_frames = window.mask_end_frame - window.mask_start_frame
    generated_start = window.context_before_frames / fps
    pieces: list[AudioPiece] = []
    if window.mask_start_frame > 0:
        pieces.append(AudioPiece(source_label, end=window.mask_start_frame / fps))
    pieces.append(
        AudioPiece(
            generated_label,
            start=generated_start,
            end=(window.context_before_frames + mask_frames) / fps,
        )
    )
    if window.mask_end_frame < window.source_frames:
        pieces.append(AudioPiece(source_label, start=window.mask_end_frame / fps))
    return audio_cut_graph(pieces)


def _middle_frames(
    *,
    source_path: str,
    generated_path: str,
    window: RetakeEncodeWindow,
    span: ReencodeSpan,
    box: SpatialLetterbox,
) -> Iterator[np.ndarray]:
    fade = fade_window_for_retake(window)
    yield from blend_fade(
        source_path=source_path,
        generated_path=generated_path,
        source_start=span.prefix_end,
        generated_start=span.fade_in_start,
        count=span.edit_start - span.fade_in_start,
        fade=fade,
        box=box,
    )
    yield from iter_generated(generated_path, span.edit_start, span.edit_end, box)
    yield from blend_fade(
        source_path=source_path,
        generated_path=generated_path,
        source_start=window.encode_start_frame + span.edit_end,
        generated_start=span.edit_end,
        count=span.fade_out_end - span.edit_end,
        fade=fade,
        box=box,
    )


def _video_filter(
    span: ReencodeSpan, window: RetakeEncodeWindow, fps: float, box: SpatialLetterbox
) -> str:
    rate = exact_rate(fps)
    restamp = f"setpts=N*{rate.denominator}/({rate.numerator}*TB)"
    scale = (
        f"scale={box.content_width}:{box.content_height}:flags=lanczos,setsar=1,format=yuv420p"
    )
    parts: list[str] = []
    labels: list[str] = []
    if span.prefix_end > 0:
        parts.append(f"[0:v]trim=end_frame={span.prefix_end},{restamp},{scale}[vp]")
        labels.append("[vp]")
    parts.append(f"[1:v]{restamp},{scale}[vm]")
    labels.append("[vm]")
    if span.suffix_start < window.source_frames:
        parts.append(f"[0:v]trim=start_frame={span.suffix_start},{restamp},{scale}[vs]")
        labels.append("[vs]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[v]")
    return ";".join(parts)


def _assemble(
    *,
    source_path: str,
    generated_path: str,
    middle_path: str,
    dest_path: str,
    window: RetakeEncodeWindow,
    span: ReencodeSpan,
    fps: float,
    box: SpatialLetterbox,
    generated_has_audio: bool,
    silence_original: bool,
) -> None:
    rate = exact_rate(fps)
    args: list[str] = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        source_path,
        "-i",
        middle_path,
    ]
    source_label = "[0:a]"
    graph = _video_filter(span, window, fps, box)
    if generated_has_audio:
        args.extend(["-i", generated_path])
        if silence_original:
            append_silence_input(args)
            source_label = "[3:a]"
        graph = graph + ";" + retake_audio_graph(window, fps, source_label=source_label)
    args.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-r",
            f"{rate.numerator}/{rate.denominator}",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
        ]
    )
    if generated_has_audio:
        args.extend(["-map", "[a]"])
    args.extend(["-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, dest_path])
    run_ffmpeg(args, timeout=timeout_for_source_duration(window.source_frames / fps))


def _nothing_to_restore(window: RetakeEncodeWindow) -> bool:
    return (
        window.encode_start_frame == 0
        and window.context_before_frames == 0
        and window.context_after_frames == 0
        and window.encode_frames >= window.source_frames
    )


def stitch_retake_output(
    *,
    source_path: str,
    windowed_path: str,
    dest_path: str,
    window: RetakeEncodeWindow,
    fps: float,
    target_width: int,
    target_height: int,
) -> None:
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    box = generation_letterbox(target_width, target_height)
    if _nothing_to_restore(window):
        _place_window(windowed_path, dest, box)
        return
    span = reencode_span(window)
    if span.suffix_start < span.prefix_end:
        raise ValueError("retake re-encode span is inverted")
    generated_has_audio = _has_audio(windowed_path)
    silence_original = generated_has_audio and not _has_audio(source_path)
    middle = dest.with_name(f"{dest.stem}.middle{dest.suffix}")
    try:
        write_rgb_video(
            middle,
            _checked_middle(
                source_path=source_path,
                generated_path=windowed_path,
                window=window,
                span=span,
                box=box,
            ),
            fps=fps,
            width=box.content_width,
            height=box.content_height,
        )
        _assemble(
            source_path=source_path,
            generated_path=windowed_path,
            middle_path=str(middle),
            dest_path=str(dest),
            window=window,
            span=span,
            fps=fps,
            box=box,
            generated_has_audio=generated_has_audio,
            silence_original=silence_original,
        )
    finally:
        middle.unlink(missing_ok=True)


def _place_window(windowed_path: str, dest: Path, box: SpatialLetterbox) -> None:
    """The whole clip was the encode. Crop the bars, or copy when there are none."""
    if box.canvas_width == box.content_width and box.canvas_height == box.content_height:
        shutil.copy2(windowed_path, dest)
        return
    run_ffmpeg(
        [
            "-y",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-i",
            windowed_path,
            "-vf",
            f"crop={box.content_width}:{box.content_height}:{box.left}:{box.top}",
            "-pix_fmt",
            "yuv420p",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            str(dest),
        ]
    )


def _checked_middle(
    *,
    source_path: str,
    generated_path: str,
    window: RetakeEncodeWindow,
    span: ReencodeSpan,
    box: SpatialLetterbox,
) -> Iterator[np.ndarray]:
    expected = span.fade_out_end - span.fade_in_start
    count = 0
    for frame in _middle_frames(
        source_path=source_path,
        generated_path=generated_path,
        window=window,
        span=span,
        box=box,
    ):
        yield frame
        count += 1
    if count != expected:
        raise ValueError(
            f"Retake blend produced {count} frames, expected {expected}"
        )
