"""RGB lerp shared by Home/Remote retake and extend stitches.

Same control flow as ltxv-api ``stitch_frames``
(``inference/pipelines/src/pipelines/retake/pipeline.py``): up to
``MAX_FADE_FRAMES`` of ``lerp``, then the generated edit, then the fade back.
ffmpeg ``xfade`` in yuv420p is not that lerp and flashes highlights.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from handlers.video_resolution import SpatialLetterbox
from services.retake_pipeline.letterbox import crop_letterbox, resize_rgb
from services.retake_pipeline.window import MAX_FADE_FRAMES


@dataclass(frozen=True)
class FadeWindow:
    """Coordinates on the generated file, which starts at context frame 0."""

    context_start: int
    context_end: int
    edit_start: int
    edit_end: int


def generated_weight(index: int, window: FadeWindow) -> float:
    """Weight of the generated frame. 0 = original, 1 = generated."""
    fade_in = min(MAX_FADE_FRAMES, window.edit_start - window.context_start)
    fade_out = min(MAX_FADE_FRAMES, window.context_end - window.edit_end)
    fade_in_start = window.edit_start - fade_in
    fade_out_end = window.edit_end + fade_out
    if fade_in > 0 and fade_in_start <= index < window.edit_start:
        return (index - fade_in_start) / fade_in
    if window.edit_start <= index < window.edit_end:
        return 1.0
    if fade_out > 0 and window.edit_end <= index < fade_out_end:
        return (fade_out_end - index - 1) / fade_out
    return 0.0


def exact_rate(fps: float) -> Fraction:
    """24000/1001 for 23.976, 25/1 for 25. The rational the source was mastered at."""
    return Fraction(fps).limit_denominator(1001)


def blend_rgb(source: np.ndarray, generated: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return source
    if weight >= 1:
        return generated
    mixed = source.astype(np.float32) * (1.0 - weight) + generated.astype(np.float32) * weight
    return np.clip(mixed, 0, 255).astype(np.uint8)


# Below this, decoding from frame 0 is cheaper than a keyframe seek and is exact.
_SEEK_MIN_START_FRAME = 48


def _frame_index(
    pts: int | None, time_base: Fraction | None, rate: Fraction, origin: int
) -> int | None:
    """Decoded-frame index. ``origin`` is the stream start, in ``time_base`` ticks."""
    if pts is None or time_base is None:
        return None
    return int(round((Fraction(pts) - origin) * time_base * rate))


def _decode_span(path: str, start_frame: int, end_frame: int) -> Iterator[np.ndarray]:
    """Frames ``[start_frame, end_frame)`` as decoded. May end short.

    A late span seeks to the previous keyframe and counts forward from the
    stream start, not from timestamp zero. If that landing is already past
    the span, or the file has no timestamps, decode from frame 0.
    """
    if end_frame <= start_frame:
        return
    with av.open(path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        time_base = stream.time_base
        origin = 0 if stream.start_time is None else int(stream.start_time)
        index = 0
        first = None
        if (
            start_frame >= _SEEK_MIN_START_FRAME
            and rate
            and time_base is not None
        ):
            target_pts = origin + int(Fraction(start_frame) / Fraction(rate) / time_base)
            try:
                container.seek(
                    max(target_pts, 0), stream=stream, backward=True, any_frame=False
                )
                decoded = container.decode(stream)
                first = next(decoded, None)
            except av.FFmpegError:
                container.seek(0, stream=stream, backward=True, any_frame=False)
                decoded = container.decode(stream)
                first = None
            else:
                landed = _frame_index(
                    None if first is None else first.pts,
                    time_base,
                    Fraction(rate),
                    origin,
                )
                if first is None or landed is None or landed < 0 or landed > start_frame:
                    container.seek(0, stream=stream, backward=True, any_frame=False)
                    decoded = container.decode(stream)
                    first = None
                else:
                    index = landed
        else:
            decoded = container.decode(stream)

        if first is not None:
            if index >= start_frame:
                yield first.to_ndarray(format="rgb24")
            index += 1
        for frame in decoded:
            if index >= end_frame:
                break
            if index >= start_frame:
                yield frame.to_ndarray(format="rgb24")
            index += 1


def iter_span(
    path: str, start_frame: int, end_frame: int, width: int, height: int
) -> Iterator[np.ndarray]:
    """Frames ``[start_frame, end_frame)`` at ``width`` x ``height``.

    Raises if the file ends early: a short span would desync the audio cut.
    """
    count = 0
    for frame in _decode_span(path, start_frame, end_frame):
        yield resize_rgb(frame, width, height)
        count += 1
    expected = max(0, end_frame - start_frame)
    if count != expected:
        raise ValueError(
            f"Span [{start_frame}, {end_frame}) of {path} has {count} frames, expected {expected}"
        )


def iter_source(
    path: str, start_frame: int, end_frame: int, box: SpatialLetterbox
) -> Iterator[np.ndarray]:
    """Original frames at the picture size."""
    return iter_span(path, start_frame, end_frame, box.content_width, box.content_height)


def iter_generated(
    path: str, start_frame: int, end_frame: int, box: SpatialLetterbox
) -> Iterator[np.ndarray]:
    """Generated frames with the encode's black bars cropped off.

    Resizing the letterboxed encode down to the picture would squash the bars
    into the image and shift the seam.
    """
    for frame in iter_span(path, start_frame, end_frame, box.canvas_width, box.canvas_height):
        yield crop_letterbox(frame, box)


def blend_fade(
    *,
    source_path: str,
    generated_path: str,
    source_start: int,
    generated_start: int,
    count: int,
    fade: FadeWindow,
    box: SpatialLetterbox,
) -> Iterator[np.ndarray]:
    """``count`` frames of the lerp between the original and the generated clip."""
    pairs = zip(
        iter_source(source_path, source_start, source_start + count, box),
        iter_generated(generated_path, generated_start, generated_start + count, box),
        strict=True,
    )
    for offset, (source_frame, generated_frame) in enumerate(pairs):
        yield blend_rgb(
            source_frame, generated_frame, generated_weight(generated_start + offset, fade)
        )


def write_rgb_video(
    dest: Path,
    frames: Iterator[np.ndarray],
    *,
    fps: float,
    width: int,
    height: int,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rate = exact_rate(fps)
    with av.open(str(dest), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)  # type: ignore[reportUnknownMemberType]
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        # yuv420p is limited-range. Tagging these full-range RGB frames as
        # JPEG/full makes the decoder skip the limited-to-full expand, so black
        # comes back as 16. Limited (MPEG) is the tag that round-trips.
        stream.codec_context.colorspace = 1
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 1
        stream.codec_context.color_range = 1
        pts = 0
        for pixels in frames:
            video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            video_frame.pts = pts
            pts += 1
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
