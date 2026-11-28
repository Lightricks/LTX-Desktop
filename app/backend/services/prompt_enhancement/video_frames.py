"""Evenly spaced stills from a source video, for the video-aware prompt enhancer."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import av
from PIL import Image

from services.media_probe import probe_file
from services.prompt_enhancement.i2v_frames import SceneFrame
from services.records import MediaError

# The number of stills follows the clip length: a short clip needs few, a long one more, and
# the cap keeps Gemma VRAM and Gemini tokens bounded.
MIN_SCENE_FRAMES = 2
MAX_SCENE_FRAMES = 8
_SECONDS_PER_SCENE_FRAME = 1.5
# Same long edge as the keyframe path: a larger still costs VRAM (Gemma) and tokens (Gemini)
# without improving the rewrite.
SCENE_FRAME_LONG_EDGE = 896
_JPEG_QUALITY = 85
_SEEK_TOLERANCE_SECONDS = 1e-3


def scene_frame_count(duration_seconds: float) -> int:
    """How many stills a clip of ``duration_seconds`` gets: about one per 1.5 s, from 2 to 8."""
    wanted = round(duration_seconds / _SECONDS_PER_SCENE_FRAME)
    return max(MIN_SCENE_FRAMES, min(MAX_SCENE_FRAMES, wanted))


def _segment_midpoints(duration_seconds: float, count: int) -> list[float]:
    """Midpoints of ``count`` equal segments. The first frame of a clip is often black."""
    return [duration_seconds * (index + 0.5) / count for index in range(count)]


def _sample_offsets(
    duration_seconds: float, count: int, replaced_window: tuple[float, float] | None
) -> list[tuple[float, str, float | None]]:
    """``(seconds, note, before)`` for each still. Nothing is sampled inside ``replaced_window``.

    ``before`` is the time that a still on the near side of the window must stay under, as the
    first frame at or after a target can fall past the window start in a clip with few frames.

    The window is a part of the clip that the job replaces (Retake), so its pixels say nothing
    about the new content. The stills spread over the rest of the clip, before and after it.
    A window that covers the whole clip leaves no context, so the whole clip is sampled.
    """
    if replaced_window is not None:
        start = max(0.0, replaced_window[0])
        end = min(duration_seconds, replaced_window[1])
        before = start
        context = before + (duration_seconds - end)
        if end > start and context > 0:
            where = f"the replaced part is {start:.2f}s to {end:.2f}s"
            return [
                (
                    (position, f"before it, {where}", start)
                    if position < before
                    else (end + position - before, f"after it, {where}", None)
                )
                for position in _segment_midpoints(context, count)
            ]
    return [(offset, "", None) for offset in _segment_midpoints(duration_seconds, count)]


# PyAV's container, stream, and frame types are not precise enough for strict mode.
# These helpers take them as Any, as the other PyAV call sites in this codebase do.
def _stream_origin(stream: Any) -> float:
    """Seconds at which the stream's timeline starts. Cut or re-muxed clips often do not start at 0."""
    if stream.start_time is None or stream.time_base is None:
        return 0.0
    return float(stream.start_time * stream.time_base)


def _frame_at(container: Any, stream: Any, target: float, *, before: float | None = None) -> Any | None:
    """First frame at or after ``target`` (stream timeline), or the last one decoded when the clip ends first.

    With ``before``, no frame at or after that time is used. The last frame under it stands in,
    because a sparse clip has no frame between the target and the replaced part.
    """
    if stream.time_base is not None:
        try:
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
        except av.FFmpegError:
            return None
    last: Any | None = None
    try:
        for frame in container.decode(stream):
            if before is not None and frame.time is not None and frame.time + _SEEK_TOLERANCE_SECONDS >= before:
                return last
            last = frame
            if frame.time is not None and frame.time + _SEEK_TOLERANCE_SECONDS >= target:
                return frame
    except av.FFmpegError:
        # A damaged stretch of an otherwise readable clip: use what decoded before it.
        return last
    return last


def _oriented_image(frame: Any) -> Image.Image:
    """The frame as an upright RGB image: a phone clip stores its pixels sideways."""
    image = cast(Image.Image, frame.to_image()).convert("RGB")
    rotation = int(frame.rotation) % 360
    if rotation in (90, 180, 270):
        image = image.rotate(rotation, expand=True)
    image.thumbnail((SCENE_FRAME_LONG_EDGE, SCENE_FRAME_LONG_EDGE), Image.Resampling.LANCZOS)
    return image


@contextmanager
def sampled_video_frames(
    video_path: str,
    *,
    count: int | None = None,
    replaced_window: tuple[float, float] | None = None,
) -> Iterator[list[SceneFrame]]:
    """JPEG stills of ``video_path`` in time order. The files go when the context exits.

    ``count`` defaults to ``scene_frame_count`` of the clip's length. ``replaced_window`` is
    ``(start, end)`` seconds of a part the job replaces. No still comes from inside it, and each
    still is labeled as before or after it.

    Raises ``MediaError`` for a file that is not a readable video. A clip with fewer distinct
    frames than ``count`` yields fewer stills rather than repeats.
    """
    source = Path(video_path)
    kind, _mime, metadata = probe_file(source)
    if kind != "video" or metadata.mediaType != "video":
        raise MediaError("UNSUPPORTED_MEDIA", "not a video file")
    duration_seconds = metadata.metadata.durationMs / 1000
    still_count = scene_frame_count(duration_seconds) if count is None else count

    with tempfile.TemporaryDirectory(prefix="ltx-enhance-frames-") as tmp:
        frames: list[SceneFrame] = []
        seen_times: set[float] = set()
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            origin = _stream_origin(stream)
            for offset, note, before in _sample_offsets(duration_seconds, still_count, replaced_window):
                frame = _frame_at(
                    container,
                    stream,
                    origin + offset,
                    before=None if before is None else origin + before,
                )
                if frame is None or frame.time is None or float(frame.time) in seen_times:
                    continue
                seen_times.add(float(frame.time))
                path = Path(tmp) / f"frame-{len(frames):02d}.jpg"
                _oriented_image(frame).save(path, "JPEG", quality=_JPEG_QUALITY)
                # Labels are positions in the clip, so they count from the clip's own start.
                frames.append(
                    SceneFrame(
                        path=str(path),
                        time_seconds=max(0.0, float(frame.time) - origin),
                        note=note,
                    )
                )
        if not frames:
            raise MediaError("UNREADABLE_MEDIA", "no decodable video frames")
        yield sorted(frames, key=lambda item: item.time_seconds)
