from __future__ import annotations

from pathlib import Path

from services.encoded_media import (
    check_output_size,
    derived_file_name,
    ingest_encoded,
)
from services.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, ffmpeg_seconds, run_ffmpeg
from services.media_probe import MAX_IMAGE_BYTES
from services.media_source import require_source
from services.records import AssetRecord, MediaError
from services.store import Store

# How far before the end the last-frame decode starts. Longer than any frame interval.
_LAST_FRAME_WINDOW_SECONDS = 1.0


def extract_frame_file(src: Path, dest: Path, at_sec: float | None) -> None:
    """``at_sec`` None takes the last frame."""
    if at_sec is None:
        # Decode the final second and keep overwriting the image: an exact seek to the
        # duration lands after the last frame and writes nothing.
        seek = ["-sseof", ffmpeg_seconds(-_LAST_FRAME_WINDOW_SECONDS)]
        frames = ["-update", "1"]
    else:
        seek = ["-ss", ffmpeg_seconds(at_sec)]
        frames = ["-frames:v", "1"]
    run_ffmpeg(
        [
            "-y",
            *seek,
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-i",
            str(src),
            "-map",
            "0:v:0",
            *frames,
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            str(dest),
        ]
    )


def extract_frame_asset(
    store: Store, asset_id: str, at_sec: float | None
) -> AssetRecord:
    record, source = require_source(store, asset_id, "video")
    metadata = record.metadata
    if at_sec is not None:
        if at_sec < 0:
            raise MediaError("INVALID_FRAME_TIME", "time must not be negative")
        if (
            metadata.mediaType == "video"
            and at_sec >= metadata.metadata.durationMs / 1000
        ):
            raise MediaError("INVALID_FRAME_TIME", "time is past the end of the video")

    def write(tmp_dir: Path) -> Path:
        suffix = "last-frame" if at_sec is None else f"frame-{at_sec:g}s"
        dest = tmp_dir / derived_file_name(record.name, suffix, ".png")
        extract_frame_file(source, dest, at_sec)
        # A time inside the last frame's interval is after its start, so ffmpeg exits 0
        # without writing. The frame on screen then is the last one.
        if not dest.is_file():
            extract_frame_file(source, dest, None)
        if not dest.is_file():
            raise MediaError("UNREADABLE_MEDIA", "the video has no decodable frame")
        check_output_size(dest, MAX_IMAGE_BYTES, "image")
        return dest

    return ingest_encoded(
        store,
        write,
        prefix="ltx-video-frame-",
        label="image",
        failure="ffmpeg failed to extract a frame",
    )
