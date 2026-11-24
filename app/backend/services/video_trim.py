from __future__ import annotations

import threading
from pathlib import Path

from services.encoded_media import (
    check_output_size,
    derived_file_name,
    ingest_encoded,
)
from services.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, ffmpeg_seconds, run_ffmpeg
from services.media_probe import MAX_VIDEO_BYTES
from services.media_source import require_source
from services.records import AssetRecord, MediaError
from services.store import Store

# Re-encoding up to a minute of 4K source is slower than an audio trim.
VIDEO_TRIM_TIMEOUT_SECONDS = 300.0

# Video trims run one at a time so parallel requests (e.g. from a paired
# phone) cannot stack encodes. A queued request gives up after this long
# rather than holding a server worker thread indefinitely.
VIDEO_TRIM_QUEUE_TIMEOUT_SECONDS = 30.0
_ENCODE_SLOT = threading.Lock()


def trim_video_file(src: Path, dest: Path, start_sec: float, end_sec: float) -> None:
    run_ffmpeg(
        [
            "-y",
            # Input-side seek is frame-accurate when re-encoding and does not
            # decode everything before the In point of a long source.
            "-ss",
            ffmpeg_seconds(start_sec),
            "-t",
            ffmpeg_seconds(end_sec - start_sec),
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            # yuv420p + libx264 reject odd dimensions; ingest accepts them.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-fs",
            str(MAX_VIDEO_BYTES),
            str(dest),
        ],
        timeout=VIDEO_TRIM_TIMEOUT_SECONDS,
    )


def trim_video_asset(
    store: Store, asset_id: str, start_sec: float, end_sec: float
) -> AssetRecord:
    if end_sec <= start_sec:
        raise ValueError("endSec must be greater than startSec")
    record, source = require_source(store, asset_id, "video")
    metadata = record.metadata
    # ffmpeg exits 0 with an empty, stream-less file when seeking past the end.
    if (
        metadata.mediaType == "video"
        and start_sec >= metadata.metadata.durationMs / 1000
    ):
        raise MediaError("INVALID_TRIM_RANGE", "startSec is past the end of the video")

    def write(tmp_dir: Path) -> Path:
        dest = tmp_dir / derived_file_name(record.name, "trim", ".mp4")
        if not _ENCODE_SLOT.acquire(timeout=VIDEO_TRIM_QUEUE_TIMEOUT_SECONDS):
            raise MediaError("TRIM_BUSY", "another video trim is in progress")
        try:
            trim_video_file(source, dest, start_sec, end_sec)
        finally:
            _ENCODE_SLOT.release()
        check_output_size(dest, MAX_VIDEO_BYTES, "video")
        return dest

    return ingest_encoded(
        store,
        write,
        prefix="ltx-video-trim-",
        label="video",
        failure="ffmpeg failed to trim video",
    )
