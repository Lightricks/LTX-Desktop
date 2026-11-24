"""Fix the frame rate the upstream encoder truncates.

``ltx_pipelines`` ``encode_video`` opens its stream with ``rate=int(fps)``, so a
23.976 fps retake/extend window is written at 23 fps: the video plays ~4% slow
against its audio, and a timestamp seek lands on the wrong frame. Rescaling
the video timestamps by a stream copy puts every frame back at
``index / source_fps`` without re-encoding.
"""

from __future__ import annotations

import os
from pathlib import Path

from services.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, run_ffmpeg

_RATE_TOLERANCE = 1e-6


def restamp_to_source_rate(path: str, *, source_fps: float) -> None:
    written_fps = int(source_fps)
    if written_fps <= 0 or abs(source_fps - written_fps) < _RATE_TOLERANCE:
        return
    target = Path(path)
    restamped = target.with_name(f"{target.stem}.restamp{target.suffix}")
    run_ffmpeg(
        [
            "-y",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-itsscale:v",
            f"{written_fps / source_fps:.12f}",
            "-i",
            str(target),
            "-map",
            "0",
            "-c",
            "copy",
            str(restamped),
        ]
    )
    os.replace(restamped, target)
