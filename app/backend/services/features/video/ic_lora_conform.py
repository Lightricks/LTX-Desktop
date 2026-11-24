"""Keep every frame and the original soundtrack of a clip that an IC-LoRA recipe restyles.

The pipeline needs 8k+1 frames and snaps a longer clip DOWN, so it would drop the tail. The
recipe repeats the last frame up to the next 8k+1 count before the run, and cuts the output
back to the clip length after it. The source soundtrack replaces the audio the pipeline
wrote, because that audio is the source decoded and encoded again.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from services.features.video.cutout_alignment import run_media_ffmpeg
from services.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, ffmpeg_seconds
from services.media_probe import first_audio_codec, video_fps, video_frame_count

# Audio codecs an mp4 holds as they are. WebM Vorbis and Opus do not fit, so they re-encode.
logger = logging.getLogger(__name__)

_MP4_COPY_AUDIO_CODECS = frozenset({"aac", "mp3", "ac3", "eac3"})
# Near lossless. The model output is already one lossy encode, so a cut must not add a visible one.
_TRIM_CRF = "14"
# yuv420p needs even sides. An odd side loses its last row or column, so no pixel is resampled.
_EVEN_CROP = "crop=trunc(iw/2)*2:trunc(ih/2)*2"


def pad_clip_to_frame_grid(
    source: Path, dest: Path, *, frames: int, target: int, keep_audio: bool
) -> Path:
    """Return ``source`` when it needs no pad, else ``dest``: the clip with its last frame
    repeated until it has ``target`` frames. ``frames`` is the frame count of ``source``.

    A ``target`` below ``frames`` needs no pad: the pipeline snaps the clip down itself.
    The pad stays lossless (``-qp 0``), so the model reads the pixels of ``source``. A side
    of odd length loses one row or column, because the encoder needs even sides.
    """
    pad = target - frames
    if pad <= 0:
        return source
    args = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        f"{_EVEN_CROP},tpad=stop_mode=clone:stop={pad}",
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
    ]
    args += ["-map", "0:a?", "-c:a", "aac"] if keep_audio else ["-an"]
    args += ["-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, str(dest)]
    run_media_ffmpeg(args, "clip preparation failed")
    return dest


def conform_output_to_clip(
    output: Path, *, frames: int, source_audio: Path | None
) -> None:
    """Rewrite ``output`` in place: at most ``frames`` frames, with the source soundtrack.

    ``source_audio`` is the original file, not the cropped copy. Its first audio stream is
    copied when an mp4 holds that codec, and cut to the video length. A video that needs no
    cut is copied too. The video is cut by frame count, not by time: a stream with B-frames
    starts its timestamps late, so a time cut keeps an extra frame.

    A ``source_audio`` with no audio stream is skipped, so the output keeps its own audio.
    Ingest can record audio for a file that no longer has it.
    """
    source_codec = None if source_audio is None else first_audio_codec(source_audio)
    if source_audio is not None and source_codec is None:
        logger.warning("[ic-lora] source %s has no audio stream, keeping the output audio", source_audio)
        source_audio = None
    output_frames = video_frame_count(output)
    trim = output_frames > frames
    if not trim and source_audio is None:
        return
    video = ["-c:v", "libx264", "-crf", _TRIM_CRF, "-preset", "medium", "-pix_fmt", "yuv420p"]
    audio = ["-c:a", "copy"]
    inputs = ["-i", str(output)]
    streams = ["-map", "0:v:0", "-map", "0:a?"]
    if source_audio is not None:
        inputs += ["-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, "-i", str(source_audio)]
        streams = ["-map", "0:v:0", "-map", "1:a:0"]
        if source_codec not in _MP4_COPY_AUDIO_CODECS:
            audio = ["-c:a", "aac"]
    conformed = output.with_name(f"{output.stem}-conform{output.suffix}")
    args = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        *inputs,
        *streams,
        *(video if trim else ["-c:v", "copy"]),
        *(["-frames:v", str(frames)] if trim else []),
        *audio,
        "-t",
        ffmpeg_seconds(min(frames, output_frames) / video_fps(output)),
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        str(conformed),
    ]
    try:
        run_media_ffmpeg(args, "output alignment failed")
        os.replace(conformed, output)
    finally:
        conformed.unlink(missing_ok=True)
