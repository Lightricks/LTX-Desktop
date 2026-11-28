from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from services.encoded_media import (
    check_output_size,
    derived_file_name,
    ingest_encoded,
)
from services.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, ffmpeg_seconds, run_ffmpeg
from services.media_probe import MAX_AUDIO_BYTES
from services.media_source import require_source
from services.records import AssetRecord, MediaError
from services.store import Store


def trim_audio_file(src: Path, dest: Path, start_sec: float, end_sec: float) -> None:
    run_ffmpeg(
        [
            "-y",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-i",
            str(src),
            "-ss",
            ffmpeg_seconds(start_sec),
            "-t",
            ffmpeg_seconds(end_sec - start_sec),
            "-vn",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-fs",
            str(MAX_AUDIO_BYTES),
            str(dest),
        ]
    )


def extract_audio_file(src: Path, dest: Path) -> None:
    run_ffmpeg(
        [
            "-y",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-i",
            str(src),
            "-vn",
            "-protocol_whitelist",
            FFMPEG_PROTOCOL_WHITELIST,
            "-fs",
            str(MAX_AUDIO_BYTES),
            str(dest),
        ]
    )


def _write_audio_dest(
    tmp_dir: Path, stem_name: str, suffix: str, encode: Callable[[Path], None]
) -> Path:
    dest_m4a = tmp_dir / derived_file_name(stem_name, suffix, ".m4a")
    try:
        encode(dest_m4a)
    except subprocess.CalledProcessError:
        dest_wav = tmp_dir / derived_file_name(stem_name, suffix, ".wav")
        encode(dest_wav)
        check_output_size(dest_wav, MAX_AUDIO_BYTES, "audio")
        return dest_wav
    check_output_size(dest_m4a, MAX_AUDIO_BYTES, "audio")
    return dest_m4a


def _ingest_encoded(
    store: Store,
    record: AssetRecord,
    suffix: str,
    encode: Callable[[Path], None],
    *,
    failure: str,
) -> AssetRecord:
    return ingest_encoded(
        store,
        lambda tmp_dir: _write_audio_dest(tmp_dir, record.name, suffix, encode),
        prefix="ltx-audio-trim-",
        label="audio",
        failure=failure,
    )


def trim_audio_asset(
    store: Store, asset_id: str, start_sec: float, end_sec: float
) -> AssetRecord:
    if end_sec <= start_sec:
        raise ValueError("endSec must be greater than startSec")
    record, source = require_source(store, asset_id, "audio")

    def encode(dest: Path) -> None:
        trim_audio_file(source, dest, start_sec, end_sec)

    return _ingest_encoded(
        store, record, "trim", encode, failure="ffmpeg failed to trim audio"
    )


def extract_audio_asset(store: Store, asset_id: str) -> AssetRecord:
    record, source = require_source(store, asset_id, "video")
    metadata = record.metadata
    if metadata.mediaType != "video":
        raise MediaError(
            "UNSUPPORTED_MEDIA",
            f"expected video asset, got {record.media_kind}",
        )
    # ffmpeg turns a silent picture into "Output file does not contain any stream".
    if metadata.metadata.audioStreamCount < 1:
        raise MediaError("NO_AUDIO_STREAM", "video has no audio track")

    def encode(dest: Path) -> None:
        extract_audio_file(source, dest)

    return _ingest_encoded(
        store, record, "audio", encode, failure="ffmpeg failed to extract audio"
    )
