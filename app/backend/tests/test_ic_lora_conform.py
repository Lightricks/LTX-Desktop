"""An IC-LoRA recipe keeps every frame and the source soundtrack of the clip it restyles."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
import pytest

from services.features.video.ic_lora_conform import (
    conform_output_to_clip,
    pad_clip_to_frame_grid,
)
from services.media_probe import first_audio_codec, video_frame_count
from tests.video_fixtures import write_pattern_clip

_FPS = 24


def _write_clip(path: Path, *, frames: int, audio: bool = False) -> Path:
    return write_pattern_clip(path, frames=frames, audio=audio, fps=_FPS)


def _decoded_frame_count(path: Path) -> int:
    """Frames a decoder returns. Packets alone can hide a duplicated or a late frame."""
    run = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(re.findall(r"frame= *(\d+)", run.stderr)[-1])


def _packet_sizes(path: Path, kind: str) -> list[int]:
    with av.open(str(path)) as container:
        stream = next(item for item in container.streams if item.type == kind)
        return [packet.size for packet in container.demux(stream) if packet.size > 0]


def _audio_seconds(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = next(item for item in container.streams if item.type == "audio")
        samples = sum(frame.samples for frame in container.decode(stream))
        return samples / stream.codec_context.sample_rate


def _gray_frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [
            frame.to_ndarray(format="gray").astype(np.float32)
            for frame in container.decode(video=0)
        ]


def test_a_clip_is_padded_with_its_last_frame_up_to_the_target(tmp_path: Path) -> None:
    source = _write_clip(tmp_path / "source.mp4", frames=20)

    padded = pad_clip_to_frame_grid(
        source, tmp_path / "grid.mp4", frames=20, target=25, keep_audio=False
    )

    assert padded == tmp_path / "grid.mp4"
    assert video_frame_count(padded) == 25
    assert _decoded_frame_count(padded) == 25
    # Only the last frame repeats. The pad is not a loop or a black tail.
    frames = _gray_frames(padded)
    assert all(abs(frame - frames[19]).max() == 0 for frame in frames[19:])
    assert abs(frames[18] - frames[19]).max() > 0


@pytest.mark.parametrize("target", [20, 17, 9])
def test_a_clip_that_needs_no_pad_is_returned_as_it_is(tmp_path: Path, target: int) -> None:
    source = _write_clip(tmp_path / "source.mp4", frames=20)

    padded = pad_clip_to_frame_grid(
        source, tmp_path / "grid.mp4", frames=20, target=target, keep_audio=False
    )

    assert padded == source
    assert not (tmp_path / "grid.mp4").exists()


def test_a_clip_with_odd_sides_is_padded_instead_of_failing_the_yuv420p_encode(
    tmp_path: Path,
) -> None:
    # An odd side reaches the pad when the crop step returns the original clip.
    source = write_pattern_clip(
        tmp_path / "source.mp4", frames=20, size="63x65", pix_fmt="yuv444p"
    )

    padded = pad_clip_to_frame_grid(
        source, tmp_path / "grid.mp4", frames=20, target=25, keep_audio=False
    )

    assert video_frame_count(padded) == 25
    with av.open(str(padded)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (62, 64)


def test_a_padded_clip_keeps_the_source_audio_when_asked(tmp_path: Path) -> None:
    source = _write_clip(tmp_path / "source.mp4", frames=20, audio=True)

    kept = pad_clip_to_frame_grid(
        source, tmp_path / "kept.mp4", frames=20, target=25, keep_audio=True
    )
    dropped = pad_clip_to_frame_grid(
        source, tmp_path / "dropped.mp4", frames=20, target=25, keep_audio=False
    )

    assert first_audio_codec(kept) == "aac"
    assert first_audio_codec(dropped) is None


def test_the_output_is_cut_back_to_the_clip_frame_count(tmp_path: Path) -> None:
    """The model returns the padded count. A time cut would keep a frame too many."""
    output = _write_clip(tmp_path / "out.mp4", frames=25)

    conform_output_to_clip(output, frames=20, source_audio=None)

    assert video_frame_count(output) == 20
    assert _decoded_frame_count(output) == 20


def test_the_cut_drops_the_frames_at_the_end_of_the_output(tmp_path: Path) -> None:
    output = _write_clip(tmp_path / "out.mp4", frames=25)
    before = _gray_frames(output)

    conform_output_to_clip(output, frames=20, source_audio=None)

    after = _gray_frames(output)
    # Each kept frame is closest to the frame at the same index before the cut.
    for index in (0, 10, 19):
        distances = [abs(after[index] - frame).mean() for frame in before]
        assert distances.index(min(distances)) == index


def test_an_output_on_the_grid_with_no_source_audio_is_left_untouched(tmp_path: Path) -> None:
    output = _write_clip(tmp_path / "out.mp4", frames=17)
    before = output.read_bytes()

    conform_output_to_clip(output, frames=17, source_audio=None)

    assert output.read_bytes() == before
    assert not (tmp_path / "out-conform.mp4").exists()


def test_an_output_with_no_cut_keeps_its_video_stream_when_the_audio_is_replaced(
    tmp_path: Path,
) -> None:
    output = _write_clip(tmp_path / "out.mp4", frames=17)
    video_before = _packet_sizes(output, "video")
    source = _write_clip(tmp_path / "source.mp4", frames=17, audio=True)

    conform_output_to_clip(output, frames=17, source_audio=source)

    assert _packet_sizes(output, "video") == video_before
    assert first_audio_codec(output) == "aac"


def test_a_cut_output_gets_the_source_audio_cut_to_the_video_length(tmp_path: Path) -> None:
    output = _write_clip(tmp_path / "out.mp4", frames=25)
    source = _write_clip(tmp_path / "source.mp4", frames=72, audio=True)
    assert _audio_seconds(source) == pytest.approx(72 / _FPS, abs=0.1)

    conform_output_to_clip(output, frames=20, source_audio=source)

    assert video_frame_count(output) == 20
    assert _audio_seconds(output) == pytest.approx(20 / _FPS, abs=0.1)


def test_an_output_shorter_than_the_clip_is_not_padded_and_the_audio_follows_it(
    tmp_path: Path,
) -> None:
    output = _write_clip(tmp_path / "out.mp4", frames=17)
    source = _write_clip(tmp_path / "source.mp4", frames=72, audio=True)

    conform_output_to_clip(output, frames=20, source_audio=source)

    assert video_frame_count(output) == 17
    assert _audio_seconds(output) == pytest.approx(17 / _FPS, abs=0.1)


def test_an_aac_source_audio_is_copied_into_the_mp4(tmp_path: Path) -> None:
    output = _write_clip(tmp_path / "out.mp4", frames=17)
    source = _write_clip(tmp_path / "source.mp4", frames=17, audio=True)
    source_packets = _packet_sizes(source, "audio")

    conform_output_to_clip(output, frames=17, source_audio=source)

    assert first_audio_codec(output) == "aac"
    # A stream copy keeps the packets. A re-encode would give new sizes.
    assert _packet_sizes(output, "audio")[:5] == source_packets[:5]


def test_a_source_audio_an_mp4_cannot_hold_is_encoded_as_aac(tmp_path: Path) -> None:
    output = _write_clip(tmp_path / "out.mp4", frames=17)
    source = _write_clip(tmp_path / "source.webm", frames=17, audio=True)
    assert first_audio_codec(source) == "opus"

    conform_output_to_clip(output, frames=17, source_audio=source)

    assert first_audio_codec(output) == "aac"
    assert _audio_seconds(output) == pytest.approx(17 / _FPS, abs=0.1)
    assert video_frame_count(output) == 17


def test_a_source_that_has_no_audio_stream_keeps_the_output_audio_and_still_cuts(
    tmp_path: Path,
) -> None:
    """Ingest can record audio for a file that no longer has it. The mux must not fail."""
    output = _write_clip(tmp_path / "out.mp4", frames=25, audio=True)
    silent = _write_clip(tmp_path / "silent.mp4", frames=25)
    assert first_audio_codec(silent) is None

    conform_output_to_clip(output, frames=20, source_audio=silent)

    assert video_frame_count(output) == 20
    assert first_audio_codec(output) == "aac"


def test_a_silent_source_leaves_an_output_on_the_grid_untouched(tmp_path: Path) -> None:
    output = _write_clip(tmp_path / "out.mp4", frames=17, audio=True)
    silent = _write_clip(tmp_path / "silent.mp4", frames=17)
    before = output.read_bytes()

    conform_output_to_clip(output, frames=17, source_audio=silent)

    assert output.read_bytes() == before
