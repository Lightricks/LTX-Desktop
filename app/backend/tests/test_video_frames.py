"""The stills a video-aware prompt rewrite sees: real PyAV decoding of real small clips."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from services.prompt_enhancement.video_frames import (
    MAX_SCENE_FRAMES,
    MIN_SCENE_FRAMES,
    sampled_video_frames,
    scene_frame_count,
)
from services.records import MediaError
from tests.video_fixtures import (
    damage_middle,
    write_mp4,
    write_mp4_with_display_rotation,
    write_mp4_with_timestamp_offset,
    write_noisy_mp4,
)


def test_frames_come_from_across_the_clip_not_only_its_opening(tmp_path: Path) -> None:
    video = write_mp4(tmp_path / "clip.mp4", seconds=6, fps=12)

    with sampled_video_frames(str(video)) as frames:
        times = [frame.time_seconds for frame in frames]

    assert len(times) == scene_frame_count(6)
    assert times == sorted(set(times))
    assert times[0] > 0.3
    assert times[-1] > 5.0


def test_a_one_frame_clip_yields_one_still_not_copies_of_it(tmp_path: Path) -> None:
    video = write_mp4(tmp_path / "tiny.mp4", seconds=0.1, fps=10)

    with sampled_video_frames(str(video)) as frames:
        assert len(frames) == 1


def test_a_sideways_phone_clip_yields_upright_stills(tmp_path: Path) -> None:
    video = write_mp4_with_display_rotation(tmp_path / "phone.mp4", angle="90")

    with sampled_video_frames(str(video)) as frames:
        width, height = Image.open(frames[0].path).size

    # Stored 64x32 (landscape). The container asks for a quarter turn, so it plays portrait.
    assert (width, height) == (32, 64)


def test_still_files_are_deleted_when_the_context_exits(tmp_path: Path) -> None:
    video = write_mp4(tmp_path / "clip.mp4")

    with sampled_video_frames(str(video)) as frames:
        paths = [Path(frame.path) for frame in frames]
        assert all(path.is_file() for path in paths)

    assert not any(path.exists() for path in paths)


def test_still_files_are_deleted_when_the_rewrite_inside_the_context_fails(
    tmp_path: Path,
) -> None:
    video = write_mp4(tmp_path / "clip.mp4")
    paths: list[Path] = []

    with pytest.raises(RuntimeError):
        with sampled_video_frames(str(video)) as frames:
            paths = [Path(frame.path) for frame in frames]
            raise RuntimeError("provider failed")

    assert paths
    assert not any(path.exists() for path in paths)


def test_a_clip_that_starts_at_a_later_timestamp_is_still_sampled_across_its_length(
    tmp_path: Path,
) -> None:
    video = write_mp4_with_timestamp_offset(tmp_path / "cut.mp4", offset_seconds=100)

    with sampled_video_frames(str(video)) as frames:
        times = [frame.time_seconds for frame in frames]

    assert len(times) == scene_frame_count(6)
    assert times == pytest.approx([0.75, 2.25, 3.75, 5.25], abs=0.2)


def test_a_damaged_stretch_never_escapes_as_a_decoder_error(tmp_path: Path) -> None:
    clean = write_noisy_mp4(tmp_path / "clean.mp4")
    sampled = 0

    for seed in range(6):
        damaged = damage_middle(clean, tmp_path / f"damaged-{seed}.mp4", seed=seed)
        try:
            with sampled_video_frames(str(damaged)) as frames:
                assert frames
                sampled += 1
        except MediaError:
            pass  # unreadable is allowed. Any other exception type fails the test.

    assert sampled > 0


def _is_inside(seconds: float, window: tuple[float, float]) -> bool:
    return window[0] < seconds < window[1]


def test_a_still_never_comes_from_inside_the_part_a_retake_replaces(tmp_path: Path) -> None:
    video = write_mp4(tmp_path / "clip.mp4", seconds=10, fps=12)

    with sampled_video_frames(str(video), replaced_window=(4, 6)) as frames:
        times = [frame.time_seconds for frame in frames]

    assert len(times) == scene_frame_count(10)
    assert not any(_is_inside(time, (4, 6)) for time in times)


def test_stills_fall_on_both_sides_of_the_window_and_say_which_side(tmp_path: Path) -> None:
    video = write_mp4(tmp_path / "clip.mp4", seconds=10, fps=12)

    with sampled_video_frames(str(video), replaced_window=(4, 6)) as frames:
        before = [frame for frame in frames if frame.time_seconds <= 4]
        after = [frame for frame in frames if frame.time_seconds >= 6]

    assert before
    assert after
    assert len(before) + len(after) == scene_frame_count(10)
    assert all(frame.note.startswith("before it") for frame in before)
    assert all(frame.note.startswith("after it") for frame in after)
    assert all("4.00s to 6.00s" in frame.note for frame in before + after)


@pytest.mark.parametrize(
    ("window", "side"),
    [((0, 3), "after"), ((8, 10), "before")],
    ids=["window-at-the-start", "window-at-the-end"],
)
def test_a_window_at_the_edge_of_the_clip_still_gives_every_still_from_the_other_side(
    tmp_path: Path, window: tuple[float, float], side: str
) -> None:
    video = write_mp4(tmp_path / "clip.mp4", seconds=10, fps=12)

    with sampled_video_frames(str(video), replaced_window=window) as frames:
        times = [frame.time_seconds for frame in frames]
        notes = [frame.note for frame in frames]

    assert len(times) == scene_frame_count(10)
    assert not any(_is_inside(time, window) for time in times)
    assert all(note.startswith(f"{side} it") for note in notes)


def test_a_window_that_covers_the_whole_clip_still_gives_stills_with_no_side_note(
    tmp_path: Path,
) -> None:
    video = write_mp4(tmp_path / "clip.mp4", seconds=10, fps=12)

    with sampled_video_frames(str(video), replaced_window=(0, 10)) as frames:
        notes = [frame.note for frame in frames]

    assert len(notes) == scene_frame_count(10)
    assert notes == [""] * scene_frame_count(10)


def test_a_window_that_ends_after_the_clip_ends_is_cut_to_the_clip(tmp_path: Path) -> None:
    video = write_mp4(tmp_path / "clip.mp4", seconds=10, fps=12)

    with sampled_video_frames(str(video), replaced_window=(9, 50)) as frames:
        times = [frame.time_seconds for frame in frames]

    assert len(times) == scene_frame_count(10)
    assert max(times) <= 9


@pytest.mark.parametrize(
    "window",
    [(3.4, 5.4), (2.2, 4.1), (1.3, 2.6), (2.6, 4.7)],
    ids=["late-window", "middle-window", "early-window", "off-grid-window"],
)
def test_a_sparse_clip_never_lends_a_replaced_frame_to_the_side_before_the_window(
    tmp_path: Path, window: tuple[float, float]
) -> None:
    """At 2 fps the first frame after a target can sit inside the window."""
    video = write_mp4(tmp_path / "clip.mp4", seconds=6, fps=2)

    with sampled_video_frames(str(video), replaced_window=window) as frames:
        times = [(frame.time_seconds, frame.note) for frame in frames]

    assert times
    assert not any(_is_inside(time, window) for time, _note in times)
    assert all(time < window[0] for time, note in times if note.startswith("before it"))
    assert all(time >= window[1] for time, note in times if note.startswith("after it"))


def test_the_still_count_follows_the_clip_length_within_a_floor_and_a_cap() -> None:
    counts = [scene_frame_count(seconds) for seconds in (0.5, 3, 6, 10, 30, 600)]

    assert counts == sorted(counts)
    assert counts[0] == MIN_SCENE_FRAMES
    assert counts[-1] == MAX_SCENE_FRAMES
    assert counts[2] < counts[3]


def test_a_short_clip_gets_fewer_stills_than_a_long_one(tmp_path: Path) -> None:
    short = write_mp4(tmp_path / "short.mp4", seconds=3, fps=12)
    long = write_mp4(tmp_path / "long.mp4", seconds=30, fps=12)

    with sampled_video_frames(str(short)) as few, sampled_video_frames(str(long)) as many:
        assert (len(few), len(many)) == (scene_frame_count(3), scene_frame_count(30))
    assert len(few) < len(many)
