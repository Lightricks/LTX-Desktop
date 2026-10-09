"""Unit tests for shared video frame-count calculations."""

from frame_math import effective_a2v_audio_seconds, num_frames_for_audio_duration


def test_audio_duration_10_563_seconds_floors_to_249_frames_under_cap() -> None:
    assert num_frames_for_audio_duration(10.563, 24, max_frames=481) == 249


def test_audio_duration_10_563_seconds_reports_over_cap() -> None:
    assert num_frames_for_audio_duration(10.563, 24, max_frames=241) is None


def test_audio_duration_5_seconds_floors_to_113_frames() -> None:
    assert num_frames_for_audio_duration(5.0, 24, max_frames=481) == 113


def test_20s_audio_at_a_10s_cell_keeps_the_opening_10s() -> None:
    assert (
        effective_a2v_audio_seconds(
            20.0,
            cell_max_seconds=10,
            longest_cell_seconds=20,
        )
        == 10.0
    )


def test_audio_past_every_cell_stays_rejectable() -> None:
    assert (
        effective_a2v_audio_seconds(
            20.1,
            cell_max_seconds=10,
            longest_cell_seconds=20,
        )
        is None
    )


def test_audio_inside_the_cell_is_unchanged() -> None:
    assert (
        effective_a2v_audio_seconds(
            10.563,
            cell_max_seconds=20,
            longest_cell_seconds=20,
        )
        == 10.563
    )


def test_audio_duration_2_seconds_is_minimum_aligned_grid() -> None:
    frames = num_frames_for_audio_duration(2.0, 24, max_frames=481)

    assert frames is not None
    assert frames >= 9
    assert (frames - 1) % 8 == 0
