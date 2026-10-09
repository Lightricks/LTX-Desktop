"""Shared lavfi silence for retake/extend pixel stitch when the source has no audio."""

SILENCE_INPUT = "anullsrc=channel_layout=stereo:sample_rate=48000"


def append_silence_input(args: list[str]) -> None:
    """Add silence as the last input, then end the output with the video.

    ``anullsrc`` never ends. A stitch that keeps its tail (``atrim=start=...``) would
    otherwise encode silence forever and ffmpeg would only stop at the timeout, so the
    output is cut when the shortest stream (the finite video) ends. ``-shortest`` is an
    output option: it must come after every ``-i``, which is why it is added here, after
    the last input.
    """
    args.extend(["-f", "lavfi", "-i", SILENCE_INPUT, "-shortest"])


def original_audio_ref(silence_original: bool) -> str:
    return "[2:a]" if silence_original else "[0:a]"
