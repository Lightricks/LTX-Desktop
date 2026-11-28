"""Tiny media helpers the Pod pipelines need. No desktop ``services.media_probe``."""

from __future__ import annotations

import av


def media_has_audio(path: str) -> bool:
    with av.open(path) as container:
        return any(stream.type == "audio" for stream in container.streams)
