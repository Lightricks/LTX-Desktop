"""Failures the pipeline can raise without importing desktop ``services.records``."""

from __future__ import annotations

VIDEO_JOB_TOO_LARGE = "VIDEO_JOB_TOO_LARGE"


class CapabilityFailedError(Exception):
    """Executor could not run this generation with the current local capability."""

    def __init__(self, detail: str, *, code: str | None = None) -> None:
        self.detail = detail
        self.code = code
        super().__init__(detail)
