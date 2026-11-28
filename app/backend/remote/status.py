"""Status payload advertised to Desktop Settings."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from state.app_settings import RemoteExposure


class RemoteStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permitted: bool
    serving: bool
    url: str | None = None
    localUrl: str | None = None
    lanUrl: str | None = None
    reason: str | None = None
    mode: RemoteExposure | None = None
