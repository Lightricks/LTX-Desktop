"""Runtime policy query handler."""

from __future__ import annotations

import os

from api_types import RuntimePolicyResponse
from runtime_config.runtime_config import RuntimeConfig


class RuntimePolicyHandler:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def get_runtime_policy(self) -> RuntimePolicyResponse:
        # force_api_generations stays for leftover API fallback. local_viable is the
        # Explore gate: unsupported machines must not be sent to an LTX API key.
        local_viable = self._config.local_generations_mode != "unsupported"
        if (
            self._config.dev_mode
            and os.environ.get("LTX_DEV_FORCE_LOCAL_VIABLE") == "1"
        ):
            local_viable = True
        return RuntimePolicyResponse(
            force_api_generations=self._config.force_api_generations,
            local_viable=local_viable,
        )
