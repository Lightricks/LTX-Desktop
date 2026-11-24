from pydantic import BaseModel, ConfigDict


class AssistantLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    attachments: int = 8
    jobs: int = 12
    model_requests: int = 16
    tool_calls: int = 24
    shown_assets: int = 30
    discovery_assets: int = 10
    local_history_bytes: int = 32000
    cloud_history_bytes: int = 128000
    local_timeout_seconds: int = 180
    cloud_timeout_seconds: int = 60
    output_tokens: int = 2048


ASSISTANT_LIMITS = AssistantLimits()
