import asyncio
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager, nullcontext

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from services.assistant_models import AssistantModelError
from services.llama_runtime import LlamaRuntime, LocalEndpoint


class FakeLlamaRuntime(LlamaRuntime):
    def __init__(self, directory, base_url="http://127.0.0.1:1/v1"):
        super().__init__(directory, system="Darwin", machine="arm64")
        self.base_url = base_url
        self.active = False
        self.install_called = False

    def installed(self):
        return self.install_called

    def install(self, on_progress):
        self.install_called = True
        on_progress(self.download_size)

    @asynccontextmanager
    async def serve(self, model, path, *, projector=None):
        self.active = True
        try:
            yield LocalEndpoint(self.base_url, "test-local-key")
        finally:
            self.active = False


class FakeAssistantModels:
    def projector_path(self, model):
        return self.path(model) / "mmproj.gguf"

    def path(self, model):
        return Path("/fake/assistant-models") / model

    def __init__(self):
        self.models = {"qwen3.5:9b": 6_600_000_000}
        self.available = True
        self.pull_error = None
        self.during_pull = None

    def installed(self):
        if not self.available:
            raise AssistantModelError("Local assistant is not supported on this platform.")
        return dict(self.models)

    def delete(self, model):
        self.models.pop(model, None)

    def pull(self, model, on_progress):
        on_progress(50, 100)
        if self.during_pull:
            self.during_pull()
        if self.pull_error:
            raise AssistantModelError(self.pull_error)
        self.models[model] = 100
        on_progress(100, 100)


@dataclass
class InterruptedTextStream:
    text: str = ""


class FakeAssistantProvider:
    """Script the model boundary while exercising the real Pydantic AI tool loop."""

    def __init__(self):
        self.responses = deque()
        # The instructions of each model call, as the model received them.
        self.instructions = []

    def create_model(self, *, provider, api_key, model):
        return nullcontext(FunctionModel(self.respond, stream_function=self.stream))

    def respond(self, messages, info):
        self.instructions.append(info.instructions or "")
        if not self.responses:
            return ModelResponse(parts=[TextPart("No further action.")])
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response()
        return response

    async def stream(self, messages, info):
        response = await asyncio.to_thread(self.respond, messages, info)
        if isinstance(response, InterruptedTextStream):
            yield response.text
            raise RuntimeError("Assistant stream disconnected")
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {index: DeltaToolCall(part.tool_name, part.args_as_json_str(), tool_call_id=part.tool_call_id)}
            else:
                raise TypeError(f"Unsupported scripted part: {part!r}")
