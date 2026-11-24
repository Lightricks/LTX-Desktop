"""Model transport boundary; orchestration and typed tools live in the agent."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from google.genai import Client
from google.genai.types import HttpOptions, HttpRetryOptions

import httpx2
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from services.llama_runtime import LlamaRuntime
from services.assistant_models import assistant_model_spec

if TYPE_CHECKING:
    from state.app_settings import AssistantProviderName

# Rides out short Gemini overload spikes. Retries only failed requests, so no tool runs twice.
# Timeouts are also retried by the client, which is why attempts and delays stay small.
_GEMINI_RETRY = HttpRetryOptions(attempts=3, initial_delay=1, max_delay=4, http_status_codes=[429, 500, 502, 503])


class AssistantProvider(Protocol):
    def create_model(self, *, provider: AssistantProviderName, api_key: str, model: str) -> AbstractAsyncContextManager[Model]: ...


class DesktopAssistantProvider:
    def __init__(
        self, runtime: LlamaRuntime, model_path: Callable[[str], Path], *,
        projector_path: Callable[[str], Path],
        reserve_generation: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> None:
        self._runtime, self._model_path = runtime, model_path
        self._projector_path, self._reserve_generation = projector_path, reserve_generation

    @asynccontextmanager
    async def create_model(self, *, provider: AssistantProviderName, api_key: str, model: str) -> AsyncIterator[Model]:
        if provider == "local":
            # Models remain resident for the tool loop, then yield the GPU to video.
            async with AsyncExitStack() as reservation:
                if self._reserve_generation:
                    await reservation.enter_async_context(self._reserve_generation())
                assistant_model_spec(model)
                async with self._runtime.serve(model, self._model_path(model), projector=self._projector_path(model)) as endpoint:
                    async with httpx2.AsyncClient(trust_env=False) as http:
                        async with OpenAIProvider(base_url=endpoint.base_url, api_key=endpoint.api_key, http_client=http) as local_provider:
                            yield OpenAIChatModel(model, provider=local_provider)
            return
        # Direct to Google: no gateway, Logfire configuration, or file uploads.
        with Client(api_key=api_key, http_options=HttpOptions(retry_options=_GEMINI_RETRY)) as client:
            yield GoogleModel(model, provider=GoogleProvider(client=client))
