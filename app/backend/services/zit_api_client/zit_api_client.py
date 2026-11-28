"""Z-Image Turbo API client protocol for FAL endpoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from api_types import ApiActivityStatus


class ZitImageBlockedError(RuntimeError):
    """The provider returned no usable image because its safety checker blocked the prompt."""


@dataclass(frozen=True)
class ZitQueuedImage:
    image_bytes: bytes
    # The prompt the provider used. It differs from the sent one when it expanded the prompt.
    prompt: str | None


class ZitAPIClient(Protocol):
    def generate_text_to_image(
        self,
        *,
        api_key: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        """One blocking request. A safety-checker block returns the provider's placeholder image."""
        ...

    def generate_text_to_image_queued(
        self,
        *,
        api_key: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
        enable_prompt_expansion: bool,
        on_status: Callable[[ApiActivityStatus], None],
        on_submitted: Callable[[str], bool],
        wait_cancelled: Callable[[float], bool],
    ) -> ZitQueuedImage:
        """Submit to the provider queue, poll, and fetch the result.

        ``on_submitted(request_id)`` runs right after the provider accepts the request and
        before the first status check. It returns False when the id was not stored. A boot
        cannot then cancel the request, so it is cancelled at once and the call raises.

        ``wait_cancelled(seconds)`` sleeps and returns True once the caller cancelled. The
        request is then cancelled at the provider (best effort) and GenerationCancelledError
        is raised. A safety-checker block raises ZitImageBlockedError.
        """
        ...

    def cancel_queued(self, *, api_key: str, request_id: str) -> None:
        """Ask the provider to drop a queued request. Safe to repeat.

        A request that already finished or is unknown counts as dropped. Any other failure
        raises, so the caller keeps the request id and tries again later.
        """
        ...

    def generate_image_to_image(
        self,
        *,
        api_key: str,
        prompt: str,
        image_bytes: bytes,
        strength: float,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        ...
