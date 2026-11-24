"""NAG negative prompt support for ``DistilledPipeline``.

``NagDistilledMixin`` goes before ``DistilledPipeline`` in the base list. It:

1. encodes the negative prompt in the same Gemma pass as the prompt (one load),
2. appends the negative embedding to the video text context along the sequence axis,
3. turns NAG on only while the denoise calls run (``nag_scope``).

The attention change is in ``services/patches/nag_cross_attention.py``. A job
with no ``NagConfig`` takes none of these steps.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Generic, Protocol, TypeVar, cast

import torch
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput

from services.fast_video_pipeline.nag_config import NagConfig
from services.patches.nag_cross_attention import NagState, nag_scope

_T = TypeVar("_T")
_RunCtx = TypeVar("_RunCtx")


class _PromptEncoderLike(Protocol):
    def __call__(self, prompts: list[str], **kwargs: Any) -> list[EmbeddingsProcessorOutput]: ...


def prompts_with_negative(prompt: str, nag: NagConfig | None) -> list[str]:
    """Prompts to pass to ``prompt_encoder`` when NAG is active vs. off."""
    return [prompt, nag.negative_prompt] if nag is not None else [prompt]


def with_negative_context(
    positive: torch.Tensor, negative: torch.Tensor, nag: NagConfig
) -> tuple[torch.Tensor, NagState]:
    """Concatenate positive and negative video text contexts and build ``NagState``."""
    return (
        torch.cat([positive, negative], dim=1),
        NagState(config=nag, positive_length=positive.shape[1]),
    )


class _NegativeCapture:
    """Adds the negative prompt to the encode call and keeps its embedding.

    The caller sees one output per prompt it asked for. The enhancer only touches
    the first prompt, so the negative prompt is never rewritten.
    """

    def __init__(self, inner: _PromptEncoderLike, negative_prompt: str) -> None:
        self._inner = inner
        self._negative_prompt = negative_prompt
        self.negative: EmbeddingsProcessorOutput | None = None

    def __call__(self, prompts: list[str], **kwargs: Any) -> list[EmbeddingsProcessorOutput]:
        outputs = self._inner([*prompts, self._negative_prompt], **kwargs)
        self.negative = outputs[-1]
        return outputs[:-1]


def scoped_chunks(chunks: Iterator[_T], state: NagState) -> Iterator[_T]:
    """Yield from a lazy chunk stream with NAG on only while a chunk is being computed.

    Denoising runs inside ``next()``. The scope must not stay open while the caller
    holds a chunk, because the caller can start another job in between.
    """
    iterator = iter(chunks)
    while True:
        with nag_scope(state):
            try:
                chunk = next(iterator)
            except StopIteration:
                return
        yield chunk


class NagDistilledMixin(Generic[_RunCtx]):
    prompt_encoder: Any  # set by DistilledPipeline.__init__
    _nag_pending: NagConfig | None = None
    _nag_state: NagState | None = None

    @contextmanager
    def _nag_request(self, config: NagConfig | None) -> Iterator[None]:
        previous = self._nag_pending
        self._nag_pending = config
        try:
            yield
        finally:
            self._nag_pending = previous

    def _nag_denoise_scope(self) -> AbstractContextManager[None]:
        """Scope for a denoise call made directly (not through ``stream_chunks``)."""
        state, self._nag_state = self._nag_state, None
        return nag_scope(state) if state is not None else nullcontext()

    def _prepare_run(self, **kwargs: Any) -> _RunCtx:
        config = self._nag_pending
        parent = cast(Any, super())
        self._nag_state = None
        if config is None:
            return cast("_RunCtx", parent._prepare_run(**kwargs))
        encoder = self.prompt_encoder
        capture = _NegativeCapture(encoder, config.negative_prompt)
        self.prompt_encoder = capture
        try:
            ctx = parent._prepare_run(**kwargs)
        finally:
            self.prompt_encoder = encoder
        negative = capture.negative
        if negative is None:
            raise RuntimeError("The negative prompt was not encoded")
        video_context, self._nag_state = with_negative_context(
            ctx.video_context, negative.video_encoding, config
        )
        return cast("_RunCtx", dataclasses.replace(ctx, video_context=video_context))

    def stream_chunks(self, *args: Any, nag: NagConfig | None = None, **kwargs: Any) -> Any:
        parent = cast(Any, super())
        config = nag if nag is not None else self._nag_pending
        with self._nag_request(config):
            chunks, num_frames, tiling_config = parent.stream_chunks(*args, **kwargs)
        state, self._nag_state = self._nag_state, None
        if state is not None:
            chunks = scoped_chunks(chunks, state)
        return chunks, num_frames, tiling_config
