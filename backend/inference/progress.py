"""Exact denoise-step totals and in-flight step reporting.

Drive the existing progress bar from real denoise steps instead of a guessed
duration. Step-based inference progress was proposed by Sergio Gil
(@jimeneztion).

``ltx_pipelines`` sampler loops have no progress callback. Totals are derived
from the same sigma schedules the pipelines pass to ``DiffusionStage``
(``len(sigmas) - 1`` per stage). Current step is advanced from the existing
``DiffusionStage.__call__`` patch as each loop iteration completes.

Inference percent occupies 15–90 so loading/encoding stay below and decode/encode
can finish the bar.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

_INFERENCE_START = 15
_INFERENCE_SPAN = 75
_INFERENCE_END = 90

# Current ltx-pipelines distilled recipe (len(sigmas) - 1 per stage). Tests pin
# these so a schedule change fails loudly instead of silently stretching the bar.
DISTILLED_STAGE_1_STEPS = len(DISTILLED_SIGMA_VALUES) - 1
DISTILLED_STAGE_2_STEPS = len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1
DISTILLED_TOTAL_STEPS = DISTILLED_STAGE_1_STEPS + DISTILLED_STAGE_2_STEPS

_Sink = Callable[[int, int], None]


@dataclass
class _Session:
    total: int
    current: int = 0


_session: ContextVar[_Session | None] = ContextVar("denoising_progress_session", default=None)
_sink: _Sink | None = None


def configure_sink(sink: _Sink | None) -> None:
    global _sink
    _sink = sink


def distilled_total_steps(*, stage_2: bool = True) -> int:
    """Denoise iterations for a distilled run (Fast / A2V / IC-LoRA).

    Matches ``DiffusionStage``: ``len(sigmas) - 1`` per stage. Stage 2 is the
    2× upsample refinement; IC-LoRA ``skip_stage_2`` omits it.
    """
    if stage_2:
        return DISTILLED_TOTAL_STEPS
    return DISTILLED_STAGE_1_STEPS


def inference_percent(current: int, total: int) -> int:
    if total <= 0:
        return _INFERENCE_START
    return min(_INFERENCE_START + int(_INFERENCE_SPAN * current / total), _INFERENCE_END)


def _emit(current: int, total: int) -> None:
    if _sink is not None:
        _sink(current, total)


@contextmanager
def track_denoising(total_steps: int) -> Iterator[None]:
    token: Token[_Session | None] = _session.set(_Session(total=total_steps, current=0))
    _emit(0, total_steps)
    try:
        yield
    finally:
        _session.reset(token)


def note_step() -> None:
    session = _session.get()
    if session is None or session.total <= 0:
        return
    session.current = min(session.current + 1, session.total)
    _emit(session.current, session.total)


def reset() -> None:
    """Drop an in-flight session and the progress sink (tests)."""
    global _sink
    _session.set(None)
    _sink = None
