"""Reusable device buffers for one tiled DiffVAE decode.
Per-tile allocation of the large stage-5 buffers leaves the caching allocator 25-45 GiB above what is ever live;
:class:`DecodeWorkspace` keeps them for the whole decode as activation volumes leased per tile, uninitialised like
``torch.empty``; without a workspace every helper is a plain one.
"""

from __future__ import annotations

import contextvars
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

__all__ = [
    "DecodeWorkspace",
    "active_decode_workspace",
    "decode_workspace",
    "workspace_phase_empty",
    "workspace_phase_reset",
    "workspace_volume",
]

# Phase tensors start on a multiple of this many elements (keeps every carved buffer aligned).
_PHASE_ALIGN_ELEMS = 128

_ACTIVE: contextvars.ContextVar[DecodeWorkspace | None] = contextvars.ContextVar(
    "ltx_kernels_decode_workspace", default=None
)


@dataclass
class _VolumeSlot:
    buffer: torch.Tensor  # flat, one dtype / device
    leased: bool = False
    phase_offset: int = 0  # elements handed out by phase_take since the last phase_reset

    @property
    def capacity(self) -> int:
        return self.buffer.numel()


class DecodeWorkspace:
    """Per-tile leased volumes and the phase arena carved from the idle ones; see the module docstring."""

    def __init__(self, large_volume_numel: int = 0) -> None:
        """``large_volume_numel``: capacity of every large slot, the decode's biggest volume, so later tiles reuse it.
        A request counts as large from half that size; smaller ones (keyframe planes) get exactly what they ask for.
        """
        self._volumes: list[_VolumeSlot] = []
        self._large_numel = max(0, int(large_volume_numel))

    # -- leased volumes ------------------------------------------------------------------
    def _is_large(self, numel: int) -> bool:
        return bool(self._large_numel) and numel * 2 >= self._large_numel

    def volume(self, numel: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Lease the smallest free buffer that holds ``numel`` elements, or allocate a new one.
        Large requests use large-capacity slots; small ones never lease a large slot, so they cannot displace one.
        """
        large = self._is_large(numel)
        best: _VolumeSlot | None = None
        for slot in self._volumes:
            buf = slot.buffer
            if slot.leased or buf.dtype != dtype or buf.device != device or slot.capacity < numel:
                continue
            if slot.phase_offset:
                continue
            if self._large_numel and (slot.capacity >= self._large_numel) != large:
                continue
            if best is None or slot.capacity < best.capacity:
                best = slot
        if best is None:
            capacity = max(numel, 1)
            if large:
                capacity = max(capacity, self._large_numel)
            best = _VolumeSlot(torch.empty(capacity, dtype=dtype, device=device))
            self._volumes.append(best)
        best.leased = True
        return best.buffer[:numel]

    def release_tile(self) -> None:
        """End of a tile: every leased volume is free for the next one (nothing is deallocated)."""
        for slot in self._volumes:
            slot.leased = False
        self.phase_reset()

    # -- phase arena: short-lived tensors carved out of the idle volume slots --------------
    def phase_take(
        self,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        not_aliasing: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor | None:
        """Bump-allocate ``numel`` elements from a free (unleased) volume slot, or ``None`` if none has room.
        For a phase whose tensors all die before the volumes are next leased; :meth:`phase_reset` frees them all.
        ``not_aliasing``: slots backing these tensors are skipped (a custom op's output may not alias its inputs).
        """
        avoid = {t.untyped_storage().data_ptr() for t in not_aliasing}
        for slot in self._volumes:
            buf = slot.buffer
            if slot.leased or buf.dtype != dtype or buf.device != device:
                continue
            if avoid and buf.untyped_storage().data_ptr() in avoid:
                continue
            offset = slot.phase_offset
            if offset + numel > slot.capacity:
                continue
            slot.phase_offset = offset + (-(-numel // _PHASE_ALIGN_ELEMS)) * _PHASE_ALIGN_ELEMS
            return buf[offset : offset + numel]
        return None

    def phase_reset(self) -> None:
        for slot in self._volumes:
            slot.phase_offset = 0


def active_decode_workspace() -> DecodeWorkspace | None:
    return _ACTIVE.get()


@contextmanager
def decode_workspace(workspace: DecodeWorkspace | None) -> Iterator[DecodeWorkspace | None]:
    """Make ``workspace`` the one :func:`workspace_volume` / :func:`workspace_phase_empty` draw from."""
    token = _ACTIVE.set(workspace)
    try:
        yield workspace
    finally:
        _ACTIVE.reset(token)


def workspace_phase_empty(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    not_aliasing: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """``torch.empty(shape)`` carved from the active workspace's idle volumes (see
    :meth:`DecodeWorkspace.phase_take`); a plain ``empty`` without a workspace or when nothing fits."""
    ws = _ACTIVE.get()
    if ws is not None:
        flat = ws.phase_take(math.prod(shape), dtype=dtype, device=device, not_aliasing=not_aliasing)
        if flat is not None:
            return flat.view(shape)
    return torch.empty(shape, dtype=dtype, device=device)


def workspace_phase_reset() -> None:
    ws = _ACTIVE.get()
    if ws is not None:
        ws.phase_reset()


def workspace_volume(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``torch.empty(shape)`` leased from the active workspace until its tile ends (plain ``empty`` without one)."""
    ws = _ACTIVE.get()
    if ws is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return ws.volume(math.prod(shape), dtype=dtype, device=device).view(shape)
