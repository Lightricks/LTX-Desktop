"""Read streamed transformer blocks with ``os.preadv`` on Darwin.

``DiskBlockReader.read_into`` runs on the ``BlockFetcher`` I/O thread. It calls
``safetensors`` ``get_tensor`` for each tensor of the next block and copies the
result into a CPU slot. ``get_tensor`` holds the GIL while it reads. The main
thread must have the GIL to queue each MPS operation, so it waits between
operations while a block loads. Stage 1 queues many small operations per block
and loses the most time.

``os.preadv`` releases the GIL during the read. This patch reads each tensor
directly from the file into its slot, with no temporary tensor. The bytes are
the same as the ``get_tensor`` bytes, so the output video does not change.

A tensor uses the direct read only when all of these are true:

- No ``SDKeyValueOperation`` in the reader's ``sd_ops`` matches its key. Such an
  operation can change the value or the key. With no match,
  ``apply_to_key_value`` returns the tensor unchanged.
- The slot is a contiguous CPU tensor with the dtype and shape from the file.

All other tensors use the original ``get_tensor`` path.

Darwin only. The I/O thread on CUDA hosts copies into pinned memory and was not
measured.

Remove once ltx-core ``DiskBlockReader`` reads without holding the GIL.

Usage:
    import services.patches.disk_streaming_direct_read as disk_streaming_direct_read
    disk_streaming_direct_read.install()
"""

from __future__ import annotations

import json
import logging
import os
import struct
import sys
from typing import Any, NamedTuple

import torch

logger = logging.getLogger(__name__)

_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}

_installed = False
_checked = False
_original_reader_init: Any = None
_original_reader_close: Any = None
_original_read_into: Any = None


class _FileSpan(NamedTuple):
    fd: int
    offset: int
    nbytes: int
    dtype: torch.dtype | None
    shape: tuple[int, ...]


def _index_file(path: str) -> tuple[int, dict[str, _FileSpan]]:
    """Open *path* and return its fd and the byte span of each tensor."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header: dict[str, Any] = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    data_start = 8 + header_len
    fd = os.open(path, os.O_RDONLY)
    spans: dict[str, _FileSpan] = {}
    for key, meta in header.items():
        start, end = meta["data_offsets"]
        spans[key] = _FileSpan(fd, data_start + start, end - start, _DTYPES.get(meta["dtype"]), tuple(meta["shape"]))
    return fd, spans


def _read_span(span: _FileSpan, target: torch.Tensor) -> None:
    view = memoryview(target.view(-1).view(torch.uint8).numpy())
    done = 0
    while done < span.nbytes:
        got = os.preadv(span.fd, [view[done:]], span.offset + done)
        if got <= 0:
            raise OSError(f"short read: {done} of {span.nbytes} bytes at offset {span.offset}")
        done += got


def _can_read_direct(span: _FileSpan | None, target: torch.Tensor | None) -> bool:
    return (
        span is not None
        and target is not None
        and target.device.type == "cpu"
        and target.is_contiguous()
        and target.dtype == span.dtype
        and tuple(target.shape) == span.shape
        and target.numel() * target.element_size() == span.nbytes
    )


def install() -> None:
    """Read streamed blocks with ``os.preadv`` on Darwin. Idempotent.

    No-op on every other platform. Does not log. Call ``log_status`` once
    logging exists.
    """
    global _installed, _checked, _original_reader_init, _original_reader_close, _original_read_into
    if _installed:
        return
    _checked = True
    if sys.platform != "darwin":
        return

    from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader  # noqa: PLC0415
    from ltx_core.block_streaming.utils import make_block_key  # noqa: PLC0415
    from ltx_core.loader.sd_ops import SDKeyValueOperation  # noqa: PLC0415

    _original_reader_init = DiskTensorReader.__init__
    _original_reader_close = DiskTensorReader.close
    _original_read_into = DiskBlockReader.read_into

    def _reader_init(self: DiskTensorReader, paths: list[str]) -> None:
        _original_reader_init(self, paths)
        fds: list[int] = []
        spans: dict[str, _FileSpan] = {}
        try:
            for path in paths:
                fd, file_spans = _index_file(path)
                fds.append(fd)
                spans.update(file_spans)
        except Exception:
            for fd in fds:
                os.close(fd)
            raise
        self._direct_fds = fds  # type: ignore[attr-defined]
        self._direct_spans = spans  # type: ignore[attr-defined]

    def _reader_close(self: DiskTensorReader) -> None:
        for fd in getattr(self, "_direct_fds", []):
            os.close(fd)
        self._direct_fds = []  # type: ignore[attr-defined]
        self._direct_spans = {}  # type: ignore[attr-defined]
        _original_reader_close(self)

    def _value_op_matches(reader: DiskBlockReader, full_key: str) -> bool:
        sd_ops = reader._sd_ops  # pyright: ignore[reportPrivateUsage]
        if sd_ops is None:
            return False
        return any(
            isinstance(op, SDKeyValueOperation)
            and full_key.startswith(op.key_matcher.prefix)
            and full_key.endswith(op.key_matcher.suffix)
            for op in sd_ops.mapping
        )

    def _read_into(self: DiskBlockReader, target: dict[str, torch.Tensor], block_idx: int) -> None:
        spans: dict[str, _FileSpan] = getattr(self._reader, "_direct_spans", {})  # pyright: ignore[reportPrivateUsage]
        blocks_prefix: str = self._blocks_prefix  # pyright: ignore[reportPrivateUsage]
        rest: list[tuple[str, str]] = []
        for sft_key, param_name in self._block_key_map[block_idx]:  # pyright: ignore[reportPrivateUsage]
            span = spans.get(sft_key)
            if (
                _can_read_direct(span, target.get(param_name))
                and not _value_op_matches(self, make_block_key(blocks_prefix, block_idx, param_name))
            ):
                assert span is not None
                _read_span(span, target[param_name])
            else:
                rest.append((sft_key, param_name))
        if rest:
            # The original read_into takes its keys from the reader's block key map.
            # A throwaway reader with only the fallback keys leaves this reader's map untouched.
            fallback = DiskBlockReader(self._reader, {block_idx: rest}, self._sd_ops, blocks_prefix)  # pyright: ignore[reportPrivateUsage]
            _original_read_into(fallback, target, block_idx)

    DiskTensorReader.__init__ = _reader_init  # type: ignore[method-assign]
    DiskTensorReader.close = _reader_close  # type: ignore[method-assign]
    DiskBlockReader.read_into = _read_into  # type: ignore[method-assign]
    _installed = True


def log_status() -> None:
    """Report the block read path. Call after logging is configured."""
    if not _checked:
        return
    if _installed:
        logger.info("Disk streaming reads blocks with os.preadv (GIL released)")
    else:
        logger.info("Disk streaming reads blocks with safetensors get_tensor; preadv is Darwin-only")
