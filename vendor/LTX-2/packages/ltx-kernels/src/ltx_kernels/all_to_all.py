"""All2All communication primitives for distributed attention.
The C++ kernels are exposed via ``torch.library.custom_op`` so that
``torch.compile`` (and CUDA Graph capture under ``mode="reduce-overhead"``)
can trace through them without a graph break. Each :class:`All2All`
instance registers itself in a class-level registry indexed by an integer
``comm_id``; the custom op takes that id, an input tensor, and the instance's
IPC buffer, and dispatches to the appropriate C++ runtime.
"""

import math
import weakref
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from all2all_cpp import All2All as All2AllCpp
from torch.library import custom_op

# Output shapes and kernel offsets are derived from the input tensor under uniform
# sharding. Callers pad before sharding, so every rank has the same token count.
# Each op writes its result into the instance's local IPC buffer, which it takes as a
# mutated argument, and returns nothing; All2All returns a view of that buffer. Returning
# the view from the op instead would declare storage the compiler may reuse for other
# tensors, while the next op on the same instance, from any rank, overwrites it.


@custom_op("ltx_kernels::send_recv_heads", mutates_args=("buffer",), device_types="cuda")
def _send_recv_heads_op(x: torch.Tensor, buffer: torch.Tensor, comm_id: int) -> None:  # noqa: ARG001
    All2All._runtime_registry[comm_id].send_recv_heads(x, False)


@_send_recv_heads_op.register_fake
def _send_recv_heads_fake(x: torch.Tensor, buffer: torch.Tensor, comm_id: int) -> None:  # noqa: ARG001
    return None


@custom_op("ltx_kernels::gather_heads", mutates_args=("buffer",), device_types="cuda")
def _gather_heads_op(x: torch.Tensor, buffer: torch.Tensor, comm_id: int) -> None:  # noqa: ARG001
    All2All._runtime_registry[comm_id].gather_heads(x, False)


@_gather_heads_op.register_fake
def _gather_heads_fake(x: torch.Tensor, buffer: torch.Tensor, comm_id: int) -> None:  # noqa: ARG001
    return None


@custom_op("ltx_kernels::rope_send_recv_heads", mutates_args=("buffer",), device_types="cuda")
def _rope_send_recv_heads_op(
    x: torch.Tensor,
    weights: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    eps: float,
    buffer: torch.Tensor,  # noqa: ARG001
    comm_id: int,
    barrier: bool = True,
) -> None:
    All2All._runtime_registry[comm_id].rope_send_recv_heads(x, sin_freqs, cos_freqs, weights, eps, barrier)


@_rope_send_recv_heads_op.register_fake
def _rope_send_recv_heads_fake(
    x: torch.Tensor,  # noqa: ARG001
    weights: torch.Tensor,  # noqa: ARG001
    cos_freqs: torch.Tensor,  # noqa: ARG001
    sin_freqs: torch.Tensor,  # noqa: ARG001
    eps: float,  # noqa: ARG001
    buffer: torch.Tensor,  # noqa: ARG001
    comm_id: int,  # noqa: ARG001
    barrier: bool = True,  # noqa: ARG001
) -> None:
    return None


@custom_op("ltx_kernels::linear_send_recv_heads", mutates_args=("buffer",), device_types="cuda")
def _linear_send_recv_heads_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    buffer: torch.Tensor,  # noqa: ARG001
    comm_id: int,
    barrier: bool = True,
) -> None:
    All2All._runtime_registry[comm_id].linear_send_recv_heads(x, weight, bias, barrier)


@_linear_send_recv_heads_op.register_fake
def _linear_send_recv_heads_fake(
    x: torch.Tensor,  # noqa: ARG001
    weight: torch.Tensor,  # noqa: ARG001
    bias: torch.Tensor,  # noqa: ARG001
    buffer: torch.Tensor,  # noqa: ARG001
    comm_id: int,  # noqa: ARG001
    barrier: bool = True,  # noqa: ARG001
) -> None:
    return None


@custom_op("ltx_kernels::exchange_barrier", mutates_args=("buffers",), device_types="cuda")
def _exchange_barrier_op(buffers: list[torch.Tensor], comm_id: int) -> None:  # noqa: ARG001
    # ``buffers`` are the exchanges this barrier completes: declaring them mutated orders every read of
    # them after it.
    All2All._runtime_registry[comm_id].exchange_barrier()


@_exchange_barrier_op.register_fake
def _exchange_barrier_fake(buffers: list[torch.Tensor], comm_id: int) -> None:  # noqa: ARG001
    return None


class All2All:
    """IPC-based All2All communication for distributed head-parallel attention.
    This class manages GPU memory buffers and IPC handles to enable efficient
    cross-GPU communication for attention head redistribution.
    Args:
        rank: Local rank of this process.
        world_size: Total number of processes in the distributed group.
        seqlen: Maximum sequence length to allocate buffers for.
        hidden_dim: Hidden dimension size (num_heads * head_dim).
        num_sms: Number of SMs to use for kernel execution.
        tensor_dtype: Data type for tensors (e.g., torch.bfloat16).
        group: PyTorch distributed process group.
    """

    _next_id: ClassVar[int] = 0
    _runtime_registry: ClassVar[dict[int, "All2AllCpp"]] = {}

    @staticmethod
    def _release_comm(comm_id: int, runtime: "All2AllCpp") -> None:
        All2All._runtime_registry.pop(comm_id, None)
        runtime.destroy()

    def __init__(
        self,
        rank: int,
        world_size: int,
        seqlen: int,
        hidden_dim: int,
        num_sms: int,
        tensor_dtype: torch.dtype,
        group: torch.distributed.ProcessGroup | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.num_sms = num_sms
        self.tensor_dtype = tensor_dtype
        self.buffer_size = int(seqlen * hidden_dim * tensor_dtype.itemsize)

        # Initialize the C++ runtime. Omit timeout_seconds to keep the kernel's default
        # (DEFAULT_BARRIER_TIMEOUT_SECONDS); it can still be changed later via set_timeout_seconds.
        if timeout_seconds is None:
            self.runtime = All2AllCpp(rank, world_size, seqlen, hidden_dim, num_sms, tensor_dtype)
        else:
            if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
                raise ValueError(f"all2all timeout seconds must be finite and non-negative, got {timeout_seconds}")
            self.runtime = All2AllCpp(rank, world_size, seqlen, hidden_dim, num_sms, tensor_dtype, timeout_seconds)

        # Register in the class-level table so the custom ops can find us.
        # Auto-cleanup via weakref.finalize covers the case where destroy()
        # isn't called explicitly — without it the registry would keep the
        # runtime (and CUDA/IPC resources) alive for the process lifetime.
        self._comm_id = All2All._next_id
        All2All._next_id += 1
        All2All._runtime_registry[self._comm_id] = self.runtime
        self._finalizer = weakref.finalize(self, All2All._release_comm, self._comm_id, self.runtime)

        # Exchange IPC handles across all ranks
        ipc_handles: list[Any] = [None] * world_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        self.runtime.sync(ipc_handles)
        self._buffer = self.runtime.local_buffer()

    def set_timeout_seconds(self, seconds: float) -> None:
        """Set the all2all barrier (deadlock-detection) timeout, in seconds.
        The C++ runtime converts to clock cycles using the device's peak SM clock. Raise it
        during the first ``torch.compile`` forward, where one rank's recompile can delay its
        all2all launch past the steady-state timeout; reset for steady state.
        """
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"all2all timeout seconds must be finite and non-negative, got {seconds}")
        self.runtime.set_timeout_seconds(seconds)

    def send_recv_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Exchange attention heads across ranks (All2All pattern).
        Args:
            x: Input tensor of shape [batch, tokens, heads, head_dim].
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Output tensor with redistributed heads. Unless ``copy_out``, a view of the IPC
            buffer that the next call on this instance overwrites.
        """
        torch.ops.ltx_kernels.send_recv_heads(x, self._buffer, self._comm_id)
        batch, tokens, heads, head_dim = x.shape
        out = self._buffer[: x.numel()].view(batch, tokens * self.world_size, heads // self.world_size, head_dim)
        return out.clone() if copy_out else out

    def gather_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Gather heads back to original distribution (reverse All2All).
        Args:
            x: Input tensor with distributed heads.
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Output tensor with gathered heads. Unless ``copy_out``, a view of the IPC buffer
            that the next call on this instance overwrites.
        """
        torch.ops.ltx_kernels.gather_heads(x, self._buffer, self._comm_id)
        batch, tokens, heads, head_dim = x.shape
        out = self._buffer[: x.numel()].view(batch, tokens // self.world_size, heads * self.world_size, head_dim)
        return out.clone() if copy_out else out

    def allgather(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Allgather operation across all ranks.
        Args:
            x: Input tensor to gather.
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Gathered tensor from all ranks.
        """
        return self.runtime.allgather(x, copy_out)

    def destroy(self) -> None:
        """Release IPC handles and GPU memory buffers."""
        self._finalizer()
