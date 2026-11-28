"""FP6 weight pack / unpack ops.
Wraps the ``ops_cpp`` pack/unpack pybind kernels as ``torch.library.custom_op``
so they trace as opaque ``call_function`` nodes (the raw pybind builtins
graph-break Dynamo, splitting the compiled ``BlockwiseFP6Linear.forward``). FP6
packing drops the two highest exponent bits (e_1, e_2) for a 25% weight memory
saving; unpack restores them as 0.
"""

import torch


@torch.library.custom_op("ltx_kernels::fp6_pack", mutates_args=())
def fp6_pack(x: torch.Tensor) -> torch.Tensor:
    """Pack an 8-bit tensor ``[m, n]`` to 6-bit ``[m, n*3//4]`` uint8 (``n % 8 == 0``)."""
    import ops_cpp  # noqa: PLC0415

    return ops_cpp.fp6_pack(x)


@fp6_pack.register_fake
def _fp6_pack_fake(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    return torch.empty(m, (n * 3) // 4, dtype=torch.uint8, device=x.device)


@torch.library.custom_op("ltx_kernels::fp6_unpack", mutates_args=())
def fp6_unpack(x: torch.Tensor, original_n: int) -> torch.Tensor:
    """Unpack a 6-bit tensor ``[m, n_packed]`` back to 8-bit ``[m, original_n]`` uint8."""
    import ops_cpp  # noqa: PLC0415

    return ops_cpp.fp6_unpack(x, original_n)


@fp6_unpack.register_fake
def _fp6_unpack_fake(x: torch.Tensor, original_n: int) -> torch.Tensor:
    return torch.empty(x.shape[0], original_n, dtype=torch.uint8, device=x.device)


def fp6_pack_tensor(x: torch.Tensor) -> torch.Tensor:
    """Pack FP8 weights (uint8 view) to FP6; see :func:`fp6_pack`."""
    return torch.ops.ltx_kernels.fp6_pack(x)


def fp6_unpack_tensor(x: torch.Tensor, original_n: int) -> torch.Tensor:
    """Unpack FP6 weights back to FP8 layout (uint8); see :func:`fp6_unpack`."""
    return torch.ops.ltx_kernels.fp6_unpack(x, original_n)
