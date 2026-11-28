"""Custom ``torch.compile`` backend that applies ltx-kernels passes.
The shim:
1. Runs the passes from :mod:`ltx_kernels.compile.passes` via
   :class:`torch.fx.passes.infra.pass_manager.PassManager` over the Dynamo
   :class:`torch.fx.GraphModule` directly -- no AOT autograd round-trip.
2. Delegates the (possibly-rewritten) graph to a registered backend named by
   ``inner_backend``, resolved via :func:`torch._dynamo.lookup_backend`.
Operating on the Dynamo graph (instead of the AOT-lowered ATen graph) keeps
us off ``torch._dynamo.backends.common.aot_autograd``'s per-call inference
runtime wrapper, which on transformer-block-sized graphs adds ~28% wall
time when the inner backend is ``"eager"`` (the wrapper costs are amortized
when the inner backend is ``"inductor"``, but punishing on ``"eager"``).
The passes in :mod:`ltx_kernels.compile.passes` are written to match the
Dynamo IR op forms accordingly.
Two pre-built backends are exposed via ``torch_dynamo_backends`` setuptools
entry points (see ``pyproject.toml``) and resolve automatically once the
package is installed:
- ``ltx_kernels_eager`` -- inner backend ``"eager"``, :func:`eager_passes`.
  Default for inference.
- ``ltx_kernels_inductor`` -- inner backend ``"inductor"``,
  :func:`inductor_passes`. Pays compile time for inductor's whole-graph fusion of
  non-matched ops.
For a custom pass list or other inner backend use
:func:`make_ltx_kernels_backend`::
    torch.compile(model, backend=make_ltx_kernels_backend(passes=[...], inner_backend="..."))
"""

import logging
from collections.abc import Callable

import torch
from torch._dynamo.backends.registry import lookup_backend
from torch.fx.passes.infra.pass_base import PassBase
from torch.fx.passes.infra.pass_manager import PassManager

from ltx_kernels.compile.passes import eager_passes, inductor_passes

logger = logging.getLogger(__name__)


def make_ltx_kernels_backend(
    passes: list[PassBase],
    inner_backend: str = "eager",
) -> Callable:
    """Build an ltx-kernels backend with a chosen pass list and inner backend.
    Args:
        passes: List of :class:`PassBase` instances to apply, in the order given.
        inner_backend: Name of the backend to dispatch to after passes run,
            resolved via :func:`torch._dynamo.lookup_backend`. Examples:
            ``"eager"``, ``"inductor"``.
    """
    pass_manager = PassManager(passes=list(passes))

    def _compile(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        result = pass_manager(gm)
        return lookup_backend(inner_backend)(result.graph_module, example_inputs)

    return _compile


# Entry-point targets in pyproject.toml. ``torch.compile`` resolves
# ``backend="ltx_kernels_eager"`` / ``"ltx_kernels_inductor"`` to these
# automatically once the package is installed.
ltx_kernels_eager_backend = make_ltx_kernels_backend(passes=eager_passes(), inner_backend="eager")
ltx_kernels_inductor_backend = make_ltx_kernels_backend(passes=inductor_passes(), inner_backend="inductor")
