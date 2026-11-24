"""CSE for identical ``ltx_kernels::blockwise_quantize`` nodes."""

import logging

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)


class BlockwiseQuantizeCSE(PassBase):
    """Dedup identical ``ltx_kernels::blockwise_quantize(x)`` nodes.
    The op is a pure ``triton_op`` (``mutates_args=()``), so two calls on the
    same input node are interchangeable. Self-attention emits three identical
    quantizes (``to_q`` / ``to_k`` / ``to_v`` share the normed input); collapse
    them to one. Run before the trailing-quantize fusion passes so a fusible
    producer is left with a single quantize consumer.
    """

    def __init__(self) -> None:
        self.match_count = 0

    def call(self, graph_module: GraphModule) -> PassResult:
        # Importing registers ltx_kernels::blockwise_quantize so the target below resolves.
        from ltx_kernels.blockwise.ops import quantize as _quantize_ops  # noqa: F401, PLC0415

        target = torch.ops.ltx_kernels.blockwise_quantize.default
        graph = graph_module.graph
        seen: dict[object, Node] = {}
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target is not target:
                continue
            key = (node.args, tuple(sorted(node.kwargs.items())))
            canonical = seen.get(key)
            if canonical is None:
                seen[key] = node
                continue
            node.replace_all_uses_with(canonical)
            graph.erase_node(node)
            self.match_count += 1
            modified = True
        if modified:
            graph.lint()
            graph_module.recompile()
            logger.info("BlockwiseQuantizeCSE merged %d duplicate quantize(s)", self.match_count)
        return PassResult(graph_module, modified)
