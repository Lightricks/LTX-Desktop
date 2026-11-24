"""Pass that merges a graph's fp8 parameter upcasts into one ``ltx_kernels::grouped_upcast``."""

import logging

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)


def _upcast_source(node: Node) -> Node | None:
    """The fp8 graph input ``node`` upcasts, if it is ``input.to(torch.bfloat16)`` of a static-shaped one."""
    is_to = node.op == "call_method" and node.target == "to" and not set(node.kwargs) - {"dtype"}
    dtype = node.kwargs.get("dtype", node.args[1] if len(node.args) == 2 else None) if is_to else None
    source = node.args[0] if is_to and len(node.args) <= 2 else None
    if dtype is not torch.bfloat16 or not isinstance(source, Node) or source.op != "placeholder":
        return None
    value = source.meta.get("example_value")
    eligible = (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float8_e4m3fn
        and value.is_cuda
        and value.is_contiguous()
        and all(isinstance(d, int) for d in value.shape)
    )
    return source if eligible else None


class GroupedUpcastPass(PassBase):
    """Batch the graph's ``param.to(torch.bfloat16)`` upcasts of fp8 inputs into ``grouped_upcast`` calls.
    Targets the per-forward weight and bias upcasts of ``Fp8CastLinear``. Upcasts are taken in the
    order of their first use and packed greedily into groups whose bf16 output is at most the
    largest single upcast in the graph; each group runs right before its first consumer, and each
    former upcast becomes a view of its group's output. A group's buffer lives until its last
    consumer, so bounding a group by the largest upcast bounds the added live memory by the same
    amount the ungrouped graph already holds for that tensor.
    """

    def __init__(self) -> None:
        self.match_count = 0

    def call(self, graph_module: GraphModule) -> PassResult:
        from ltx_kernels.ops import grouped_upcast as _grouped_upcast_ops  # noqa: F401, PLC0415

        graph = graph_module.graph
        upcasts: dict[Node, list[Node]] = {}
        upcast_sources: dict[Node, Node] = {}
        # Nodes are in topological order, so a source's first consumer follows its upcasts, and the
        # insertion order of ``first_use`` is the order of first use.
        first_use: dict[Node, Node] = {}
        for node in graph.nodes:
            for input_node in node.all_input_nodes:
                if input_node in upcast_sources:
                    first_use.setdefault(upcast_sources[input_node], node)
            source = _upcast_source(node)
            if source is not None:
                upcasts.setdefault(source, []).append(node)
                upcast_sources[node] = source

        sources = list(first_use)
        budget = max((source.meta["example_value"].numel() for source in sources), default=0)
        groups: list[list[Node]] = [[]]
        size = 0
        for source in sources:
            numel = source.meta["example_value"].numel()
            if groups[-1] and size + numel > budget:
                groups.append([])
                size = 0
            groups[-1].append(source)
            size += numel
        # A single-tensor group keeps its plain upcast.
        groups = [group for group in groups if len(group) > 1]
        if not groups:
            return PassResult(graph_module, modified=False)

        for group in groups:
            # Sources are in first-use order, so the group's first consumer is its first source's.
            with graph.inserting_before(first_use[group[0]]):
                flat = graph.call_function(torch.ops.ltx_kernels.grouped_upcast.default, (group,))
                offset = 0
                for source in group:
                    value = source.meta["example_value"]
                    narrowed = graph.call_function(torch.narrow, (flat, 0, offset, value.numel()))
                    view = graph.call_method("view", (narrowed, *value.shape))
                    offset += value.numel()
                    for node in upcasts[source]:
                        node.replace_all_uses_with(view)
                        graph.erase_node(node)
                        self.match_count += 1
        graph.lint()
        graph_module.recompile()
        logger.info("GroupedUpcastPass grouped %d fp8 upcast(s) into %d group(s)", self.match_count, len(groups))
        return PassResult(graph_module, modified=True)
