"""Pass that folds the all2all head exchange into the kernel producing the exchanged tensor."""

import logging
import operator

import torch
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)


def _is_target(node: object, *targets: object) -> bool:
    return isinstance(node, Node) and node.op == "call_function" and node.target in targets


def _sole_user(node: Node) -> Node | None:
    return next(iter(node.users)) if len(node.users) == 1 else None


class FusedSendRecvPass(PassBase):
    """Rewrite ``send_recv_heads(producer(...).view(b, t, heads, d))`` to a fused producer-exchange op.
    Producers taken, when the exchange is their only consumer:
    - ``ltx_kernels::rms_norm_split_rope`` (left by :class:`PreAttentionPass`) ->
      ``ltx_kernels::rope_send_recv_heads``, which normalizes, rotates and writes each head straight
      into its owning rank's buffer.
    - ``linear`` -> ``ltx_kernels::linear_send_recv_heads``, which runs the projection per rank-owned
      column block with the output in that rank's buffer.
    Both write the same bytes ``send_recv_heads`` would, into the same buffer, so everything reading
    the buffer afterwards is unchanged. Must run after :class:`PreAttentionPass`.
    A plain pass rather than a :class:`SubgraphRewritePass`: ``send_recv_heads`` returns nothing,
    so a pattern ending in it has no returning node for the matcher to anchor on, and capturing it
    through Dynamo would run the exchange.
    """

    def __init__(self) -> None:
        self.match_count = 0

    def call(self, graph_module: GraphModule) -> PassResult:
        from ltx_kernels import all_to_all as _all_to_all_ops  # noqa: F401, PLC0415
        from ltx_kernels.ops import rms_norm_split_rope as _rope_ops  # noqa: F401, PLC0415

        send_recv = (torch.ops.ltx_kernels.send_recv_heads, torch.ops.ltx_kernels.send_recv_heads.default)
        rope = torch.ops.ltx_kernels.rms_norm_split_rope.default
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if not _is_target(node, *send_recv) or node.kwargs:
                continue
            view, buffer, comm_id = node.args
            if not (isinstance(view, Node) and view.op == "call_method" and view.target == "view"):
                continue
            producer = view.args[0]
            if _sole_user(view) is not node or not isinstance(producer, Node) or _sole_user(producer) is not view:
                continue
            if _is_target(producer, rope) and not producer.kwargs:
                x, weights, cos, sin, eps = producer.args
                target, args = torch.ops.ltx_kernels.rope_send_recv_heads.default, (x, weights, cos, sin, eps)
            elif _is_target(producer, torch._C._nn.linear) and len(producer.args) == 3 and not producer.kwargs:
                x, weight, bias = producer.args
                if bias is None:
                    continue
                target, args = torch.ops.ltx_kernels.linear_send_recv_heads.default, (x, weight, bias)
            else:
                continue
            value = args[0].meta.get("example_value") if isinstance(args[0], Node) else None
            buffer_value = buffer.meta.get("example_value") if isinstance(buffer, Node) else None
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or value.ndim != 3:
                continue
            # The fused op takes batch and tokens from the producer input, so the exchanged view must
            # split only the hidden dim; a view regrouping batch and tokens has another buffer layout.
            heads_view = view.meta.get("example_value")
            if not (
                isinstance(heads_view, torch.Tensor)
                and heads_view.ndim == 4
                and all(statically_known_true(heads_view.shape[i] == value.shape[i]) for i in (0, 1))
            ):
                continue
            # The fused producers write bf16; a blockwise-fp8 exchange keeps its own copy kernel.
            if not isinstance(buffer_value, torch.Tensor) or buffer_value.dtype != torch.bfloat16:
                continue
            with graph.inserting_before(node):
                fused = graph.call_function(target, (*args, buffer, comm_id))
            node.replace_all_uses_with(fused)
            graph.erase_node(node)
            graph.erase_node(view)
            graph.erase_node(producer)
            self.match_count += 1
            modified = True
        if modified:
            graph.lint()
            graph_module.recompile()
            logger.info("FusedSendRecvPass fused %d exchange(s) into their producers", self.match_count)
        return PassResult(graph_module, modified)


# Methods that always return a view of their input.
_VIEW_METHODS = frozenset({"view", "unflatten", "transpose", "permute", "unsqueeze", "squeeze"})
# Methods that return a view only when the input's strides allow one, and a copy otherwise.
_MAYBE_VIEW_METHODS = frozenset({"flatten", "reshape"})


def _is_basic_index(index: object) -> bool:
    """Whether ``tensor[index]`` is basic indexing (a view), not advanced indexing (a copy)."""
    if isinstance(index, tuple):
        return all(_is_basic_index(i) for i in index)
    if isinstance(index, slice):
        return all(_is_basic_index(bound) for bound in (index.start, index.stop, index.step))
    if isinstance(index, Node):
        # A traced size; a tensor index is advanced.
        index = index.meta.get("example_value")
    return (
        index is None or index is Ellipsis or (isinstance(index, (int, torch.SymInt)) and not isinstance(index, bool))
    )


def _shares_storage(node: Node, source: Node) -> bool:
    """Whether the traced values prove ``node`` is a view of ``source``."""
    from torch.multiprocessing.reductions import StorageWeakRef  # noqa: PLC0415

    value, source_value = node.meta.get("example_value"), source.meta.get("example_value")
    if not (isinstance(value, torch.Tensor) and isinstance(source_value, torch.Tensor)):
        return False
    return StorageWeakRef(value.untyped_storage()) == StorageWeakRef(source_value.untyped_storage())


def _is_view(node: Node) -> bool:
    """Whether ``node`` only re-views its first argument, so reaching it does not read the data."""
    if node.op == "call_method" and node.target in _VIEW_METHODS:
        return True
    if node.op == "call_method" and node.target in _MAYBE_VIEW_METHODS:
        return isinstance(node.args[0], Node) and _shares_storage(node, node.args[0])
    return _is_target(node, operator.getitem) and len(node.args) == 2 and _is_basic_index(node.args[1])


class CoalescedExchangeBarrierPass(PassBase):
    """Complete consecutive fused exchanges with one barrier instead of one each.
    Attention reads its q, k and v buffers only after all three exchanges, so their per-exchange
    barriers -- a cross-rank round trip each -- can be one ``ltx_kernels::exchange_barrier``, placed
    before the first node that reads any of their buffers. Views of a buffer are not reads; ops that
    may copy (advanced indexing, a ``flatten`` the strides do not allow as a view) are, as is a later
    exchange taking a pending buffer, which then starts the next group. The barrier names the buffers
    as mutated, so everything that reads them is ordered after it. A group that is not read inside the
    graph keeps its barriers. Must run after :class:`FusedSendRecvPass`.
    """

    def __init__(self) -> None:
        self.match_count = 0

    def call(self, graph_module: GraphModule) -> PassResult:
        from ltx_kernels import all_to_all as _all_to_all_ops  # noqa: F401, PLC0415

        fused = (
            torch.ops.ltx_kernels.rope_send_recv_heads.default,
            torch.ops.ltx_kernels.linear_send_recv_heads.default,
        )
        graph = graph_module.graph
        removed = 0
        pending: list[Node] = []
        aliases: set[Node] = set()
        for node in list(graph.nodes):
            if pending and _is_view(node) and node.args[0] in aliases:
                aliases.add(node)
                continue
            if pending and any(input_node in aliases for input_node in node.all_input_nodes):
                if len(pending) > 1:
                    for exchange in pending:
                        exchange.update_kwarg("barrier", False)
                    with graph.inserting_before(node):
                        buffers = [exchange.args[-2] for exchange in pending]
                        graph.call_function(
                            torch.ops.ltx_kernels.exchange_barrier.default, (buffers, pending[-1].args[-1])
                        )
                    removed += len(pending) - 1
                pending, aliases = [], set()
            # Checked after the reads: an exchange whose inputs are a pending buffer completes that group first.
            if _is_target(node, *fused) and node.kwargs.get("barrier", True):
                pending.append(node)
                aliases.add(node.args[-2])
        if removed:
            graph.lint()
            graph_module.recompile()
            logger.info("CoalescedExchangeBarrierPass removed %d exchange barrier(s)", removed)
        self.match_count += removed
        return PassResult(graph_module, bool(removed))
