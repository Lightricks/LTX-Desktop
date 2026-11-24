"""Reusable base class for FX passes that rewrite a single subgraph pattern."""

import logging
import warnings
from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

logger = logging.getLogger(__name__)


def is_cuda_bf16(*values: torch.Tensor) -> bool:
    """Whether every matched input is a cuda bf16 tensor."""
    return all(value.is_cuda and value.dtype == torch.bfloat16 for value in values)


def static_dim(dim: int | torch.SymInt) -> int | None:
    """A tensor dim as a plain ``int``, or ``None`` when it is dynamic.
    A dim marked dynamic reaches a pass as a ``SymInt``, and comparing one
    installs a guard outside the tracing context -- so an eligibility check
    treats it as unknown rather than reading it.
    """
    return dim if isinstance(dim, int) else None


class SubgraphRewritePass(PassBase):
    """Rewrites a single subgraph pattern via
    :class:`torch.fx.passes.utils.matcher_utils.SubgraphMatcher`.
    Both pattern and replacement are captured via Dynamo (using a
    capture-only inner backend) -- the same tracer the user graph goes
    through. This keeps op identities aligned: ``call_method`` with string
    targets for ``.view`` / ``.unsqueeze``, ``call_function`` with library
    specials like ``einops_einops_rearrange``, and in-place ops like
    ``addcmul_`` all survive intact.
    The pass operates at the Dynamo IR level (no AOT lowering), so the
    backend shim that drives it does not have to wrap calls in
    :func:`torch._dynamo.backends.common.aot_autograd` -- avoiding the
    per-call inference runtime wrapper overhead.
    Subclass contract:
    - ``pattern_fn`` -- ``@staticmethod`` Python function whose body is the
      composite to match.
    - ``replacement_fn`` -- ``@staticmethod`` Python function whose body is
      the rewritten subgraph. Same signature as ``pattern_fn``.
    - ``example_inputs()`` -- tuple of tensors (on cuda + bf16 where the
      replacement needs them) used to drive Dynamo capture of both
      ``pattern_fn`` and ``replacement_fn``.
    - ``_register_ops()`` (optional) -- hook called once before the first
      capture; override to lazily import op modules whose decorators populate
      ``torch.ops.<ns>``.
    Match counts are exposed via the public ``match_count`` attribute.
    """

    # ``Any`` here covers both single-tensor patterns and multi-output
    # ones (see the trailing-quantize passes, which return a 2-tuple).
    pattern_fn: ClassVar[Callable[..., Any]]
    replacement_fn: ClassVar[Callable[..., Any]]
    # Whether the matcher should treat scalar literals (ints, floats) in
    # the pattern as wildcards. ``True`` lets one pattern match user graphs
    # with different baked-in shape constants (e.g. H=4096 vs H=2048), but
    # can confuse the matcher when a literal value (like a bare ``1``)
    # could plausibly map to a placeholder slot.
    ignore_literals: ClassVar[bool] = True

    def __init__(self) -> None:
        self.match_count = 0
        self._pattern_gm: GraphModule | None = None
        self._replacement_gm: GraphModule | None = None
        self._capture_error: Exception | None = None

    def call(self, graph_module: GraphModule) -> PassResult:
        if self._capture_error is not None:
            return PassResult(graph_module, modified=False)
        if self._pattern_gm is None:
            try:
                self._register_ops()
                self._pattern_gm = self._capture_via_dynamo(self.pattern_fn)
            except Exception as error:
                self._disable(error)
                return PassResult(graph_module, modified=False)

        replaced = self._rewrite_matches(graph_module)
        if replaced:
            logger.info("%s replaced %d subgraph(s)", type(self).__name__, replaced)
        self.match_count += replaced
        return PassResult(graph_module, modified=bool(replaced))

    def _rewrite_matches(self, graph_module: GraphModule) -> int:  # noqa: PLR0912, PLR0915
        """Replace every eligible match and return how many were replaced, which is
        fewer than were matched when a match has no valid insertion point.
        A replacement goes before the earliest user of the match's outputs that lies
        outside the match; a match whose latest input is produced after that point has
        no valid insertion point and is skipped. Matching restarts after each
        replacement, and pattern and replacement placeholders are bound by name, since
        Dynamo orders placeholders by first use and the two captures can differ.
        The replacement is captured on the first eligible match, not up front, because
        capturing runs the replacement op.
        """
        if self._pattern_gm is None:
            raise AssertionError("pattern must be captured before rewriting")

        pattern_graph = self._pattern_gm.graph
        pattern_placeholders = {node.target: node for node in pattern_graph.nodes if node.op == "placeholder"}

        replaced = 0
        ineligible = 0
        blocked_anchors: set[Node] = set()
        while True:
            matcher = SubgraphMatcher(
                pattern_graph,
                match_output=False,
                match_placeholder=False,
                remove_overlapping_matches=True,
                ignore_literals=self.ignore_literals,
            )
            found = [match for match in matcher.match(graph_module.graph) if match.anchors[0] not in blocked_anchors]
            matches = [match for match in found if self._is_match_eligible(match)]
            # Rematching re-sees every rejected match, so only the last round's count
            # is meaningful: it is what the pattern still matches and the kernel cannot
            # take.
            ineligible = len(found) - len(matches)
            if not matches:
                break
            match = matches[0]

            if self._replacement_gm is None:
                try:
                    self._replacement_gm = self._capture_via_dynamo(self.replacement_fn)
                except Exception as error:
                    self._disable(error)
                    break
            replacement_graph = self._replacement_gm.graph
            replacement_placeholders = [node for node in replacement_graph.nodes if node.op == "placeholder"]
            if set(pattern_placeholders) != {node.target for node in replacement_placeholders}:
                raise AssertionError("pattern and replacement placeholder names must match")

            val_map: dict[Node, Node | object] = {}
            for replacement_placeholder in replacement_placeholders:
                pattern_placeholder = pattern_placeholders[replacement_placeholder.target]
                val_map[replacement_placeholder] = match.nodes_map[pattern_placeholder]

            matched_nodes = {
                graph_node
                for pattern_node, graph_node in match.nodes_map.items()
                if pattern_node.op not in ("placeholder", "output") and isinstance(graph_node, Node)
            }
            external_users = {
                user
                for returning_node in match.returning_nodes
                for user in returning_node.users
                if user not in matched_nodes
            }

            ordered_nodes = list(graph_module.graph.nodes)
            node_index = {node: index for index, node in enumerate(ordered_nodes)}
            mapped_inputs = [node for node in val_map.values() if isinstance(node, Node)]
            latest_input_index = max(node_index[node] for node in mapped_inputs)
            if external_users:
                insert_point = min(external_users, key=node_index.__getitem__)
            else:
                insert_point = next(node for node in ordered_nodes if node.op == "output")

            if latest_input_index >= node_index[insert_point]:
                logger.warning(
                    "%s skipped a match with no valid insertion point",
                    type(self).__name__,
                )
                blocked_anchors.add(match.anchors[0])
                continue

            with graph_module.graph.inserting_before(insert_point):
                copied_outputs = graph_module.graph.graph_copy(replacement_graph, val_map)
            if isinstance(copied_outputs, Node):
                copied_outputs = (copied_outputs,)
            if len(match.returning_nodes) != len(copied_outputs):
                raise AssertionError(
                    f"returning node count mismatch: {len(match.returning_nodes)} vs {len(copied_outputs)}"
                )

            for old_output, new_output in zip(match.returning_nodes, copied_outputs, strict=True):
                for user in list(old_output.users):
                    if user not in matched_nodes:
                        user.replace_input_with(old_output, new_output)

            for node in reversed(ordered_nodes):
                if node in matched_nodes:
                    graph_module.graph.erase_node(node)

            replaced += 1

        if replaced:
            graph_module.graph.lint()
            graph_module.recompile()
        logger.debug(
            "%s rewrote %d subgraph(s), left %d ineligible pattern match(es)",
            type(self).__name__,
            replaced,
            ineligible,
        )
        return replaced

    def _is_match_eligible(self, match: InternalMatch) -> bool:  # noqa: ARG002
        """Whether the replacement op can serve this match.
        Passes override this to state what their kernel handles, usually over
        ``_matched_inputs``, and call ``super()`` for the one condition they
        share: no replacement op has a backward, so none fires under grad.
        """
        return not torch.is_grad_enabled()

    def _matched_inputs(self, match: InternalMatch) -> tuple[torch.Tensor, ...]:
        """The traced example values bound to the pattern's placeholders, in order.
        Placeholder order is ``pattern_fn``'s signature order, since Dynamo
        orders placeholders by first use. Empty when a placeholder is bound to
        anything other than a traced tensor.
        """
        if self._pattern_gm is None:
            return ()

        values = []
        for placeholder in (node for node in self._pattern_gm.graph.nodes if node.op == "placeholder"):
            bound = match.nodes_map[placeholder]
            value = bound.meta.get("example_value") if isinstance(bound, Node) else None
            if not isinstance(value, torch.Tensor):
                return ()
            values.append(value)
        return tuple(values)

    def example_inputs(self) -> tuple:
        raise NotImplementedError

    def _capture_via_dynamo(self, fn: Callable) -> GraphModule:
        captured: list[GraphModule] = []

        def _capture(gm: GraphModule, _example_inputs: list[torch.Tensor]) -> Callable:
            captured.append(gm)
            return gm.forward

        # Inference mode mirrors the user pipeline's execution mode so the
        # captured graph uses the same dispatch path as the production code.
        # ``dynamic=False`` forces a static capture: the pattern is a shape-agnostic
        # op structure, and capturing it inside the dynamic-shape outer compile would
        # otherwise leak symbolic seq-dim placeholders into the pattern graph, which
        # ``SubgraphMatcher`` cannot map onto the user graph (KeyError on the symbol).
        with torch.inference_mode():
            torch.compile(fn, backend=_capture, fullgraph=True, dynamic=False)(*self.example_inputs())
        return captured[0]

    def _disable(self, error: Exception) -> None:
        """Retire the pass after a capture failure, warning once.
        A pass whose ops this build cannot run must not break the compile: the
        graph it would have rewritten stays as it is. Keeping the error is what
        makes the pass inert -- it is never retried.
        """
        self._capture_error = error
        warnings.warn(
            f"{type(self).__name__} is unusable on this build and will be skipped: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _register_ops(self) -> None:
        """Override to import the modules whose decorators register the
        ``ltx_kernels`` ops that ``pattern_fn`` / ``replacement_fn`` reference."""
