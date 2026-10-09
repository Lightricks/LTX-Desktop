"""Per-call helper swap for multi-keyframe generate.

``combined_image_conditionings`` replaces frame 0 and treats later images as
keyframes. MKF interpolation needs ``image_conditionings_by_adding_guiding_latent``,
which keeps every image (including frame 0) as a keyframe. The swap is scoped
to one Fast generate so first/last i2v keeps replace-at-0.

v1.4 calls the helper from ``ltx_pipelines.chunks.conditionings``, which binds
the name at import, and the stage-1-only path looks it up on ``helpers``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_SWAP_LOCK = threading.Lock()


@contextmanager
def distilled_keyframe_guiding() -> Iterator[None]:
    import ltx_pipelines.chunks.conditionings as chunk_conditionings
    import ltx_pipelines.utils.helpers as helpers
    from ltx_pipelines.utils.helpers import image_conditionings_by_adding_guiding_latent

    with _SWAP_LOCK:
        original_helpers = helpers.combined_image_conditionings
        original_chunks = chunk_conditionings.combined_image_conditionings
        helpers.combined_image_conditionings = image_conditionings_by_adding_guiding_latent
        chunk_conditionings.combined_image_conditionings = image_conditionings_by_adding_guiding_latent
        try:
            yield
        finally:
            helpers.combined_image_conditionings = original_helpers
            chunk_conditionings.combined_image_conditionings = original_chunks
