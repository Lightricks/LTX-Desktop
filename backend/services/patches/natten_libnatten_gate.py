"""Treat Flex-only natten as missing so DiffVAE can fall back to Triton.

ltx-core's ``natten_available()`` is True after ``import natten`` succeeds.
CHUNKED_EAGER then pins ``cutlass-fna``, which needs compiled ``libnatten``.
A PyPI Flex-Attention wheel imports fine, ``HAS_LIBNATTEN`` is False, and
decode crashes instead of taking the Triton/eager remap in ``apply.py``.

Our Windows GCS wheel has ``HAS_LIBNATTEN=True``. This gate keeps that path
and fails closed for Flex-only installs.

Remove once ltx-core's ``natten_available()`` checks ``natten.HAS_LIBNATTEN``.

Usage:
    import services.patches.natten_libnatten_gate  # noqa: F401
"""

from __future__ import annotations

import logging
from typing import Any

from ltx_core.model.video_vae.transformer import attention as na_mod

logger = logging.getLogger(__name__)


def _has_libnatten() -> bool:
    try:
        import natten
    except ImportError:
        return False
    return bool(getattr(natten, "HAS_LIBNATTEN", False))


def _gate_natten_available(attention_mod: Any, *, has_libnatten: bool) -> None:
    if attention_mod._NATTEN_AVAILABLE and not has_libnatten:
        logger.warning(
            "natten imported without libnatten; "
            "DiffVAE will fall back to Triton/eager instead of cutlass-fna"
        )
        attention_mod._NATTEN_AVAILABLE = False


_gate_natten_available(na_mod, has_libnatten=_has_libnatten())
