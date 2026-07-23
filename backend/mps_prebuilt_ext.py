"""Make mps-sdpa's zero-copy `mpsgraph_zc` backend load from a bundled prebuilt
binary — with no compiler, ninja, or recompile on the user's machine.

Background
----------
mps-sdpa's fast attention backend (`mpsgraph_zc`) is a torch C++/Obj-C++ extension.
mps-sdpa loads it via ``torch.utils.cpp_extension.load()`` (JIT), which — even for a
warm, bundled cache — re-derives an absolute-path compile command and asks ninja to
"rebuild"; because the cache was built at a different path than where it runs, ninja
recompiles → needs Xcode Command Line Tools. End users don't have those, so it falls
back to the pure-pyobjc `mpsgraph` backend, which leaks Metal memory per attention
call and OOMs longer generations (see docs/mps-attention-memory-leak.md).

Fix: the extension is pre-compiled at build time (prepare-python.sh Step 6.5) and
bundled. Here we intercept ``cpp_extension.load(name="mps_sdpa_zc_ext", ...)`` and
**import the bundled .so directly** (like a normal compiled module) instead of
rebuilding it. The .so resolves its libtorch symbols from the already-loaded torch
in-process, so it's location-independent and needs no toolchain. Import this at
backend startup, BEFORE mps-sdpa is first imported.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

from server_utils.units import gib

logger = logging.getLogger(__name__)

_EXT_NAME = "mps_sdpa_zc_ext"


def _prebuilt_so() -> Path | None:
    """The bundled ``mps_sdpa_zc_ext.so`` (electron sets LTX_MPS_EXT_PREBUILT_DIR)."""
    d = os.environ.get("LTX_MPS_EXT_PREBUILT_DIR")
    if not d:
        return None
    so = Path(d) / f"{_EXT_NAME}.so"
    return so if so.is_file() else None


def setup_prebuilt_mps_extension() -> None:
    if sys.platform != "darwin":
        return

    so = _prebuilt_so()
    if so is None:
        logger.info("mps_prebuilt_ext: no bundled prebuilt .so; torch will JIT-build if a compiler is present")
        return

    try:
        from torch.utils import cpp_extension as _cppext  # noqa: PLC0415
    except Exception:
        logger.warning("mps_prebuilt_ext: torch.utils.cpp_extension unavailable; skipping", exc_info=True)
        return

    _orig_load = _cppext.load  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    _cached: dict[str, object] = {}

    def _patched_load(name: Any = None, *args: Any, **kwargs: Any) -> Any:  # noqa: A002
        if name == _EXT_NAME:
            mod = _cached.get(_EXT_NAME)
            if mod is not None:
                return mod
            try:
                spec = importlib.util.spec_from_file_location(_EXT_NAME, str(so))
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    # Register before exec so the module is importable by name (normal
                    # import semantics) during its own execution; undo on failure.
                    sys.modules[_EXT_NAME] = mod
                    try:
                        spec.loader.exec_module(mod)  # type: ignore[union-attr]
                    except Exception:
                        sys.modules.pop(_EXT_NAME, None)
                        raise
                    _cached[_EXT_NAME] = mod
                    return mod
            except Exception:
                logger.warning(
                    "mps_prebuilt_ext: direct import of bundled %s failed; falling back to torch JIT",
                    so, exc_info=True,
                )
        return _orig_load(name, *args, **kwargs)

    _cppext.load = _patched_load  # pyright: ignore[reportUnknownMemberType]
    logger.info("mps_prebuilt_ext: cpp_extension.load patched to direct-import bundled %s", so)


def log_mps_backend_status() -> None:
    """Log which mps-sdpa attention backend is actually active.

    Distinguishes the fast bounded `mpsgraph_zc` from the leaky pyobjc `mpsgraph`
    fallback — the transformer's own "attention backends -- self: MPS-SDPA" line
    cannot. Call AFTER logging is configured. Importing mps_sdpa here also
    triggers the (already-patched) extension load, so this reflects the real
    in-process state. No-op off Darwin.
    """
    if sys.platform != "darwin":
        return
    try:
        from mps_sdpa import api  # noqa: PLC0415  # type: ignore[reportMissingModuleSource]

        st = api.backend_status(backend="auto", device="mps")
        logger.info(
            "mps-sdpa attention backend: picked=%s active=%s available=%s",
            st["picked"], st["active"], st["available"],
        )
        if st["picked"] != "mpsgraph_zc":
            logger.warning(
                "mps-sdpa: fast zero-copy 'mpsgraph_zc' NOT active (picked=%s) — "
                "attention may leak Metal memory. unavailable=%s",
                st["picked"], st.get("unavailable"),
            )
    except Exception:
        logger.warning("mps-sdpa: could not determine active attention backend", exc_info=True)


def _mps_sdpa_call_stats() -> str:
    """Per-call attention backend counts, e.g. 'mpsgraph_zc=1200 mpsgraph=0 stock=0'.

    This is the definitive signal that the fast zero-copy backend is actually
    serving calls — the transformer's "MPS-SDPA" name cannot distinguish it from
    the leaky pyobjc fallback. Empty string if the stats API is unavailable.
    """
    try:
        from mps_sdpa import api  # noqa: PLC0415  # type: ignore[reportMissingModuleSource]

        stats = api.get_call_stats()
        return " ".join(f"{k}={v}" for k, v in sorted(stats.items())) or "(no calls yet)"
    except Exception:
        return ""


def reset_mps_sdpa_stats() -> None:
    """Zero the per-call sdpa backend counters so a run's counts start fresh. No-op off MPS."""
    import torch  # noqa: PLC0415

    if sys.platform != "darwin" or not torch.backends.mps.is_available():
        return
    try:
        from mps_sdpa import api  # noqa: PLC0415  # type: ignore[reportMissingModuleSource]

        api.reset_call_stats()
    except Exception:
        pass


def mps_memory_sample() -> str | None:
    """One-line MPS memory + active-sdpa-backend snapshot, or None when not on MPS.

    driver-allocated climbing while torch-tracked (current) stays flat is the
    pyobjc-fallback leak signature; the sdpa counts show which backend served calls.
    Intended to be called on an interval by the generic heartbeat (server_utils.heartbeat).
    """
    import torch  # noqa: PLC0415

    if sys.platform != "darwin" or not torch.backends.mps.is_available():
        return None
    try:
        cur = gib(torch.mps.current_allocated_memory())
        drv = gib(torch.mps.driver_allocated_memory())
        mx = gib(torch.mps.recommended_max_memory())
        return f"torch={cur:.2f}GiB driver={drv:.2f}GiB max={mx:.2f}GiB | sdpa: {_mps_sdpa_call_stats()}"
    except Exception:
        logger.warning("[mps-mem] sample failed", exc_info=True)
        return None
