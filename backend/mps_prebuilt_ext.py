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
from collections.abc import Iterator
from typing import Any, Callable, cast

from server_utils.units import gib

logger = logging.getLogger(__name__)

_EXT_NAME = "mps_sdpa_zc_ext"
_GUARD_ATTR = "_ltx_coerces_none_fused_min"

# Mirror of mps-sdpa's M4-tuned defaults (backends/_calibrate._DEFAULT_THRESHOLDS).
# Used when calibration stored fused_min_bytes=None ("always stock") — that sentinel
# is a speed conclusion at L<=2048, not a memory-safe choice for video attention.
_FALLBACK_FUSED_MIN_BYTES = {
    "bf16": 4 * 1024**2,
    "fp16": 4 * 1024**2,
    "fp32": 8 * 1024**2,
}


def _prebuilt_so() -> Path | None:
    """The bundled ``mps_sdpa_zc_ext.so`` (electron sets LTX_MPS_EXT_PREBUILT_DIR)."""
    d = os.environ.get("LTX_MPS_EXT_PREBUILT_DIR")
    if not d:
        return None
    so = Path(d) / f"{_EXT_NAME}.so"
    return so if so.is_file() else None


def coerce_mps_sdpa_thresholds(
    thresholds: dict[str, Any],
    *,
    defaults: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Replace fused_min_bytes=None ("always stock") with safe defaults.

    mps-sdpa v0.2.0 calibration benches pyobjc mpsgraph vs stock at L<=2048. If
    pyobjc never wins by 5%, it caches ``null``. Both ``mpsgraph_zc`` and
    ``mpsgraph`` then treat ``fused_min is None`` as "use stock for every
    shape", including ~14k-token video self-attention. Stock MPS SDPA
    materializes the S×S score matrix (~47 GiB) and OOMs
    (https://github.com/Lightricks/LTX-Desktop/issues/161).
    """
    fused_raw = thresholds.get("fused_min_bytes")
    if not isinstance(fused_raw, dict):
        return thresholds
    fused = cast(dict[str, Any], fused_raw)
    replacements = defaults or _FALLBACK_FUSED_MIN_BYTES
    out_fused: dict[str, Any] = dict(fused)
    changed = False
    for key, default in replacements.items():
        if out_fused.get(key) is None:
            out_fused[key] = default
            changed = True
    if not changed:
        return thresholds
    return {**thresholds, "fused_min_bytes": out_fused}


def install_mps_sdpa_threshold_guard() -> None:
    """Wrap mps-sdpa ``get_thresholds()`` so None fused_min never reaches dispatch.

    No-op off Darwin or when mps-sdpa isn't installed. Idempotent.
    """
    if sys.platform != "darwin":
        return
    try:
        from mps_sdpa.backends import _calibrate  # noqa: PLC0415  # type: ignore[reportMissingModuleSource]
    except ImportError:
        return

    current = cast(
        Callable[[], dict[str, Any]],
        _calibrate.get_thresholds,  # pyright: ignore[reportUnknownMemberType]
    )
    if getattr(current, _GUARD_ATTR, False):
        return

    def _set_cached(value: dict[str, Any]) -> None:
        setattr(_calibrate, "_cached_thresholds", value)

    def _guarded() -> dict[str, Any]:
        raw = current()
        thresholds = coerce_mps_sdpa_thresholds(raw)
        if thresholds is not raw:
            _set_cached(thresholds)
            logger.warning(
                "mps-sdpa fused_min_bytes had None (always-stock); using defaults %s "
                "(raw=%s). Stock MPS SDPA OOMs video attention (LTX-Desktop#161).",
                thresholds.get("fused_min_bytes"),
                raw.get("fused_min_bytes"),
            )
        return thresholds

    setattr(_guarded, _GUARD_ATTR, True)
    _calibrate.get_thresholds = _guarded  # pyright: ignore[reportUnknownMemberType]
    cached = getattr(_calibrate, "_cached_thresholds", None)
    if isinstance(cached, dict):
        _set_cached(coerce_mps_sdpa_thresholds(cast(dict[str, Any], cached)))
    logger.info("mps_prebuilt_ext: installed fused_min_bytes=None → defaults guard")


def _torch_extensions_cache() -> Path:
    override = os.environ.get("TORCH_EXTENSIONS_DIR")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/torch_extensions"
    return Path.home() / ".cache" / "torch_extensions"


def _unlink_lock(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("mps_prebuilt_ext: could not remove stale %s", path, exc_info=True)
        return False
    return True


def _remove_uncontended_ninja_lock(path: Path) -> bool:
    """Unlink ninja's ``.ninja_lock`` only if no process holds ``LOCK_EX``.

    Ninja flocks this file for the compile. FileBaton's ``lock`` does not —
    that path is unlinked separately.
    """
    if sys.platform == "win32":
        return _unlink_lock(path)

    import fcntl

    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("mps_prebuilt_ext: could not open %s", path, exc_info=True)
        return False

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path.unlink()
        return True
    except BlockingIOError:
        return False
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("mps_prebuilt_ext: could not remove stale %s", path, exc_info=True)
        return False
    finally:
        os.close(fd)


def _iter_mps_sdpa_ext_dirs(root: Path) -> Iterator[Path]:
    """Yield ``mps_sdpa_zc_ext`` dirs under both torch cache layouts.

    Default cache: ``<root>/<pyXX_cpu>/mps_sdpa_zc_ext``.
    ``TORCH_EXTENSIONS_DIR`` override: ``<root>/mps_sdpa_zc_ext`` (no version folder).
    """
    direct = root / _EXT_NAME
    if direct.is_dir():
        yield direct
    yield from (path for path in root.glob(f"*/{_EXT_NAME}") if path.is_dir())


def clear_stale_mps_extension_locks(cache_root: Path | None = None) -> list[Path]:
    """Remove leftover JIT lock files from a killed mps-sdpa compile.

    SIGTERM during ``cpp_extension.load`` / ninja leaves ``lock`` and
    ``.ninja_lock`` behind. The next import then sleeps on those files and
    never reaches ``/health``, so the desktop loader stays on backendLoading.

    ``lock`` is torch FileBaton (O_CREAT|O_EXCL + exists-poll, no flock, no
    pid). Live vs crash leftover is the same on disk; we unlink it at backend
    start, before this process imports mps-sdpa.

    ``.ninja_lock`` is flocked by ninja. Unlink only after a non-blocking
    exclusive flock succeeds, so a live ninja is not interrupted.
    """
    root = cache_root if cache_root is not None else _torch_extensions_cache()
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for ext_dir in _iter_mps_sdpa_ext_dirs(root):
        baton = ext_dir / "lock"
        if _unlink_lock(baton):
            removed.append(baton)
        ninja_lock = ext_dir / ".ninja_lock"
        if _remove_uncontended_ninja_lock(ninja_lock):
            removed.append(ninja_lock)
    return removed


def setup_prebuilt_mps_extension() -> None:
    if sys.platform != "darwin":
        return

    stale_locks = clear_stale_mps_extension_locks()
    if stale_locks:
        logger.warning(
            "mps_prebuilt_ext: removed stale JIT lock(s) %s",
            [str(path) for path in stale_locks],
        )

    so = _prebuilt_so()
    if so is None:
        logger.info("mps_prebuilt_ext: no bundled prebuilt .so; torch will JIT-build if a compiler is present")
        install_mps_sdpa_threshold_guard()
        return

    try:
        from torch.utils import cpp_extension as _cppext  # noqa: PLC0415
    except Exception:
        logger.warning("mps_prebuilt_ext: torch.utils.cpp_extension unavailable; skipping", exc_info=True)
        install_mps_sdpa_threshold_guard()
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
    install_mps_sdpa_threshold_guard()


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
    install_mps_sdpa_threshold_guard()
    try:
        from mps_sdpa import api  # noqa: PLC0415  # type: ignore[reportMissingModuleSource]
        from mps_sdpa.backends import _calibrate  # noqa: PLC0415  # type: ignore[reportMissingModuleSource]

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
        get_thresholds = cast(
            Callable[[], dict[str, Any]],
            _calibrate.get_thresholds,  # pyright: ignore[reportUnknownMemberType]
        )
        thresholds = get_thresholds()
        logger.info(
            "mps-sdpa fused_min_bytes=%s calibrated=%s",
            thresholds.get("fused_min_bytes"),
            thresholds.get("calibrated"),
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
        parts = [" ".join(f"{k}={v}" for k, v in sorted(stats.items()))] if stats else []
        get_fallback = getattr(api, "get_fallback_stats", None)
        fallback = get_fallback() if callable(get_fallback) else None
        if isinstance(fallback, dict) and fallback:
            fb_items = cast(dict[str, Any], fallback)
            parts.append("fb:" + " ".join(f"{k}={v}" for k, v in sorted(fb_items.items())))
        return " ".join(parts) or "(no calls yet)"
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
        reset_fallback = getattr(api, "reset_fallback_stats", None)
        if callable(reset_fallback):
            reset_fallback()
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
