"""Request-side settings for Normalized Attention Guidance (NAG).

No torch or ltx imports: handlers and the pipeline protocol import this module.
The attention patch lives in ``services/patches/nag_cross_attention.py``.

NAG runs whenever ``negative_prompt`` is non-empty and ``scale > 0`` (unless
``LTX_NAG_ENABLED=0``). The scale follows the KJNodes ``LTX2_NAG`` convention:
``guidance = scale * Z_pos - (scale - 1) * Z_neg``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from runtime_config.nag_limits import (
    DEFAULT_ALPHA,
    DEFAULT_SCALE,
    DEFAULT_TAU,
    ENV_ALPHA,
    ENV_ENABLED,
    ENV_SCALE,
    ENV_TAU,
    MAX_ALPHA,
    MAX_SCALE,
    MAX_TAU,
)


class NagOffReason(StrEnum):
    DISABLED_BY_ENV = "disabled_by_env"
    NO_NEGATIVE = "no_negative"
    STRENGTH_ZERO = "strength_zero"
    API_ENCODING = "api_encoding"
    TORCH_COMPILE = "torch_compile"


NAG_OFF_MESSAGE: Mapping[NagOffReason, str | None] = {
    NagOffReason.DISABLED_BY_ENV: None,
    NagOffReason.NO_NEGATIVE: None,
    NagOffReason.STRENGTH_ZERO: (
        "Negative prompt guidance (NAG) is turned off in Settings (strength is 0)."
    ),
    NagOffReason.API_ENCODING: (
        "Negative prompt guidance (NAG) requires the local text encoder and is "
        "skipped while LTX API text encoding is active."
    ),
    NagOffReason.TORCH_COMPILE: (
        "Negative prompt guidance (NAG) is skipped while Torch Compile is "
        "enabled in Settings."
    ),
}

_NAG_SKIP_LOG_REASON: Mapping[NagOffReason, str | None] = {
    NagOffReason.DISABLED_BY_ENV: None,
    NagOffReason.NO_NEGATIVE: None,
    NagOffReason.STRENGTH_ZERO: None,
    NagOffReason.API_ENCODING: "the negative prompt needs local text encoding",
    NagOffReason.TORCH_COMPILE: "torch.compile is enabled",
}


@dataclass(frozen=True)
class NagConfig:
    negative_prompt: str
    scale: float = DEFAULT_SCALE
    alpha: float = DEFAULT_ALPHA
    tau: float = DEFAULT_TAU


@dataclass(frozen=True)
class NagDecision:
    config: NagConfig | None
    reason: NagOffReason | None = None

    @property
    def active(self) -> bool:
        return self.config is not None

    @property
    def enhancer_negative(self) -> str:
        return self.config.negative_prompt if self.config is not None else ""

    @property
    def user_message(self) -> str | None:
        return NAG_OFF_MESSAGE.get(self.reason) if self.reason is not None else None


def _read_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    low: float,
    high: float,
) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}, got {value:g}")
    return value


def decide_nag(
    negative_prompt: str,
    environ: Mapping[str, str],
    *,
    scale: float | None = None,
    alpha: float | None = None,
    tau: float | None = None,
    default_scale: float = DEFAULT_SCALE,
    local_encoding: bool = True,
    compiled: bool = False,
) -> NagDecision:
    """Decide whether NAG runs for one job, and why when it stays off."""
    if environ.get(ENV_ENABLED, "").strip() == "0":
        return NagDecision(config=None, reason=NagOffReason.DISABLED_BY_ENV)
    negative = negative_prompt.strip()
    if not negative:
        return NagDecision(config=None, reason=NagOffReason.NO_NEGATIVE)
    resolved_scale = (
        scale
        if scale is not None
        else _read_float(environ, ENV_SCALE, default_scale, low=0.0, high=MAX_SCALE)
    )
    if resolved_scale <= 0.0:
        return NagDecision(config=None, reason=NagOffReason.STRENGTH_ZERO)
    if not local_encoding:
        return NagDecision(config=None, reason=NagOffReason.API_ENCODING)
    if compiled:
        return NagDecision(config=None, reason=NagOffReason.TORCH_COMPILE)
    resolved_alpha = (
        alpha
        if alpha is not None
        else _read_float(environ, ENV_ALPHA, DEFAULT_ALPHA, low=0.0, high=MAX_ALPHA)
    )
    resolved_tau = (
        tau
        if tau is not None
        else _read_float(environ, ENV_TAU, DEFAULT_TAU, low=0.0, high=MAX_TAU)
    )
    return NagDecision(
        config=NagConfig(
            negative_prompt=negative,
            scale=resolved_scale,
            alpha=resolved_alpha,
            tau=resolved_tau,
        ),
        reason=None,
    )


def nag_config_for_request(
    negative_prompt: str,
    environ: Mapping[str, str],
    *,
    scale: float | None = None,
    alpha: float | None = None,
    tau: float | None = None,
    default_scale: float = DEFAULT_SCALE,
    local_encoding: bool = True,
    compiled: bool = False,
) -> NagConfig | None:
    """Return the NAG settings for one job, or ``None`` when NAG must stay off.

    ``None`` keeps the unguided path, so output stays bit-identical to a job
    without NAG.
    """
    return decide_nag(
        negative_prompt,
        environ,
        scale=scale,
        alpha=alpha,
        tau=tau,
        default_scale=default_scale,
        local_encoding=local_encoding,
        compiled=compiled,
    ).config


def log_nag_decision(
    log: logging.Logger, capability: str, decision: NagDecision
) -> NagConfig | None:
    """Log ``NAG on`` or ``NAG skipped`` for a run and return ``decision.config``."""
    if decision.config is not None:
        cfg = decision.config
        log.info(
            "[%s] NAG on (scale=%g, alpha=%g, tau=%g)",
            capability,
            cfg.scale,
            cfg.alpha,
            cfg.tau,
        )
        return cfg
    skip_reason = (
        _NAG_SKIP_LOG_REASON.get(decision.reason)
        if decision.reason is not None
        else None
    )
    if skip_reason is not None:
        log.warning("[%s] NAG skipped: %s", capability, skip_reason)
    return None
