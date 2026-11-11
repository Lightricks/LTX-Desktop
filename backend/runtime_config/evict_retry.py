"""One-shot evict-and-resample for W-band execute rejects.

After the runner's release-then-sample, the execute-time free sample still counts warm
live weights W as occupied while the ask prices them (double-count). A reject whose ask
lands in ``(free', free'+W]`` is an artifact of sample ordering, not a true miss — so a
total-possible CUDA reject whose shortfall live weights cover gets one evict of the
resident pipeline and exactly one re-decide on a fresh sample instead of failing
outright. The re-sample tests ``ask <= T - O`` (T total, O other-apps) — the true fit
condition the first sample missed.

Split on purpose: :func:`evict_retry_shortfall_gib` is pure policy (directly
unit-testable, no torch), :func:`evict_retry_filter` is the impure executor (log, evict,
rebuild) driven by caller-owned closures. Callers sample live bytes and own the
re-decide, where the decision types differ per feature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from runtime_config.budget_admission import PricedJob
from runtime_config.budget_decision import VideoJobReject
from runtime_config.video_job_budget import (
    BudgetRoute,
    LocalJobFilter,
    budget_route,
    vram_snapshot_suffix,
)
from server_utils.units import gib

logger = logging.getLogger(__name__)


def evict_retry_shortfall_gib(
    *,
    route: BudgetRoute,
    is_vram_occupancy_miss: bool,
    stream_gib: float,
    job_filter: LocalJobFilter,
    live_allocated_bytes: int | None,
) -> float | None:
    """W-band shortfall eligible for evict-and-resample, or None.

    Narrow by construction — every condition below must hold, otherwise the
    caller keeps the reject exactly as decided:

    - ``windows_measured`` route only: legacy curves compare frozen integers
      (``memory_gb``), never live free, so a re-sample could not change them.
    - VRAM-occupancy miss only: callers map their own reject vocabulary (RAM
      shortfall, unsupported mode, and the IC-LoRA spatial/fence verdicts are
      not occupancy verdicts — evicting cannot help them).
    - Total-possible (``stream_gib <= total``): a bare card must be able to run
      it. Total-only admission rejects are total-short by construction, so
      queue-create preflights can never trigger a retry from here.
    - CUDA-measured free (``free_vram_bytes`` present): Mac pool, CPU, and
      generic paths have nothing CUDA to resample.
    - Shortfall covered by the passed live figure: sampled at the inter-job
      moment, when nothing else is live, so it approximates the warm weights W
      — the most an evict could free. A shortfall past that is other-apps
      occupancy and churning is pointless. None fails closed.
    """
    if route != "windows_measured" or not is_vram_occupancy_miss:
        return None
    budget = job_filter.budget
    if budget.free_vram_bytes is None:
        return None
    total = budget.usable_vram_gib
    available = budget.available_vram_gib
    if total is None or available is None:
        return None
    if stream_gib > total:
        return None
    shortfall = stream_gib - available
    if shortfall <= 0:
        return None
    if live_allocated_bytes is None:
        return None
    if shortfall > gib(live_allocated_bytes):
        return None
    return shortfall


def evict_retry_filter(
    *,
    priced: PricedJob,
    job_filter: LocalJobFilter,
    live_allocated_bytes: int | None,
    reserved_bytes: int | None,
    log_prefix: str,
    evict: Callable[[], None],
    rebuild: Callable[[], LocalJobFilter],
) -> LocalJobFilter | None:
    """Evict-and-resample once when the W-band covers the shortfall; else None.

    Logs the first-attempt line, runs the caller's evict, and returns the
    caller's rebuilt filter for exactly one re-price. None means keep the
    priced verdict — the policy declined, nothing was touched.
    """
    is_vram_occupancy_miss = (
        isinstance(priced.decision, VideoJobReject)
        and priced.decision.reason == "stream_over"
    )
    shortfall = evict_retry_shortfall_gib(
        route=budget_route(priced.feature, job_filter.profile),
        is_vram_occupancy_miss=is_vram_occupancy_miss,
        stream_gib=priced.stream_gib,
        job_filter=job_filter,
        live_allocated_bytes=live_allocated_bytes,
    )
    if shortfall is None:
        return None
    logger.info(
        "[%s] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> reject; evict-retry (shortfall %.1fGiB covered by live weights)%s",
        log_prefix,
        priced.seq,
        priced.full_gib,
        priced.stream_gib,
        shortfall,
        vram_snapshot_suffix(
            allocated_bytes=live_allocated_bytes,
            reserved_bytes=reserved_bytes,
            free_bytes=job_filter.budget.free_vram_bytes,
        ),
    )
    evict()
    return rebuild()
