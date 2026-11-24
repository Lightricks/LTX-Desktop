"""Drop jobs that may still run at a provider.

A job at a provider has its own id there, and the provider can bill it. The store keeps that id
on the row of a job that failed or was cancelled, until a cancel at the provider succeeds.
The lane of the job knows how to cancel it (LaneSpec.cancel_orphan).

Two callers use this module:

- The runner, right after a job of a provider lane failed or was cancelled.
- The boot, for every row that still holds an id.

A cancel that fails, or a process that stops first, leaves the id on the row. The next boot
tries again. A lane with no callback, or a lane that is not registered, is skipped and its
rows keep their id.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence

from services.generation_queue.lanes import LaneSpec
from services.records import ProviderOrphan
from services.store import Store

logger = logging.getLogger(__name__)

# Finds the lane spec of a lane id. None when the lane is not registered.
SpecFor = Callable[[str], LaneSpec | None]
# Called with an orphan the provider dropped, to forget its id.
OnCancelled = Callable[[ProviderOrphan], None]


def forget_in(db: Store) -> OnCancelled:
    """An OnCancelled that clears the stored id of the orphan."""

    def forget(orphan: ProviderOrphan) -> None:
        db.clear_provider_job_id(orphan.generation_id, orphan.provider_job_id)

    return forget


def cancel_provider_orphans(
    orphans: Sequence[ProviderOrphan], spec_for: SpecFor, on_cancelled: OnCancelled
) -> int:
    """Ask each lane to cancel its orphans, one by one. Returns how many calls succeeded.

    Best effort: a failure is logged and never stops the others. The callback must be
    safe to repeat and must bound its own network time.
    """
    cancelled = 0
    for orphan in orphans:
        spec = spec_for(orphan.lane)
        if spec is None or spec.cancel_orphan is None:
            logger.warning(
                "Provider job %s of generation %s cannot be cancelled: lane %s has no cancel. "
                "The id stays on the row.",
                orphan.provider_job_id,
                orphan.generation_id,
                orphan.lane,
            )
            continue
        logger.info(
            "Cancelling provider job %s of generation %s (lane %s)",
            orphan.provider_job_id,
            orphan.generation_id,
            orphan.lane,
        )
        try:
            spec.cancel_orphan(orphan.provider_job_id)
        except Exception:
            logger.warning(
                "Could not cancel provider job %s of generation %s",
                orphan.provider_job_id,
                orphan.generation_id,
                exc_info=True,
            )
            continue
        cancelled += 1
        try:
            on_cancelled(orphan)
        except Exception:
            logger.warning(
                "Provider job %s of generation %s was cancelled but its id was not cleared",
                orphan.provider_job_id,
                orphan.generation_id,
                exc_info=True,
            )
    return cancelled


def cancel_provider_orphans_in_background(
    orphans: Sequence[ProviderOrphan], spec_for: SpecFor, on_cancelled: OnCancelled
) -> threading.Thread | None:
    """Run cancel_provider_orphans on a daemon thread, so a slow provider never holds a worker."""
    if not orphans:
        return None
    thread = threading.Thread(
        target=cancel_provider_orphans,
        args=(tuple(orphans), spec_for, on_cancelled),
        name="provider-orphan-cancel",
        daemon=True,
    )
    thread.start()
    return thread
