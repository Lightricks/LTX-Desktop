"""In-memory Done/Failed tray for the generation queue.

Lives on the desktop process and is merged into the queue snapshot so the
Electron app and the remote app share the same rows. A restart clears it.
Nothing here is written to sqlite.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from services.records import GenerationRecord, QueueGenerationRecord, UnavailableError

FINISHED_TRAY_LIMIT = 20


class FinishedTray:
    def __init__(self) -> None:
        self._lock = Lock()
        self._previous: list[QueueGenerationRecord] = []
        self._has_previous = False
        self._done: list[QueueGenerationRecord] = []
        self._failed: list[QueueGenerationRecord] = []
        self._seen: set[str] = set()
        self._dismissed: set[str] = set()

    def observe(
        self,
        live: list[QueueGenerationRecord],
        load: Callable[[str], GenerationRecord | None],
    ) -> None:
        """Record jobs that left the live queue since the previous snapshot."""
        with self._lock:
            live_ids = {entry.generation.id for entry in live}
            previous_ids = {entry.generation.id for entry in self._previous}
            # Only an id that left and came back is a retry. Releasing every id
            # in this list undoes a dismiss when the list was read before the
            # job finished.
            if self._has_previous:
                self._release_live(live_ids - previous_ids)
            departed = [
                entry
                for entry in self._previous
                if entry.generation.id not in live_ids
            ]
        pending: list[QueueGenerationRecord] = []
        loaded: list[tuple[QueueGenerationRecord, GenerationRecord | None]] = []
        for entry in departed:
            try:
                loaded.append((entry, load(entry.generation.id)))
            except UnavailableError:
                raise
            except Exception:
                pending.append(entry)
        with self._lock:
            for entry, fresh in loaded:
                self._remember_locked(entry, fresh)
            self._previous = [*pending, *live]
            self._has_previous = True

    def release(self, generation_id: str) -> None:
        """Drop tray state for a job that is live again.

        Retry must call this itself. A completion can land before the next
        queue read, and `remember` ignores ids that are still dismissed.
        """
        with self._lock:
            self._release_live({generation_id})

    def remember(self, entry: QueueGenerationRecord) -> None:
        """Record a job that just succeeded or failed."""
        with self._lock:
            self._remember_locked(entry, entry.generation)

    def lists(
        self,
    ) -> tuple[list[QueueGenerationRecord], list[QueueGenerationRecord], list[str]]:
        with self._lock:
            unseen = [
                entry.generation.id
                for entry in self._done
                if entry.generation.id not in self._seen
            ]
            return list(self._done), list(self._failed), unseen

    def mark_seen(self, generation_id: str) -> bool:
        with self._lock:
            if not any(entry.generation.id == generation_id for entry in self._done):
                return False
            self._seen.add(generation_id)
            return True

    def dismiss_done(self, generation_id: str) -> bool:
        with self._lock:
            if not any(entry.generation.id == generation_id for entry in self._done):
                return False
            self._done = [
                entry for entry in self._done if entry.generation.id != generation_id
            ]
            self._seen.discard(generation_id)
            self._dismissed.add(generation_id)
            return True

    def clear_done(
        self, feature_is_allowed: Callable[[str], bool] | None = None
    ) -> None:
        with self._lock:
            self._done = self._dismiss_matching(
                self._done, feature_is_allowed, drop_seen=True
            )

    def clear_failed(
        self, feature_is_allowed: Callable[[str], bool] | None = None
    ) -> None:
        with self._lock:
            self._failed = self._dismiss_matching(
                self._failed, feature_is_allowed, drop_seen=False
            )

    def forget(self, generation_id: str) -> None:
        with self._lock:
            self._done = [
                entry for entry in self._done if entry.generation.id != generation_id
            ]
            self._failed = [
                entry
                for entry in self._failed
                if entry.generation.id != generation_id
            ]
            self._previous = [
                entry
                for entry in self._previous
                if entry.generation.id != generation_id
            ]
            self._seen.discard(generation_id)
            self._dismissed.add(generation_id)

    def _release_live(self, live_ids: set[str]) -> None:
        self._dismissed.difference_update(live_ids)
        self._seen.difference_update(live_ids)
        self._done = [
            entry for entry in self._done if entry.generation.id not in live_ids
        ]
        self._failed = [
            entry for entry in self._failed if entry.generation.id not in live_ids
        ]

    def _remember_locked(
        self,
        previous: QueueGenerationRecord,
        fresh: GenerationRecord | None,
    ) -> None:
        if fresh is None or fresh.id in self._dismissed:
            return
        if previous.generation.status in {"cancelling", "cancelled"}:
            return
        if fresh.status not in {"succeeded", "failed"}:
            return
        record = previous.model_copy(update={"generation": fresh})
        generation_id = fresh.id
        if fresh.status == "succeeded":
            self._failed = [
                entry
                for entry in self._failed
                if entry.generation.id != generation_id
            ]
            self._done = [
                record,
                *[
                    entry
                    for entry in self._done
                    if entry.generation.id != generation_id
                ],
            ]
            del self._done[FINISHED_TRAY_LIMIT:]
            return
        self._seen.discard(generation_id)
        self._done = [
            entry for entry in self._done if entry.generation.id != generation_id
        ]
        self._failed = [
            record,
            *[
                entry
                for entry in self._failed
                if entry.generation.id != generation_id
            ],
        ]
        del self._failed[FINISHED_TRAY_LIMIT:]

    def _dismiss_matching(
        self,
        entries: list[QueueGenerationRecord],
        feature_is_allowed: Callable[[str], bool] | None,
        *,
        drop_seen: bool,
    ) -> list[QueueGenerationRecord]:
        kept: list[QueueGenerationRecord] = []
        for entry in entries:
            if feature_is_allowed is None or feature_is_allowed(
                entry.generation.feature
            ):
                self._dismissed.add(entry.generation.id)
                if drop_seen:
                    self._seen.discard(entry.generation.id)
                continue
            kept.append(entry)
        return kept
