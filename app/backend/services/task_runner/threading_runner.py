"""Background task runner service."""

from __future__ import annotations

import threading
from collections.abc import Callable

from logging_policy import log_background_exception


class ThreadingRunner:
    """Runs tasks on daemon threads."""

    def run_background(
        self,
        target: Callable[[], None],
        *,
        task_name: str,
        on_error: Callable[[Exception], None] | None = None,
        daemon: bool = True,
    ) -> None:
        def _run() -> None:
            try:
                target()
            except Exception as exc:
                log_background_exception(task_name, exc)
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception as on_error_exc:
                        log_background_exception(f"{task_name}:error-handler", on_error_exc)

        thread = threading.Thread(target=_run, daemon=daemon)
        thread.start()
