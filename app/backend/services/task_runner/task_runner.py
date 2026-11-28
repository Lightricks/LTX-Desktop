"""Task runner service protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class TaskRunner(Protocol):
    def run_background(
        self,
        target: Callable[[], None],
        *,
        task_name: str,
        on_error: Callable[[Exception], None] | None = None,
        daemon: bool = True,
    ) -> None:
        ...
