"""Small helpers for atomically persisting files that contain secrets."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def harden_file_permissions(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows and some mounted filesystems do not expose POSIX modes.
        pass


def secure_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        harden_file_permissions(temporary_path)
        temporary_path.replace(path)
        harden_file_permissions(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
