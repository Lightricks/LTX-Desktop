"""Startup migration helpers for model storage layout changes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_legacy_models_layout(app_data_dir: Path) -> None:
    """Move legacy models/ltx-2/* into models/* without overwriting existing targets."""
    models_root = app_data_dir / "models"
    legacy_root = models_root / "ltx-2"
    if not legacy_root.exists():
        return

    models_root.mkdir(parents=True, exist_ok=True)

    for source in legacy_root.iterdir():
        target = models_root / source.name
        if target.exists():
            logger.warning("Skipping legacy model path %s because target already exists at %s", source, target)
            continue
        try:
            source.rename(target)
        except OSError:
            # Cross-device or platform-specific rename failures should not block startup.
            logger.warning("Rename failed for legacy model path %s -> %s, falling back to move()", source, target, exc_info=True)
            shutil.move(str(source), str(target))

    try:
        if not any(legacy_root.iterdir()):
            legacy_root.rmdir()
    except Exception:
        logger.warning("Could not remove legacy models directory: %s", legacy_root, exc_info=True)
