"""Import every Desktop ltx-2 patch.

Catches import-time AssertionError (the 1.3.0 recommended_tiling_config class
of boot failure) without starting the FastAPI server.
"""

from __future__ import annotations

import importlib
from pathlib import Path

PATCHES_DIR = Path(__file__).resolve().parents[1] / "services" / "patches"


def test_every_patch_module_imports() -> None:
    modules = sorted(p.stem for p in PATCHES_DIR.glob("*.py") if p.stem != "__init__")
    assert modules, "expected services/patches/*.py modules"
    for name in modules:
        importlib.import_module(f"services.patches.{name}")
