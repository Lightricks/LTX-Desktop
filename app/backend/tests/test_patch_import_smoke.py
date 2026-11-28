"""Import every Desktop ltx-2 patch.

Catches import-time AssertionError (the 1.3.0 recommended_tiling_config class
of boot failure) without starting the FastAPI server.
"""

from __future__ import annotations

import importlib
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_PATCH_PACKAGES = (
    (_BACKEND_DIR / "inference" / "patches", "inference.patches"),
    (_BACKEND_DIR / "services" / "patches", "services.patches"),
)


def test_every_patch_module_imports() -> None:
    for patches_dir, package in _PATCH_PACKAGES:
        modules = sorted(p.stem for p in patches_dir.glob("*.py") if p.stem != "__init__")
        assert modules, f"expected {package}/*.py modules"
        for name in modules:
            importlib.import_module(f"{package}.{name}")
