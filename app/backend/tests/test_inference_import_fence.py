"""The Pod image copies only inference/ and installs without the desktop extra.

A sibling import breaks the copy. An import of fastapi, starlette, uvicorn, or
multipart breaks the install.
"""

from __future__ import annotations

import ast
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_ROOT = _BACKEND_ROOT / "inference"
_SKIP_DIRS = frozenset({"inference", "__pycache__", ".venv", ".pytest_cache"})
# Installed by `uv sync --extra desktop`. A Pod image omits that extra.
_DESKTOP_ONLY_ROOTS = frozenset({"fastapi", "starlette", "uvicorn", "multipart"})


def _sibling_roots() -> frozenset[str]:
    roots: set[str] = set()
    for path in _BACKEND_ROOT.iterdir():
        if path.name in _SKIP_DIRS or path.name.startswith("."):
            continue
        if path.is_dir() or path.suffix == ".py":
            roots.add(path.stem if path.suffix == ".py" else path.name)
    return frozenset(roots)


def test_inference_import_of_sibling_backend_module_fails() -> None:
    roots = _sibling_roots() | _DESKTOP_ONLY_ROOTS
    assert "services" in roots
    assert "fastapi" in roots
    leaks: list[str] = []
    for path in _INFERENCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(_INFERENCE_ROOT)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
                names = [node.module]
            for name in names:
                root = name.split(".", 1)[0]
                if root in roots:
                    leaks.append(f"{relative}: {name}")
    assert leaks == []
