"""Entry point for ComfyUI under the LTX Desktop embeddable Python interpreter."""

from __future__ import annotations

import runpy
import site
import sys
from pathlib import Path


def main() -> None:
    core, packages = (Path(arg).resolve(strict=True) for arg in sys.argv[1:3])
    # The Windows embedded interpreter ignores PYTHONPATH. Its site initialization
    # has already registered the installed torch runtime's DLL directories. Enable the private
    # wheels' .pth initialization and give Comfy its own PyAV before any imports.
    package_path = str(packages)
    sys.path[:0] = [str(core), package_path]
    site.addsitedir(package_path)
    entry = str(core / "main.py")
    sys.argv = [entry, *sys.argv[3:]]
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
