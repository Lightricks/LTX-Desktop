"""Export the production DiffVAE CuTe DSL AOT bundle."""

from __future__ import annotations

import argparse
import sys
import types
from collections.abc import Callable
from pathlib import Path


def _load_exporter() -> Callable[..., None]:
    """Import the exporter without executing ``ltx_kernels/__init__.py``.
    This script runs before the wheel exists, so the package's eager ``All2All``
    import would require the unbuilt ``all2all_cpp`` extension. Seeding the
    package module lets the VAE submodules load from this source tree on their own.
    """
    package_dir = Path(__file__).resolve().parents[1] / "src" / "ltx_kernels"
    package = types.ModuleType("ltx_kernels")
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    package.__package__ = "ltx_kernels"
    package.__file__ = str(package_dir / "__init__.py")
    sys.modules["ltx_kernels"] = package
    from ltx_kernels.vae.aot import export_production_artifacts  # noqa: PLC0415

    return export_production_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-arch", required=True)
    parser.add_argument("--host-target", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_production_artifacts = _load_exporter()
    export_production_artifacts(
        args.output_dir,
        gpu_arch=args.gpu_arch,
        host_target=args.host_target,
    )


if __name__ == "__main__":
    main()
