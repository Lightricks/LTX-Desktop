"""Rebuild and verify the two LTX Python wheels used by the Windows Radeon installer."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

SOURCE_ROOT = Path(__file__).resolve().parents[1]
WHEELS = {
    "ltx-core": (
        "ltx_core-1.6.0+radeon1-py3-none-any.whl",
        "c5e53be1394b97607a0d29a68eb566558819ce239e98fc882aeefe6db9cb7c44",
    ),
    "ltx-pipelines": (
        "ltx_pipelines-1.6.0-py3-none-any.whl",
        "909bee1394edf069ef0c810c3396f5ac9bc2daea9cb706b367d870bc858612fb",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    target = parser.parse_args().out_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target) as temp:
        staging = Path(temp)
        for package, (name, expected) in WHEELS.items():
            subprocess.run(
                ["uv", "build", "--wheel", str(SOURCE_ROOT / "packages" / package), "--out-dir", str(staging)],
                check=True,
            )
            actual = hashlib.sha256((staging / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"{name}: SHA-256 {actual}, expected {expected}. "
                                   "Use uv 0.13.9 and a checkout with LF source line endings.")
        for name, expected in WHEELS.values():
            (staging / name).replace(target / name)
            print(f"{expected}  {name}")


if __name__ == "__main__":
    main()
