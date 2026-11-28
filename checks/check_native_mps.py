"""Verify local model eligibility and GPU execution with the packaged Mac interpreter."""

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys


def main() -> None:
    python_dir = Path(sys.executable).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=python_dir.parent / "backend",
                        help="Path to the bundled backend (defaults to Resources/backend)")
    args = parser.parse_args()

    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("Run this check with the community app’s Apple silicon macOS interpreter.")
    if not (args.backend / "runtime_config" / "runtime_policy.py").is_file():
        raise SystemExit(f"The packaged backend is missing from {args.backend}.")
    sys.path.insert(0, str(args.backend))
    # Exercise only native Metal operations, even when the shell sets a fallback.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    from runtime_config.runtime_policy import decide_local_generation_mode

    version = platform.mac_ver()[0]
    memsize = subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True)
    ram_gib = int(memsize.strip()) // (1024 ** 3)
    # Determine the release's OS/RAM eligibility separately from driver health.
    mode = decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None,
                                        mps_available=True, ram_gb=ram_gib, macos_major=int(version.split(".")[0]))
    if mode == "unsupported":
        print(f"Native app, API-only mode: macOS {version}, {ram_gib} GiB unified memory.")
        return
    if not torch.backends.mps.is_available():
        raise SystemExit(f"Metal (MPS) is unavailable to the native arm64 app on macOS {version}.")

    # Check the exact extension shipped for long video attention, without a JIT build.
    ext_dir = python_dir / "mps-ext-prebuilt" / "mps_sdpa_zc_ext"
    if not (ext_dir / "mps_sdpa_zc_ext.so").is_file():
        raise SystemExit(f"The packaged Metal attention extension is missing from {ext_dir}.")
    sys.path.insert(0, str(ext_dir))
    import mps_sdpa_zc_ext

    with torch.inference_mode():
        q = torch.ones((1, 2, 16, 64), dtype=torch.bfloat16, device="mps")
        product = q @ q.transpose(-1, -2)
        attention = mps_sdpa_zc_ext.sdpa_forward(q, q, q)
        torch.mps.synchronize()
        if product.device.type != "mps" or attention.device.type != "mps":
            raise SystemExit("The GPU operation returned a tensor on an unexpected device.")
        if not (product.float().cpu() == 64).all() or not (attention.float().cpu() == 1).all():
            raise SystemExit("The Metal BF16 matrix multiplication or zero-copy attention returned an incorrect result.")
    print(f"Metal GPU check passed: macOS {version}, {ram_gib} GiB unified memory, "
          f"PyTorch {torch.__version__}, {mode}.")


if __name__ == "__main__":
    main()
