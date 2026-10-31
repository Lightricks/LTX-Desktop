#!/usr/bin/env python3
"""Stock MPS Conv3d vs the per-tap conv2d patch (services/patches/mps_conv3d_as_conv2d).

Two checks, both on Apple Silicon only:

- Default: one 3x3x3 Conv3d per conv VAE decoder stage, bf16, in the decoder's
  ``channels_last_3d`` layout and in contiguous layout. Prints time, TFLOPS,
  speedup, and the largest difference from stock. No model weights needed.
- ``--vae``: decodes one 1024x576, 121-frame latent with the real conv VAE
  (``ltx-2.5-video-vae-conv``), stock then patched, the way the pipeline builds
  it. Prints both decode times and the PSNR of patched vs stock frames.

Use it to check the patch on your Mac and after every torch bump. Remove the
patch when stock is as fast in the ``channels_last_3d`` rows.

Usage (from backend/, no backend running):
    uv run python performance_runner/bench_mps_conv3d.py
    uv run python performance_runner/bench_mps_conv3d.py --vae
    uv run python performance_runner/bench_mps_conv3d.py --vae --checkpoint /path/to/ltx-2.5-video-vae-conv-bf16.safetensors
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import services.patches.mps_conv3d_as_conv2d as patch  # noqa: E402

# One Conv3d per decoder stage: (channels, frames, height, width) for a 1024x576 decode.
SHAPES = [(1024, 16, 18, 32), (512, 32, 36, 64), (256, 64, 72, 128), (128, 32, 144, 256)]
DEFAULT_CHECKPOINT = (
    Path.home() / "Library/Application Support/LTXDesktop/models/ltx-2.5/ltx-2.5-video-vae-conv-bf16.safetensors"
)
LATENT_SHAPE = (1, 128, 16, 18, 32)  # stage-2 latent for 1024x576, 121 frames
WARMUP, ITERS = 2, 5


def _time(fn: Callable[[], torch.Tensor]) -> float:
    for _ in range(WARMUP):
        fn()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) / ITERS


@torch.inference_mode()
def bench_ops() -> None:
    torch.manual_seed(0)
    print(f"{'layout':17} {'C':>5} {'T':>3} {'H':>4} {'W':>4} | {'stock ms':>9} {'TFLOPS':>6} | {'patch ms':>9} {'TFLOPS':>6} | speedup  max|d|")
    for layout in (torch.channels_last_3d, torch.contiguous_format):
        for c, t, h, w in SHAPES:
            conv = nn.Conv3d(c, c, 3, padding=(0, 1, 1)).to("mps", torch.bfloat16).to(memory_format=layout)
            x = torch.randn(1, c, t + 2, h, w, device="mps", dtype=torch.bfloat16).contiguous(memory_format=layout)
            flops = 2 * c * c * 27 * t * h * w
            t_stock = _time(lambda: patch._original_forward(conv, x))
            t_patch = _time(lambda: patch._patched_forward(conv, x))
            diff = (patch._patched_forward(conv, x).float() - patch._original_forward(conv, x).float()).abs().max()
            name = "channels_last_3d" if layout is torch.channels_last_3d else "contiguous"
            print(
                f"{name:17} {c:>5} {t:>3} {h:>4} {w:>4} | {t_stock * 1e3:>9.1f} {flops / t_stock / 1e12:>6.2f} | "
                f"{t_patch * 1e3:>9.1f} {flops / t_patch / 1e12:>6.2f} | {t_stock / t_patch:>6.2f}x  {diff.item():.4f}",
                flush=True,
            )
            del conv, x
            torch.mps.empty_cache()


@torch.inference_mode()
def _decode(checkpoint: Path, latent: torch.Tensor) -> tuple[torch.Tensor, float]:
    from ltx_pipelines.utils.blocks import VideoDecoder
    from ltx_pipelines.utils.helpers import tiling_config_for_vae

    device = torch.device("mps")
    tiling = tiling_config_for_vae(str(checkpoint), height=576, width=1024, num_frames=121, device=device)
    decoder = VideoDecoder(str(checkpoint), torch.bfloat16, device)
    generator = torch.Generator(device="mps").manual_seed(0)
    torch.mps.synchronize()
    start = time.perf_counter()
    frames = torch.cat([chunk.float().cpu() for chunk in decoder(latent, tiling, generator)], dim=0)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    torch.mps.empty_cache()
    return frames, elapsed


def bench_vae(checkpoint: Path) -> None:
    if not checkpoint.exists():
        sys.exit(f"Conv VAE checkpoint not found: {checkpoint}\nDownload it in the app or pass --checkpoint.")
    torch.manual_seed(0)
    latent = torch.randn(LATENT_SHAPE).to("mps", torch.bfloat16)
    original = nn.Conv3d.forward
    try:
        nn.Conv3d.forward = original  # type: ignore[method-assign]
        stock, t_stock = _decode(checkpoint, latent)
        print(f"stock   decode {t_stock:6.1f}s", flush=True)
        nn.Conv3d.forward = patch._patched_forward  # type: ignore[method-assign]
        patched, t_patch = _decode(checkpoint, latent)
        print(f"patched decode {t_patch:6.1f}s  ({t_stock / t_patch:.2f}x)", flush=True)
    finally:
        nn.Conv3d.forward = original  # type: ignore[method-assign]
    if not torch.isfinite(patched).all():
        sys.exit("FAIL: patched output has NaN or Inf values")
    mse = (patched - stock).pow(2).mean().item()
    span = (stock.max() - stock.min()).item()
    psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(span**2 / mse)).item()
    print(f"patched vs stock: PSNR {psnr:.1f} dB, max|d| {(patched - stock).abs().max().item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", action="store_true", help="also decode with the real conv VAE (needs the checkpoint)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available. This benchmark needs an Apple Silicon Mac.")
    print(f"torch {torch.__version__}, macOS {platform.mac_ver()[0]}, {platform.machine()}, bf16\n")
    bench_ops()
    if args.vae:
        print()
        bench_vae(args.checkpoint)


if __name__ == "__main__":
    main()
