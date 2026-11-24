#!/usr/bin/env python3
"""Stock MPS Conv3d vs the per-tap conv2d patch (services/patches/mps_conv3d_as_conv2d).

Two checks, both on Apple Silicon only:

- Default: one 3x3x3 Conv3d per conv VAE decoder stage, bf16, in the decoder's
  ``channels_last_3d`` layout and in contiguous layout. Reports time, TFLOPS,
  speedup, and the largest difference from stock. No model weights needed.
- ``--vae``: decodes one 1024x576, 121-frame latent with the real conv VAE
  (``ltx-2.5-video-vae-conv``), stock then patched, the way the pipeline builds
  it. Reports both decode times and the PSNR of patched vs stock frames.

Measurement and reporting are separate: the ``bench_*`` functions return results
and ``report`` prints them once at the end.

Use it to check the patch on your Mac and after every torch bump. Remove the
patch when stock is as fast in the ``channels_last_3d`` rows.

Credit: written by Sabbagh Fares (@Sabmit) in https://github.com/Lightricks/LTX-Desktop/pull/189

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
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import services.patches.mps_conv3d_as_conv2d as patch  # noqa: E402

SEED = 42  # any fixed value; makes runs repeatable across machines and torch versions

# What the ``--vae`` check decodes: the app's default 540p, 5 s clip.
HEIGHT, WIDTH, NUM_FRAMES = 576, 1024, 121
# LTX-2 video VAE: 32x spatial and 8x temporal compression, 128 latent channels.
SPATIAL_SCALE, TEMPORAL_SCALE, LATENT_CHANNELS = 32, 8, 128
LATENT_SHAPE = (
    1,
    LATENT_CHANNELS,
    (NUM_FRAMES - 1) // TEMPORAL_SCALE + 1,
    HEIGHT // SPATIAL_SCALE,
    WIDTH // SPATIAL_SCALE,
)

# One Conv3d per decoder stage: (channels, frames, height, width) for the clip above.
SHAPES = [(1024, 16, 18, 32), (512, 32, 36, 64), (256, 64, 72, 128), (128, 32, 144, 256)]
KERNEL_TAPS = 3 * 3 * 3  # a 3x3x3 kernel: 27 multiply-adds per output value per input channel
FLOPS_PER_MAC = 2  # one multiply and one add

DEFAULT_CHECKPOINT = (
    Path.home() / "Library/Application Support/LTXDesktop/models/ltx-2.5/ltx-2.5-video-vae-conv-bf16.safetensors"
)
WARMUP, ITERS = 2, 5


@dataclass(frozen=True)
class OpResult:
    layout: str
    channels: int
    frames: int
    height: int
    width: int
    stock_s: float
    patched_s: float
    flops: int
    max_abs_diff: float


@dataclass(frozen=True)
class VaeResult:
    stock_s: float
    patched_s: float
    psnr_db: float
    max_abs_diff: float
    finite: bool


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
def bench_ops() -> list[OpResult]:
    torch.manual_seed(SEED)
    results: list[OpResult] = []
    for layout in (torch.channels_last_3d, torch.contiguous_format):
        for channels, frames, height, width in SHAPES:
            conv = nn.Conv3d(channels, channels, 3, padding=(0, 1, 1)).to("mps", torch.bfloat16)
            conv = conv.to(memory_format=layout)
            # No temporal padding: the 3-tap kernel turns frames + 2 input frames into frames output frames.
            x = torch.randn(1, channels, frames + 2, height, width, device="mps", dtype=torch.bfloat16)
            x = x.contiguous(memory_format=layout)
            stock_s = _time(lambda: patch._original_forward(conv, x))
            patched_s = _time(lambda: patch._conv3d_as_conv2d(conv, x))
            diff = (patch._conv3d_as_conv2d(conv, x).float() - patch._original_forward(conv, x).float()).abs().max()
            results.append(
                OpResult(
                    layout="channels_last_3d" if layout is torch.channels_last_3d else "contiguous",
                    channels=channels,
                    frames=frames,
                    height=height,
                    width=width,
                    stock_s=stock_s,
                    patched_s=patched_s,
                    # in_channels == out_channels in every decoder stage, hence channels squared.
                    flops=FLOPS_PER_MAC * channels * channels * KERNEL_TAPS * frames * height * width,
                    max_abs_diff=diff.item(),
                )
            )
            del conv, x
            torch.mps.empty_cache()
    return results


@torch.inference_mode()
def _decode(checkpoint: Path, latent: torch.Tensor) -> tuple[torch.Tensor, float]:
    from ltx_pipelines.utils.blocks import VideoDecoder
    from ltx_pipelines.utils.helpers import tiling_config_for_vae

    device = torch.device("mps")
    tiling = tiling_config_for_vae(str(checkpoint), height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES, device=device)
    decoder = VideoDecoder(str(checkpoint), torch.bfloat16, device)
    generator = torch.Generator(device="mps").manual_seed(SEED)
    torch.mps.synchronize()
    start = time.perf_counter()
    frames = torch.cat([chunk.float().cpu() for chunk in decoder(latent, tiling, generator)], dim=0)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    torch.mps.empty_cache()
    return frames, elapsed


def bench_vae(checkpoint: Path) -> VaeResult:
    torch.manual_seed(SEED)
    latent = torch.randn(LATENT_SHAPE).to("mps", torch.bfloat16)
    original = nn.Conv3d.forward
    was_enabled = patch._enabled
    try:
        nn.Conv3d.forward = original  # type: ignore[method-assign]
        stock, stock_s = _decode(checkpoint, latent)
        nn.Conv3d.forward = patch._forward  # type: ignore[method-assign]
        patch._enabled = True
        patched, patched_s = _decode(checkpoint, latent)
    finally:
        nn.Conv3d.forward = original  # type: ignore[method-assign]
        patch._enabled = was_enabled
    mse = (patched - stock).pow(2).mean().item()
    span = (stock.max() - stock.min()).item()
    psnr_db = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(span**2 / mse)).item()
    return VaeResult(
        stock_s=stock_s,
        patched_s=patched_s,
        psnr_db=psnr_db,
        max_abs_diff=(patched - stock).abs().max().item(),
        finite=bool(torch.isfinite(patched).all()),
    )


def report(op_results: list[OpResult], vae: VaeResult | None) -> None:
    print(f"torch {torch.__version__}, macOS {platform.mac_ver()[0]}, {platform.machine()}, bf16\n")
    print(
        f"{'layout':17} {'C':>5} {'T':>3} {'H':>4} {'W':>4} | {'stock ms':>9} {'TFLOPS':>6} | "
        f"{'patch ms':>9} {'TFLOPS':>6} | speedup  max|d|"
    )
    for row in op_results:
        print(
            f"{row.layout:17} {row.channels:>5} {row.frames:>3} {row.height:>4} {row.width:>4} | "
            f"{row.stock_s * 1e3:>9.1f} {row.flops / row.stock_s / 1e12:>6.2f} | "
            f"{row.patched_s * 1e3:>9.1f} {row.flops / row.patched_s / 1e12:>6.2f} | "
            f"{row.stock_s / row.patched_s:>6.2f}x  {row.max_abs_diff:.4f}"
        )
    if vae is None:
        return
    print()
    print(f"stock   decode {vae.stock_s:6.1f}s")
    print(f"patched decode {vae.patched_s:6.1f}s  ({vae.stock_s / vae.patched_s:.2f}x)")
    print(f"patched vs stock: PSNR {vae.psnr_db:.1f} dB, max|d| {vae.max_abs_diff:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", action="store_true", help="also decode with the real conv VAE (needs the checkpoint)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available. This benchmark needs an Apple Silicon Mac.")
    if args.vae and not args.checkpoint.exists():
        sys.exit(f"Conv VAE checkpoint not found: {args.checkpoint}\nDownload it in the app or pass --checkpoint.")
    op_results = bench_ops()
    vae = bench_vae(args.checkpoint) if args.vae else None
    report(op_results, vae)
    if vae is not None and not vae.finite:
        sys.exit("FAIL: patched output has NaN or Inf values")


if __name__ == "__main__":
    main()
