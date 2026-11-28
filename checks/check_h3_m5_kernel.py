"""Run a model-free H3 INT8 test with the installed LTX private Mac interpreter.

In Terminal, from the extracted online installer directory, run

    "$HOME/Applications/LTX Desktop - SpaceSpeak.app/Contents/Resources/python/bin/ltx-comfy-python" -I -B check_h3_m5_kernel.py

Set up the private video engine in the app first. The check reads its installed
libraries and runs a small synthetic H3 layer. No model download is needed.
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import platform
import re
import site
import statistics
import subprocess
import sys
import time


def exercise_projection(comfy_ops, model_class, runtime):
    import torch
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    # Use the real model constructor so a missing H3-only hook fails the test.
    # No other H3 layer or real checkpoint is needed for this projection.
    rows, width, channels, group = 129, 512, 128, 256
    dtype = torch.bfloat16
    ops = comfy_ops.mixed_precision_ops(compute_dtype=dtype, disabled=["int8_tensorwise"])
    model = model_class(hidden_size=channels, num_layers=0, token_refiner_num_layers=0,
                        num_attention_heads=2, attention_head_dim=64, ffn_hidden_size=256,
                        text_dim=width, time_embed_hidden_size=128, time_embed_dim=128,
                        device="cpu", dtype=dtype, operations=ops)
    projection = model.condition_proj
    rng = torch.Generator(device="cpu").manual_seed(41273)
    weight = torch.randint(-4, 5, (channels, width), dtype=torch.int8, generator=rng)
    scale = torch.tensor([0.5, 0.25, 0.125, 0.0625]).repeat(channels // 4)
    bias = (torch.arange(channels).float() % 9 - 4) / 8
    metadata = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
    projection.load_state_dict({
        "weight": weight, "weight_scale": scale.view(channels, 1), "bias": bias,
        "comfy_quant": torch.tensor(list(json.dumps(metadata).encode()), dtype=torch.uint8),
    })
    projection.to("mps")
    # A sparse signed pulse in each group has an exactly representable rotation.
    # Vary rows, scales and bias to catch a swapped axis or broken epilogue.
    # Row zero is silent and row 128 exercises the final partial tile.
    indices = torch.arange(rows)
    amplitudes = torch.tensor([0.0, 1.0, -2.0, 4.0])[indices % 4]
    amplitudes[-1] = 1
    activations = torch.zeros(rows, width)
    activations[indices, (indices * 23) % group] = amplitudes
    activations[indices, group + (indices * 41 + 3) % group] = -amplitudes
    rotation = _build_hadamard(group, device="cpu", dtype=torch.float32)
    rotated = (activations.reshape(rows, width // group, group) @ rotation).reshape(rows, width)
    expected = ((rotated @ weight.float().T) * scale).to(dtype) + bias.to(dtype)

    before = runtime.native_calls
    with torch.inference_mode():
        actual = projection(activations.reshape(1, rows, width).to(device="mps", dtype=dtype))
    torch.mps.synchronize()
    if runtime.native_calls <= before or actual.device.type != "mps":
        raise RuntimeError("The H3 projection used a fallback. No M5 INT8 model dispatch was observed.")
    torch.testing.assert_close(actual[0].cpu(), expected, rtol=0.01, atol=0.015625)
    return type(projection), ops.Linear


def benchmark_projection(native_class, stock_class, runtime):
    import torch

    # MiniMaxH3Model's default attention projects 56 heads × 128 back to 5376.
    # Time the actual wrapped/stock Linear (rotation and scales included).
    tokens, width, channels = 512, 7168, 5376
    dtype = torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(41274)
    metadata = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
    state = {"weight": torch.randint(-127, 128, (channels, width), generator=generator, dtype=torch.int8),
             "weight_scale": torch.full((channels, 1), 1 / 128),
             "comfy_quant": torch.tensor(list(metadata), dtype=torch.uint8)}
    native, stock = [cls(width, channels, bias=False, dtype=dtype, device="cpu")
                     for cls in (native_class, stock_class)]
    for layer in (native, stock):
        layer.load_state_dict(state)
        layer.to("mps")
    del state
    x = torch.randn((tokens, width), generator=generator).to(device="mps", dtype=dtype)
    print(f"Timing one synthetic H3 attention projection ({tokens} tokens, {width} → {channels}) …", flush=True)
    native_times, stock_times = [], []
    with torch.inference_mode():
        early = runtime.native_calls
        n = native(x)
        s = stock(x)
        torch.mps.synchronize()
        if runtime.native_calls <= early or not torch.isfinite(n).all() or not torch.isfinite(s).all():
            raise RuntimeError("The benchmark warmup failed to execute both paths with finite output.")
        for index in range(3):
            order = [(native, native_times), (stock, stock_times)]
            for layer, timings in order if index % 2 == 0 else reversed(order):
                before = runtime.native_calls
                torch.mps.synchronize()
                start = time.perf_counter()
                layer(x)
                torch.mps.synchronize()
                timings.append(time.perf_counter() - start)
                if (runtime.native_calls > before) != (layer is native):
                    raise RuntimeError("Native INT8 dispatch counters did not match the benchmark path.")
    native_ms = statistics.median(native_times) * 1000
    stock_ms = statistics.median(stock_times) * 1000
    print(f"Median of three warm runs: stock {stock_ms:.1f} ms, M5 INT8 {native_ms:.1f} ms "
          f"(stock time ÷ M5 time = {stock_ms / native_ms:.2f}).")
    print("These timings cover a single synthetic layer. Full video timings also include other layers, "
          "prompt encoding and decoding.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine-root", type=Path,
                        default=Path.home() / "Library/Application Support/LTXDesktopSpaceSpeak/managed-comfy")
    parser.add_argument("--skip-other-macs", action="store_true",
                        help="Skip on Macs without an M5-series chip, for general installer checks.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Compare one H3-sized projection against the stock MPS path after the strict check.")
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("Run this check on native Apple silicon macOS.")
    chip = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if args.skip_other_macs and not re.search(r"\bM5\b", chip, re.IGNORECASE):
        print(f"Skipped M5-specific dispatch check on {chip}.")
        return
    resources = Path(sys.executable).resolve().parents[2]
    sys.path.insert(0, str(resources / "backend"))
    from services.amd_comfy_provision import ManagedComfyLayout, needs_install, private_comfy_command

    layout = ManagedComfyLayout(args.engine_root, resources, "mps", "darwin")
    if Path(sys.executable).resolve() != layout.python_exe or not (resources / "comfy-engine").is_dir():
        raise SystemExit("Use the installed application's python/bin/ltx-comfy-python interpreter.")
    if needs_install(layout):
        raise SystemExit("Prepare the updated private video engine in LTX Desktop before running this check.")

    # Use the installed worker's paths, inherited settings and registered extension.
    # General Comfy Desktop and Python packages supplied by the shell are excluded.
    command, env, cwd = private_comfy_command(layout, 8188)
    os.environ.clear()
    os.environ.update(env)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.chdir(cwd)
    sys.argv = [str(layout.main_py), *command[command.index(str(layout.extra_packages)) + 1:]]
    sys.path[:0] = [str(layout.core_dir), str(layout.extra_packages)]
    site.addsitedir(str(layout.extra_packages))
    logging.basicConfig(level=logging.INFO)

    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes

    path = Path(env["LTX_COMFY_H3_METAL_NODE"])
    if not asyncio.run(nodes.load_custom_node(str(path), module_parent="comfy_extras")):
        raise RuntimeError(f"ComfyUI could not load the installed H3 Metal module at {path}.")
    module = sys.modules[str(path).replace(".", "_x_")]
    runtime = module.get_runtime()
    if runtime is None:
        raise RuntimeError(f"Native M5 INT8 is unavailable on {chip}. See the preceding kernel diagnostic.")
    import torch
    from comfy import ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    print(f"Testing H3 on {chip}, macOS {platform.mac_ver()[0]}, PyTorch {torch.__version__}.", flush=True)
    native_class, stock_class = exercise_projection(ops, MiniMaxH3Model, runtime)
    print("PASS · H3 ConvRot projection dispatched M5 INT8 on Metal and matched the CPU reference.")
    if args.benchmark:
        benchmark_projection(native_class, stock_class, runtime)


if __name__ == "__main__":
    main()
