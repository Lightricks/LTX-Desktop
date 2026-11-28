# LTX inference source for the Radeon build

This directory accompanies the Lightricks LTX-2 v1.6.0 source at commit `c1b4675863c07c59c92e502446cccdaedbe5889c`. The modifications for this edition are in `packages/ltx-core`. The `1.6.0+radeon1` wheel uses tiled PyTorch HIP attention for the denoiser and eager neighborhood attention for the diffusion video decoder. The upstream LTX pipelines package remains at version 1.6.0. Its source and the upstream licenses are included here. [Upstream source](https://github.com/Lightricks/LTX-2/tree/c1b4675863c07c59c92e502446cccdaedbe5889c)

From the root of the Desktop repository, use Python 3 and uv 0.13.9 to rebuild the two pure-Python wheels. The script checks the results against the SHA-256 hashes of the wheels in the installer.

```text
python vendor/LTX-2/radeon/build_wheels.py --out-dir wheels
```

Run the numerical check with the Python interpreter from the installed runtime. It verifies FP32 and BF16 denoiser attention, broadcast and per-query masks, and the video decoder's neighborhood attention against independent reference calculations.

```text
python vendor/LTX-2/radeon/check_ops.py --device cuda
```

PyTorch uses the `cuda` device name for HIP. This command checks that PyTorch reports HIP, the GPU is accessible, and the LTX dispatcher selects the Radeon implementation. Use `--device cpu` for the kernel-independent numerical regression test.
