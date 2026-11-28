"""Compare the packaged LTX Radeon attention implementation with full PyTorch attention.

Run with --device cuda inside the bundled ROCm interpreter for a device-side check.
Without it, this is a kernel-independent regression test of the indexing and masks.
"""
import argparse
from importlib.metadata import version

import torch
from ltx_core.model.transformer.attention import RadeonSdpaAttention, automatic_attention
from ltx_core.model.video_vae.transformer.apply import resolve_attention_for_host
from ltx_core.model.video_vae.transformer.config import DiffVAEMode, NAttentionKind
from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d


def check_attention(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(183)
    batch, queries, keys, heads, width = 2, 9, 13, 2, 8
    q = torch.randn(batch, queries, heads * width, dtype=dtype, device=device)
    k = torch.randn(batch, keys, heads * width, dtype=dtype, device=device)
    v = torch.randn_like(k)
    attention = RadeonSdpaAttention()
    attention.score_elements = 3 * batch * heads * keys
    # Three query tiles. A key-only mask must broadcast on each one; pairwise
    # masks must use the pairwise rows for that tile, including the batch case.
    for shape in (None, (1, keys), (queries, keys), (batch, queries, keys), (1, heads, queries, keys)):
        mask = None if shape is None else torch.randn(shape, dtype=dtype, device=device)
        actual = attention(q, k, v, heads, mask)
        qh, kh, vh = (t.float().reshape(batch, -1, heads, width).transpose(1, 2) for t in (q, k, v))
        logits = qh @ kh.transpose(-1, -2) * width ** -0.5
        if mask is not None:
            reference_mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
            logits = logits + reference_mask.float()
        expected = (logits.softmax(-1) @ vh).transpose(1, 2).reshape_as(q)
        torch.testing.assert_close(actual.float(), expected, atol=0.023 if dtype == torch.bfloat16 else 1e-5,
                                   rtol=0.023 if dtype == torch.bfloat16 else 1e-5)
        assert actual.device == q.device


def check_neighborhood(device: torch.device) -> None:
    torch.manual_seed(5331)
    shape = (1, 2, 3, 4, 2, 8)
    q, k, v = [torch.randn(shape, device=device) for _ in range(3)]
    actual = na3d(q, k, v, kernel_size=[1, 2, 3], scale=1.0)
    expected = torch.empty_like(actual)
    for t in range(shape[1]):
        for h in range(shape[2]):
            for w in range(shape[3]):
                sh = min(max(0, h - 1), shape[2] - 2)
                sw = min(max(0, w - 1), shape[3] - 3)
                kh, vh = [x[:, t, sh:sh+2, sw:sw+3].reshape(1, 6, 2, 8) for x in (k, v)]
                scores = (kh * q[:, t, h, w].unsqueeze(1)).sum(-1)
                expected[:, t, h, w] = (scores.softmax(1).unsqueeze(-1) * vh).sum(1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    device = torch.device(parser.parse_args().device)
    if device.type == 'cuda':
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("PyTorch on ROCm with a visible Radeon device is required.")
        print(f"Device: {torch.cuda.get_device_name(device)}; HIP: {torch.version.hip}")
        assert isinstance(automatic_attention(), RadeonSdpaAttention)
        effective = resolve_attention_for_host(DiffVAEMode.CHUNKED_EAGER.resolve())
        assert effective.attention == NAttentionKind.EAGER_SDPA and not effective.compile_blocks
    check_attention(device, torch.float32)
    check_attention(device, torch.bfloat16)
    check_neighborhood(device)
    print(f"PASS: masked and unmasked, FP32 and BF16 denoising and DiffVAE neighborhood attention. "
          f"device={device}; ltx-core={version('ltx-core')}; torch={torch.__version__}")


if __name__ == '__main__':
    main()
