"""Mac-only changes to the pinned ComfyUI v0.38 MiniMax H3 worker."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

# On MPS, constant temporal padding of a large 5-D tensor can silently corrupt
# the encoder latent. Explicit zero frames produce the same values on all devices.
# See pytorch/pytorch#194922 and Comfy-Org/ComfyUI#15902.
_VAE_SHA256 = "0168d55410794a2171c0507ca2dcb8a80e8cd288fa3e40b52fb460e4c92f3ffa"
_PAD = '                    x = F.pad(x, (0, 0, 0, 0, front, 0), mode="constant")'
_ZERO_FRAMES = """                    prefix = x.new_zeros((*x.shape[:2], front, *x.shape[3:]))
                    x = torch.cat((prefix, x), dim=2)"""

# Use Comfy's built-in node loader once its normal nodes are ready. The worker
# receives the exact node directory in the signed app, including after upgrades.
_NODES_SHA256 = "fe624b0c1b6a3cd124a2214e5700b61337589144c2f0668900a04f8ccc031e38"
_NODE_LOOP = b"    import_failed = []\n    for node_file in extras_files:"
_METAL_NODE_LOOP = (
    b'    if h3_metal := os.environ.get("LTX_COMFY_H3_METAL_NODE"):\n'
    b'        extras_files.append(h3_metal)\n\n' + _NODE_LOOP
)


def register_h3_metal_node(core: Path) -> bool:
    """Upgrade an existing private core in place. Unknown sources need fresh extraction."""
    source = core / "nodes.py"
    try:
        data = source.read_bytes()
    except FileNotFoundError:
        return False
    if _METAL_NODE_LOOP in data:
        upstream = data.replace(_METAL_NODE_LOOP, _NODE_LOOP, 1)
        return hashlib.sha256(upstream).hexdigest() == _NODES_SHA256
    if hashlib.sha256(data).hexdigest() != _NODES_SHA256 or data.count(_NODE_LOOP) != 1:
        return False
    with TemporaryDirectory(dir=core, prefix=".h3-nodes-") as staging:
        temporary = Path(staging) / "nodes.py"
        temporary.write_bytes(data.replace(_NODE_LOOP, _METAL_NODE_LOOP))
        temporary.replace(source)
    return True


def prepare_h3_mps_core(core: Path) -> None:
    """Apply only to a fresh, verified source tree before the worker is activated."""
    vae = core / "comfy" / "ldm" / "minimax" / "vae.py"
    raw = vae.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _VAE_SHA256:
        raise ValueError("Unexpected MiniMax H3 VAE source for the Apple silicon worker.")
    text = raw.decode("utf-8")
    if text.count(_PAD) != 1:
        raise ValueError("Cannot locate the MiniMax H3 temporal padding operation.")
    vae.write_text(text.replace(_PAD, _ZERO_FRAMES), encoding="utf-8")
