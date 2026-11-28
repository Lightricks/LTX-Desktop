import ast
import zipfile
from pathlib import Path

import pytest
import torch

from services.amd_h3_mps import prepare_h3_mps_core


def test_pinned_metal_worker_padding_preserves_all_frames_and_channels(tmp_path):
    archive = Path(__file__).resolve().parents[3] / "vendor/ComfyUI/comfyui-v0.38.0.zip"
    if not archive.exists():
        pytest.skip("The pinned source archive is restored by restore_sources.py")
    relative = Path("comfy/ldm/minimax/vae.py")
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        raw = bundle.read(next(n for n in bundle.namelist() if n.endswith("/" + relative.as_posix())))
    source.write_bytes(raw)
    prepare_h3_mps_core(tmp_path)
    tree = ast.parse(source.read_bytes())
    layer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CausalConv3d")
    padding = next(node for node in ast.walk(layer) if isinstance(node, ast.If) and ast.unparse(node.test) == "front")
    executable = compile(ast.Module(body=padding.body, type_ignores=[]), str(source), "exec")
    # Execute the actual transformed statements at a resolution large enough to
    # trigger pytorch#194922. On a Metal host also compare its result to CPU.
    expected = torch.arange(2 * 4 * 240 * 480, dtype=torch.float32).reshape(1, 2, 4, 240, 480)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    namespace = {"torch": torch, "x": expected.to(device), "front": 2}
    exec(executable, namespace)
    assert torch.equal(namespace["x"].cpu(), torch.nn.functional.pad(expected, (0, 0, 0, 0, 2, 0)))

    source.write_bytes(raw + b"# different upstream source")
    with pytest.raises(ValueError, match="Unexpected"):
        prepare_h3_mps_core(tmp_path)
