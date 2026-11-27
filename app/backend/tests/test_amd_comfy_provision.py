"""Private ComfyUI uses release inputs and keeps each install separate from models."""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from services.amd_comfy_provision import (
    ARCHIVE_NAME, CORE_REVISION, ManagedComfyLayout, needs_install, private_comfy_command, provision_private_comfy,
)


def layout_with_payload(tmp_path: Path, backend="rocm") -> ManagedComfyLayout:
    layout = ManagedComfyLayout(tmp_path / "Desktop data", tmp_path / "Program Files" / "resources", backend)
    layout.asset_dir.mkdir(parents=True)
    release_zip = Path(__file__).resolve().parents[3] / "vendor" / "ComfyUI" / ARCHIVE_NAME
    shutil.copyfile(release_zip, layout.asset_dir / ARCHIVE_NAME)
    return layout


def test_concurrent_setup_and_source_repair_leave_weights_and_custom_nodes_in_place(tmp_path: Path):
    layout = layout_with_payload(tmp_path)
    assert needs_install(layout)
    with ThreadPoolExecutor(max_workers=2) as threads:
        setups = [threads.submit(provision_private_comfy, layout) for _ in range(2)]
        for setup in setups:
            assert setup.result() == layout
    assert not needs_install(layout)
    assert 'import comfy.options' in layout.main_py.read_text(encoding="utf-8")
    assert not (layout.core_dir / f"ComfyUI-{CORE_REVISION}").exists()
    weights = layout.models_dir / "test-model"
    weights.write_bytes(b"keep")
    node = layout.custom_nodes_dir / "user-data"
    node.mkdir()
    (layout.core_dir / "server.py").unlink()
    assert needs_install(layout)
    provision_private_comfy(layout)
    assert not needs_install(layout)
    assert weights.read_bytes() == b"keep" and node.is_dir()


def test_corrupt_packaged_archive_cannot_replace_existing_core(tmp_path: Path):
    layout = layout_with_payload(tmp_path)
    layout.core_dir.mkdir(parents=True)
    layout.main_py.write_text("existing core", encoding="ascii")
    with (layout.asset_dir / ARCHIVE_NAME).open("r+b") as damaged:
        damaged.seek(1024)
        damaged.write(b"X")
    with pytest.raises(ValueError, match="failed verification"):
        provision_private_comfy(layout)
    assert layout.main_py.read_text(encoding="ascii") == "existing core"
    assert not list(layout.core_dir.parent.glob(".setup-*"))


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_isolated_launch_resolves_private_modules_and_propagates_comfy_arguments(tmp_path: Path, backend, monkeypatch):
    layout = layout_with_payload(tmp_path, backend)
    visible = "GPU-00112233-4455-6677-8899-aabbccddeeff" if backend == "cuda" else "1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    provision_private_comfy(layout)
    layout.python_exe.parent.mkdir(parents=True)
    layout.python_exe.touch()
    layout.extra_packages.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="dependencies"):
        private_comfy_command(layout, 9087)
    (layout.extra_packages / "ltx-comfy-deps.json").write_text("{}", encoding="ascii")
    nested = layout.extra_packages / "private wheel"
    nested.mkdir()
    (nested / "comfy_bootstrap_fixture.py").write_text("value = 27", encoding="ascii")
    (layout.extra_packages / "initialize.pth").write_text("private wheel\n", encoding="ascii")
    layout.main_py.write_text(
        "import comfy_bootstrap_fixture, sys, os, json\n"
        "print(json.dumps([comfy_bootstrap_fixture.value, sys.argv, os.environ.get('HF_HUB_OFFLINE')]))",
        encoding="ascii",
    )
    argv, env, cwd = private_comfy_command(layout, 9087)
    assert Path(argv[0]) == layout.python_exe
    assert cwd == layout.core_dir
    assert Path(env["TEMP"]).is_dir() and Path(env["TEMP"]) != layout.temp_dir
    # Use the test-host Python with the production isolation flags and wrapper.
    # The Windows executable and ROCm libraries are validated on the target machine.
    if os.name != "nt":
        env["LC_ALL"] = "C.UTF-8"
    result = subprocess.run([sys.executable, *argv[1:]], env=env, cwd=cwd, check=True,
                            capture_output=True, text=True, timeout=20)
    value, args, offline = json.loads(result.stdout)
    assert value == 27 and offline == "1" and args[0] == str(layout.main_py)
    assert args[args.index("--listen") + 1] == "127.0.0.1"
    assert args[args.index("--base-directory") + 1] == str(layout.base_dir)
    assert args[args.index("--whitelist-custom-nodes") + 1] == "kandinsky6"
    assert ("--disable-dynamic-vram" in args) == (backend == "rocm")
    assert env["CUDA_VISIBLE_DEVICES"] == visible and "--enable-manager" not in args
    if backend == "cuda":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        with pytest.raises(RuntimeError, match="UUID"):
            private_comfy_command(layout, 9087)
