"""Stage the private ComfyUI Windows wheels on the release builder.

Usage: python runtimework/prepare_comfy_deps.py --out PATH/resources/comfy-engine/site-packages --cache PATH

Only the ComfyUI process adds this directory to its import path. The native LTX
process retains its original PyAV, AMD PyTorch, TorchVision and TorchAudio.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlsplit
import zipfile

from install_runtime import wheel_path
from prepare_runtime import acquire

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "comfy-windows-wheels.json"
BASE_MANIFEST = HERE / "runtime-manifest.json"
RECEIPT = "ltx-comfy-deps.json"


def package_name(filename: str) -> str:
    return filename.partition("-")[0].replace("_", "-").lower()


def compatible_tags(filename: str, python_version: str) -> bool:
    fields = filename.removesuffix(".whl").rsplit("-", 3)
    if len(fields) < 4:
        return False
    _, interpreter, abi, platform = fields
    if platform not in {"win_amd64", "any"}:
        return False
    target = int("".join(python_version.split(".")[:2]))
    if abi == "none":
        return "py3" in interpreter.split(".")
    if platform != "win_amd64" or not re.fullmatch(r"cp3[0-9]+", interpreter):
        return False
    return ((abi == "abi3" and int(interpreter[2:]) <= target)
            or (abi == interpreter and int(interpreter[2:]) == target))


def validate_lock(manifest: dict, base: dict) -> None:
    if manifest["python"] != base["python"] or manifest["target"] != "win_amd64":
        raise ValueError("Private ComfyUI wheel lock does not match embedded Windows Python")
    protected = {package_name(item["filename"]) for item in base["wheels"]} - {"av"}
    names = set()
    for item in manifest["wheels"]:
        filename = item["filename"]
        name = package_name(filename)
        if (not re.fullmatch(r"[\w.+-]+\.whl", filename, re.ASCII)
                or not compatible_tags(filename, manifest["python"])
                or urlsplit(item["url"]).hostname != "files.pythonhosted.org"
                or not item["url"].startswith("https://")
                or not re.fullmatch("[0-9a-f]{64}", item["sha256"])
                or item["size"] <= 0):
            raise ValueError(f"Invalid Windows wheel record: {filename}")
        if name in protected or name in names or name.startswith(("nvidia-", "rocm-", "triton")):
            raise ValueError(f"Private engine must not replace the protected distribution {name}")
        names.add(name)
    for required in ("av", "comfy-kitchen", "comfy-aimdo", "aiohttp", "librosa", "soundfile"):
        if required not in names:
            raise ValueError(f"Private engine lock is missing {required}")


def unpack(source: Path, site: Path) -> None:
    with zipfile.ZipFile(source) as wheel:
        for entry in wheel.infolist():
            if entry.is_dir():
                continue
            dest = wheel_path(entry.filename, site, site)
            # All installed files stay within this private prefix, including
            # native libraries, upstream LICENSE/NOTICE files and dist-info.
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise ValueError(f"Overlapping private wheel payload at {dest.relative_to(site)}")
            with wheel.open(entry) as src, dest.open("xb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)


def prepare(out: Path, cache: Path) -> dict:
    manifest = json.loads(MANIFEST.read_bytes())
    validate_lock(manifest, json.loads(BASE_MANIFEST.read_bytes()))
    out = out.resolve()
    cache = cache.resolve()
    if out.exists() and (not out.is_dir() or (any(out.iterdir()) and not (out / RECEIPT).is_file())):
        raise ValueError(f"A private wheel staging directory is required: {out}")
    cache.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "python": manifest["python"], "target": manifest["target"],
        "wheel_count": len(manifest["wheels"]),
    }
    with tempfile.TemporaryDirectory(prefix="comfy-wheels-", dir=out.parent) as work:
        site = Path(work) / "new"
        site.mkdir()
        with ThreadPoolExecutor(max_workers=6) as workers:
            for wheel in workers.map(lambda item: acquire(item, cache), manifest["wheels"]):
                unpack(wheel, site)
        (site / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        old = Path(work) / "old"
        if out.exists():
            out.rename(old)
        try:
            site.rename(out)
        except OSError:
            if old.exists():
                old.rename(out)
            raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=HERE.parent / "build" / ".cache" / "comfy-wheels")
    args = parser.parse_args()
    receipt = prepare(args.out, args.cache)
    print(f"Staged {receipt['wheel_count']} locked Windows engine wheels in {args.out}", flush=True)


if __name__ == "__main__":
    main()
