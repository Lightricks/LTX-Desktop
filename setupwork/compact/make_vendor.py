"""Produce the small bootstrap's pinned vendor manifest and Electron launcher patch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import zipfile

from vendor_materialize import _sha256, hydrate

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
ELECTRON_SHA = "dcd36396a606a5ae2f5651b4ee6bb463a624dbf15f786eda57cee2cc361c138c"
SOURCES = {
    "comfyui-v0.38.0.zip": (
        "https://codeload.github.com/Comfy-Org/ComfyUI/zip/6b747c0428c343e1417219641db93a4fb7cb69ae",
        "4a144a553c77397c81f6067f3f8c8ab9dffd42889af48fb35fec2c64a71c75b2"
    ),
    "kandinsky6-1.0.2.zip": (
        "https://cdn.comfy.org/kandinskylab/kandinsky6/1.0.2/node.zip",
        "5ddf0bef14d0e571f8760403d270f2b1d6f9b6b10e56b8b707743c15751b7f80"
    ),
}


def record(path, name, url, sha):
    if _sha256(path) != sha:
        raise ValueError(f"Invalid release build input: {name}")
    return dict(filename=name, url=url, size=path.stat().st_size, sha256=sha)


def build(repo: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shellwork = repo / "shellwork"
    shell = shellwork / "win-x64"
    stock = shellwork / "electron.zip"
    engine = shell / "resources" / "comfy-engine"
    name = "electron-v41.0.2-win32-x64.zip"
    url = f"https://github.com/electron/electron/releases/download/v41.0.2/{name}"
    electron = record(stock, name, url, ELECTRON_SHA)
    product = json.loads((repo / "app/package.json").read_bytes())["productName"]
    launcher = shell / (product + ".exe")
    patch_path = out / "vendor-launcher.zip"
    with zipfile.ZipFile(stock) as jar, jar.open("electron.exe") as old, launcher.open("rb") as new:
        with zipfile.ZipFile(patch_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as patch:
            offset = 0
            while data := new.read(4096):
                if old.read(len(data)) != data:
                    entry = zipfile.ZipInfo(str(offset))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    patch.writestr(entry, data, compresslevel=9)
                offset += len(data)
    manifest = {
        "electron": electron,
        "launcher": dict(filename=launcher.name, sha256=_sha256(launcher), size=launcher.stat().st_size,
                         patch_sha256=_sha256(patch_path)),
        "sources": [record(engine / name, name, url, sha) for name, (url, sha) in SOURCES.items()],
        "wheels": json.loads((repo / "runtimework/comfy-windows-wheels.json").read_bytes())["wheels"],
        "receipt": (engine / "site-packages/ltx-comfy-deps.json").read_bytes().decode("utf-8"),
    }
    (out / "vendor-lock.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    shutil.copyfile(HERE / "vendor_materialize.py", out / "vendor_materialize.py")


def owns(rel: str) -> bool:
    return "/" not in rel or rel.startswith(("locales/", "resources/comfy-engine/"))


def verify(repo: Path, res: Path, stage: Path) -> None:
    shell = repo / "shellwork/win-x64"
    def cached(item, _cache):
        name = item["filename"]
        if name.startswith("electron-"):
            return repo / "shellwork/electron.zip"
        if name in SOURCES:
            return shell / "resources/comfy-engine" / name
        return repo / "build/.cache/comfy-wheels" / name
    hydrate(stage, res, res.parent / "verify-cache", cached)
    got = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    expected = {rel for p in shell.rglob("*") if p.is_file() and owns(rel := p.relative_to(shell).as_posix())}
    if expected != got:
        raise ValueError(f"Vendor file set mismatched: extra={got-expected}, missing={expected-got}")
    for rel in expected:
        if _sha256(stage / rel) != _sha256(shell / rel):
            raise ValueError(f"Vendor file bytes mismatched: {rel}")
    print(f"Verified every byte of {len(got)} original vendor shell files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=HERE / "generated")
    parser.add_argument("--verify", type=Path, metavar="EMPTY_STAGE")
    args = parser.parse_args()
    build(args.repo, args.out)
    if args.verify:
        verify(args.repo, args.out, args.verify)
