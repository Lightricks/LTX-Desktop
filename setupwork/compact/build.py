"""Build compact online installation media for the assembled AMD .8 Windows application."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from nsis import select_nsis
from build_assets import UPSTREAM, build as build_assets
from make_vendor import build as build_vendors


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(source, output):
    runtime = ROOT / "build" / "runtime"
    shell = ROOT / "shellwork" / "win-x64"
    work = ROOT / "build"
    icon = ROOT / "app/resources/icon.ico"
    nsis, env = select_nsis(None, work / ".cache/nsis")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work, prefix="compact-") as tmp:
        base = Path(tmp)
        support = base / "compact"
        support.mkdir()
        build_vendors(ROOT, support)
        build_assets(shell, source, support / "asset-payload.zip")
        for name in ("asset_materialize.py", "compact_setup.py"):
            shutil.copyfile(HERE / name, support / name)
        lock = json.loads((support / "vendor-lock.json").read_bytes())
        sources = [lock["electron"], *lock["sources"], *lock["wheels"], UPSTREAM]
        required = sum(record["size"] for record in sources)
        files = {path.relative_to(shell).as_posix(): (path.stat().st_size, file_digest(path))
                 for path in sorted(shell.rglob("*")) if path.is_file()}
        manifest = {"downloadBytes": required, "files": files}
        (support / "shell-inventory.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        src = HERE.parent / "src"
        generated = base / output.name
        subprocess.run([nsis, "-V2", f"-DOUT={generated}", f"-DSRC={src}", f"-DBUNDLE={runtime}",
                        f"-DCOMPACT={support}", f"-DICONFILE={icon}", str(src / "setup.nsi")], check=True, env=env)
        shutil.copyfile(generated, output)
        print(f"{output} ({output.stat().st_size:,} bytes); SHA-256 {file_digest(output)}")
        print(f"{len(files)} desktop files; {required:,} bytes of downloaded desktop components")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Lightricks Desktop pinned source ZIP")
    p.add_argument("--output", type=Path, default=ROOT / "build/release/LTX-AMD-395-MAX-Compact-Setup.exe")
    a = p.parse_args()
    build(a.source, a.output)
