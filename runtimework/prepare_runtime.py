"""Stage a Windows x64 embedded-Python runtime or the inputs to its web installer."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request
import zipfile

import install_runtime
import visual_cpp

HERE = Path(__file__).resolve().parent
OWNER = '.ltx-amd-runtime-staging'
PUBLIC_LOCK = HERE / 'runtime-manifest.json'


def verified(path: Path, asset: dict) -> bool:
    if not path.is_file() or path.stat().st_size != asset['size']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == asset['sha256']


def acquire(asset: dict, cache: Path) -> Path:
    filename = asset['filename']
    if filename != Path(filename).name or '/' in filename or '\\' in filename:
        raise ValueError(f'Invalid filename in release manifest: {filename}')
    if asset['url'] is None:
        local = HERE / 'local-wheels' / filename
        if not verified(local, asset):
            raise ValueError(f'Local wheel failed its release integrity check: {filename}')
        return local
    if not asset['url'].startswith('https://'):
        raise ValueError(f'Expected an HTTPS release URL for {filename}')
    dest = cache / filename
    if verified(dest, asset):
        return dest
    for attempt in range(3):
        part = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False, suffix='.part') as output:
                part = Path(output.name)
                with urllib.request.urlopen(asset['url'], timeout=90) as response:
                    if not response.geturl().startswith('https://'):
                        raise ValueError('Expected HTTPS after release URL redirect')
                    shutil.copyfileobj(response, output, 1024 * 1024)
            if not verified(part, asset):
                raise OSError(f'Size or SHA-256 mismatch downloading {filename}')
            part.replace(dest)
            print(f'Downloaded and verified {filename}', flush=True)
            return dest
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
        finally:
            if part is not None:
                part.unlink(missing_ok=True)
    raise RuntimeError(f'Could not acquire {filename}')


def link_or_copy(source: Path, dest: Path) -> None:
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)


def prepare(stage: Path, mode: str, cache: Path, sevenzip: str) -> None:
    bootstrap = json.loads((HERE / 'bootstrap-sources.json').read_text(encoding='utf-8'))
    manifest = json.loads(PUBLIC_LOCK.read_text(encoding='utf-8'))
    chosen = [item for item in manifest['wheels'] if mode == 'offline' or item['url'] is None]
    (stage / 'wheels').mkdir()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for item, source in zip(chosen, pool.map(lambda it: acquire(it, cache), chosen)):
            link_or_copy(source, stage / 'wheels' / item['filename'])

    python_zip = acquire(bootstrap['python'], cache)
    redist = acquire(bootstrap['visual_cpp'], cache)
    python = stage / 'python'
    with zipfile.ZipFile(python_zip) as archive:
        archive.extractall(python)
    visual_cpp.extract(redist.read_bytes(), stage / 'vcredist', sevenzip)
    shutil.copytree(stage / 'vcredist', python, dirs_exist_ok=True)
    shutil.copy(PUBLIC_LOCK, stage / PUBLIC_LOCK.name)
    for script in ('install_runtime.py', 'repair_single_gpu.py'):
        shutil.copy(HERE / script, stage / script)
    install_runtime.configure_rocm_loader(stage)
    if mode == 'offline':
        dest = stage / 'python-embed'
        shutil.copytree(python, dest)
        install_runtime.install(dest, stage / 'wheels', stage / PUBLIC_LOCK.name)
    (stage / OWNER).write_text(mode + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Dedicated generated directory')
    parser.add_argument('--mode', choices=['web', 'offline'], required=True)
    parser.add_argument('--cache', type=Path, help='Shared flat download directory; default OUT/../.cache/runtime')
    parser.add_argument('--sevenzip', help='7z/7zz executable with CAB support')
    args = parser.parse_args()
    out = args.out.resolve()
    cache = (args.cache or out.parent / '.cache' / 'runtime').resolve()
    if cache.is_relative_to(out) or out == HERE or HERE.is_relative_to(out):
        parser.error('Choose separate source, build and cache directories')
    if out.exists() and (not out.is_dir() or (any(out.iterdir()) and not (out / OWNER).is_file())):
        parser.error('The build output is not an empty or managed runtime directory')
    sevenzip = args.sevenzip or shutil.which('7zz') or shutil.which('7z') or shutil.which('7za')
    if not sevenzip:
        parser.error('Install 7-Zip with CAB support or pass --sevenzip /path/to/7z')
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.runtime-', dir=out.parent) as tmp:
        tmpdir = Path(tmp)
        stage = tmpdir / 'new'
        stage.mkdir()
        prepare(stage, args.mode, cache, sevenzip)
        prior = tmpdir / 'previous'
        if out.exists():
            out.rename(prior)
        try:
            stage.rename(out)
        except BaseException:
            if prior.exists():
                prior.rename(out)
            raise
    print(f'Prepared {args.mode} runtime at {out}', flush=True)


if __name__ == '__main__':
    main()
