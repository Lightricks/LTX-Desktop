"""Install a hash-locked Radeon or API-only wheel set into embedded Windows Python.

Usage: python install_runtime.py --python-dir DIR --wheelhouse DIR --manifest FILE
The caller supplies a *staging directory*, with stock python.exe extracted into it.
Run a GPU preflight on the destination PC before promoting a staged installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile


def long_path(value: Path) -> Path:
    result = value.resolve()
    if os.name == 'nt' and not str(result).startswith('\\\\?\\'):
        value_str = str(result)
        if value_str.startswith('\\\\'):
            return Path('\\\\?\\UNC\\' + value_str[2:])
        return Path('\\\\?\\' + value_str)
    return result


def wheel_path(name: str, python_dir: Path, site: Path) -> Path:
    if not name or any(p in ('', '.', '..') or ':' in p or '\\' in p for p in name.rstrip('/').split('/')):
        raise ValueError(f'Invalid wheel entry {name!r}')
    parts = PurePosixPath(name).parts
    if parts[0].endswith('.data'):
        if len(parts) < 3:
            raise ValueError(f'Invalid wheel data path {name!r}')
        targets = {'purelib': site, 'platlib': site, 'scripts': python_dir / 'Scripts',
                   'data': python_dir, 'headers': python_dir / 'Include'}
        return targets[parts[1]].joinpath(*parts[2:])
    return site.joinpath(*parts)


def configure_rocm_loader(site: Path) -> None:
    # AMD splits its DLLs across target-specific and common directories. Keep
    # both searchable for dependencies loaded after the initial Torch import.
    # These handles live until exit and do not change the computer's PATH.
    (site / 'sitecustomize.py').write_text(
        "import os\n"
        "from pathlib import Path\n\n"
        "if os.name == 'nt':\n"
        "    _site = Path(__file__).resolve().parent\n"
        "    os.environ['ROCM_SDK_TARGET_FAMILY'] = 'gfx1151'\n"
        "    _rocm_dll_dirs = [os.add_dll_directory(str(_site / path)) for path in (\n"
        "        '_rocm_sdk_core/bin', '_rocm_sdk_libraries_gfx1151/bin', 'torch/lib'\n"
        "    )]\n",
        encoding='ascii',
    )


def install(python_dir: Path, wheelhouse: Path, manifest_path: Path) -> None:
    python_dir = long_path(python_dir)
    wheelhouse = long_path(wheelhouse)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not (python_dir / 'python.exe').is_file() or manifest['python'] != '3.13.12':
        raise ValueError('The target directory needs the 3.13.12 Windows embeddable Python distribution.')
    site = python_dir / 'Lib' / 'site-packages'
    site.mkdir(parents=True, exist_ok=True)
    target = manifest['target']
    if target not in ('gfx1151', 'cpu-api'):
        raise ValueError(f'Unknown Windows wheel target: {target}')
    marker = python_dir / ('ltx-amd-runtime.json' if target == 'gfx1151' else 'ltx-api-runtime.json')
    marker.unlink(missing_ok=True)
    count = len(manifest['wheels'])
    for i, item in enumerate(manifest['wheels'], 1):
        filename = item['filename']
        if filename != Path(filename).name or not filename.endswith('.whl'):
            raise ValueError(f'Invalid wheel filename {filename!r}')
        source = wheelhouse / filename
        print(f'[{i}/{count}] {filename}', flush=True)
        if source.stat().st_size != item['size']:
            raise ValueError(f'Incomplete wheel: {filename}')
        with source.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
            if digest != item['sha256']:
                raise ValueError(f'Wheel SHA-256 mismatch: {filename}')
            f.seek(0)
            with zipfile.ZipFile(f) as wheel:
                for entry in wheel.infolist():
                    if entry.is_dir():
                        continue
                    path = wheel_path(entry.filename, python_dir, site)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with wheel.open(entry) as src, path.open('wb') as out:
                        shutil.copyfileobj(src, out, 1024 * 1024)
    # Isolated embedded Python ignores PYTHONPATH and user site-packages.
    # Enable site-managed DLL initialization and .pth hooks supplied by the wheels.
    (python_dir / 'python313._pth').write_text(
        'python313.zip\n.\nLib\\site-packages\nimport site\n', encoding='ascii'
    )
    # The embedded Windows interpreter uses isolated ._pth paths; when the
    # setup executable runs this script, its directory is not on sys.path.
    if target == 'gfx1151':
        import runpy
        patch = runpy.run_path(str(Path(__file__).with_name('repair_single_gpu.py')))
        patch['apply_to_site'](site)
        configure_rocm_loader(site)
    marker.write_text(json.dumps({
        'python': manifest['python'], 'target': manifest['target'], 'wheel_count': count,
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-dir', required=True, type=Path)
    parser.add_argument('--wheelhouse', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    install(args.python_dir, args.wheelhouse, args.manifest)
