"""Rebuild the official AMD gfx1151 Python bootstrap metadata wheel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
EXPECTED_SDIST = 'cd1d876484f90094973e18ba02a9d06170677279f701aa32e6f5dcb13f01e25d'
VERSION = '7.13.0'
FILENAME = f'rocm-{VERSION}-py3-none-any.whl'


def rebuild(out: Path) -> None:
    sdist = HERE / 'upstream' / f'rocm-{VERSION}.tar.gz'
    if hashlib.sha256(sdist.read_bytes()).hexdigest() != EXPECTED_SDIST:
        raise ValueError('The AMD source archive failed its integrity check.')
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        with tarfile.open(sdist, 'r:gz') as archive:
            for entry in archive:
                if not entry.isfile():
                    if entry.isdir():
                        continue
                    raise ValueError('Unexpected entry in locked AMD source archive')
                path = stage / entry.name
                if not path.resolve().is_relative_to(stage):
                    raise ValueError('Unexpected path in AMD source archive')
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target)
        constraints = stage / 'build-constraints.txt'
        constraints.write_text('setuptools==84.2.1\n', encoding='ascii')
        env = dict(os.environ, ROCM_SDK_TARGET_FAMILY='gfx1151', SOURCE_DATE_EPOCH='1735689600')
        subprocess.run(['uv', 'build', '--wheel', '--build-constraint', str(constraints),
                        '--out-dir', str(stage / 'wheel'), str(stage / f'rocm-{VERSION}')],
                       check=True, env=env)
        built = stage / 'wheel' / FILENAME
        locked = json.loads((HERE / 'runtime-manifest.json').read_text(encoding='utf-8'))
        expected = next(item['sha256'] for item in locked['wheels'] if item['filename'] == FILENAME)
        digest = hashlib.sha256(built.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'The built {FILENAME} differs from the locked release ({digest}).')
        shutil.copyfile(built, out / FILENAME)
        print(f'{FILENAME} {digest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    rebuild(parser.parse_args().out_dir)
