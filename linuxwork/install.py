"""Reconstruct the native SpaceSpeak Debian package from the compact release ZIP."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from zipfile import ZipFile

DONOR_SHA = '0ce3f07ab0120e83a07b0f11f3f10e503343b8cb068ad4658ce5f9038099bf20'
DONOR_URL = 'https://github.com/Lightricks/LTX-Desktop/releases/download/v1.3.3/LTX-Desktop-amd64.deb'
PACKAGE = 'ltx-desktop-spacespeak'
BUFFER = 1024 * 1024


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(BUFFER), b''):
            h.update(block)
    return h.hexdigest()


def download(cache):
    cache.mkdir(parents=True, exist_ok=True)
    donor = cache / 'LTX-Desktop-v1.3.3-amd64.deb'
    if donor.is_file() and sha256_file(donor) == DONOR_SHA:
        print('Using the verified LTX v1.3.3 download.', flush=True)
        return donor
    print('Downloading the pinned LTX v1.3.3 Linux desktop (490 MB).', flush=True)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        part = Path(directory) / 'download'
        with urllib.request.urlopen(DONOR_URL, timeout=120) as response, part.open('wb') as out:
            shutil.copyfileobj(response, out, BUFFER)
        if sha256_file(part) != DONOR_SHA:
            raise ValueError('The downloaded upstream package failed its SHA-256 check.')
        part.replace(donor)
    return donor


def ar_header(name, size):
    return (f'{name + "/":<16}{0:<12}{0:<6}{0:<6}{"100644":<8}{size:<10}`\n').encode('ascii')


def write_ar_member(stream, name, data):
    stream.write(ar_header(name, len(data)))
    stream.write(data)
    if len(data) % 2:
        stream.write(b'\n')


def make_package(payload, donor, work, destination):
    if sha256_file(donor) != DONOR_SHA:
        raise ValueError('This upstream package does not match Lightricks v1.3.3 for amd64.')
    with tempfile.TemporaryDirectory(dir=work) as directory:
        tar = Path(directory) / 'upstream.tar'
        print('Reconstructing the community fork and verifying its contents.', flush=True)
        with tar.open('wb') as out:
            subprocess.run(['dpkg-deb', '--fsys-tarfile', str(donor)], stdout=out, check=True)
        with ZipFile(payload) as release, tar.open('rb') as original, tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix='.ltx-', suffix='.deb', delete=False) as deb:
            tmp = Path(deb.name)
            try:
                plan = json.loads(release.read('plan.json'))
                control = release.read('control.tar.gz')
                if hashlib.sha256(gzip.decompress(control)).hexdigest() != plan['controlSha256']:
                    raise ValueError('The Debian control archive failed verification.')
                deb.write(b'!<arch>\n')
                write_ar_member(deb, 'debian-binary', b'2.0\n')
                write_ar_member(deb, 'control.tar.gz', control)
                header = deb.tell()
                deb.write(ar_header('data.tar.gz', 0))
                data_start = deb.tell()
                h, count = hashlib.sha256(), 0
                with gzip.GzipFile(filename='', mode='wb', fileobj=deb, compresslevel=1, mtime=0) as out:
                    for source, length in plan['chunks']:
                        if isinstance(source, int):
                            original.seek(source)
                            size = length
                            while size:
                                block = original.read(min(BUFFER, size))
                                if not block:
                                    raise ValueError('Unexpected end of upstream data.')
                                out.write(block)
                                h.update(block)
                                size -= len(block)
                        else:
                            block = release.read('data/' + source)
                            if len(block) != length or hashlib.sha256(block).hexdigest() != source:
                                raise ValueError('A community resource failed integrity verification.')
                            out.write(block)
                            h.update(block)
                        count += length
                if count != plan['size'] or h.hexdigest() != plan['sha256']:
                    raise ValueError('The reconstructed desktop failed its release SHA-256 check.')
                size = deb.tell() - data_start
                if size % 2:
                    deb.write(b'\n')
                deb.seek(header)
                deb.write(ar_header('data.tar.gz', size))
                deb.flush()
                identity = subprocess.check_output(['dpkg-deb','--show','--showformat=${Package} ${Architecture}',str(tmp)],text=True)
                if identity != PACKAGE + ' amd64':
                    raise ValueError('Unexpected Debian package name or architecture.')
                os.chmod(tmp, 0o644)
                tmp.replace(destination)
            finally:
                tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write the verified .deb here and stop before installation')
    parser.add_argument('--upstream-deb', type=Path, help='Use an existing Lightricks v1.3.3 amd64 .deb')
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine().lower() not in ('amd64','x86_64') or not shutil.which('dpkg-deb'):
        parser.exit(2, 'Use this installer on a 64-bit x86 Debian/Ubuntu system with dpkg-deb.\n')
    if not args.output and os.geteuid() == 0:
        parser.exit(2, 'Start this installer as your regular desktop user. Apt will request permission when needed.\n')
    cache = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / PACKAGE / 'installer'
    cache.mkdir(parents=True, exist_ok=True)
    donor = args.upstream_deb or download(cache)
    release = Path(sys.argv[0]).resolve()
    target = (args.output or cache / (PACKAGE + '-amd64.deb')).resolve()
    make_package(release, donor, cache, target)
    print('Verified Debian package:', target, flush=True)
    if not args.output:
        subprocess.run(['sudo','apt','install',str(target)], check=True)


if __name__ == '__main__':
    main()
