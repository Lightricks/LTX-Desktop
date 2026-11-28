"""Create the compact installer from the final community .deb and pinned Lightricks .deb."""
import argparse
import gzip
import hashlib
import json
import mmap
from pathlib import Path
import struct
import subprocess
import tarfile
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from install import DONOR_SHA, sha256_file


def asar_regions(data, start, size):
    header_length = struct.unpack_from("<I", data, start + 4)[0] + 8
    json_length = struct.unpack_from("<I", data, start + 12)[0]
    tree = json.loads(data[start + 16:start + 16 + json_length])
    def walk(files):
        for entry in files.values():
            if "files" in entry:
                yield from walk(entry["files"])
            elif "offset" in entry and entry["size"]:
                yield start + header_length + int(entry["offset"]), entry["size"]
    yield from sorted(walk(tree["files"]))


def regions(data, tar):
    """Keep tar headers/modes/links byte-exact, and reuse individual packed Electron files."""
    pos = 0
    for member in tar:
        if not member.isfile() or not member.size:
            continue
        start = member.offset_data
        pieces = asar_regions(data, start, member.size) if member.name.endswith('/app.asar') else [(start, member.size)]
        for offset, length in pieces:
            if offset < pos or offset + length > start + member.size:
                raise ValueError("Invalid file boundaries in input archive")
            if pos < offset:
                yield pos, offset - pos
            yield offset, length
            pos = offset + length
    if pos < len(data):
        yield pos, len(data) - pos


def data_tar(package, destination):
    with destination.open('wb') as output:
        subprocess.run(['dpkg-deb', '--fsys-tarfile', str(package)], stdout=output, check=True)


def build(donor, package, output):
    if sha256_file(donor) != DONOR_SHA:
        raise ValueError("Wrong upstream package")
    identity = subprocess.check_output(['dpkg-deb','--show','--showformat=${Package} ${Architecture}',str(package)],text=True)
    if identity != 'ltx-desktop-spacespeak amd64':
        raise ValueError(f"Unexpected community package identity: {identity}")
    with tempfile.TemporaryDirectory() as tmp, ZipFile(output, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        path = Path(tmp) / 'data.tar'
        data_tar(donor, path)
        offsets = {}
        with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data, tarfile.open(path, 'r:') as tar:
            for start, length in regions(data, tar):
                key = hashlib.sha256(memoryview(data)[start:start + length]).hexdigest()
                offsets[key] = start
        data_tar(package, path)
        chunks, included = [], set()
        with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data, tarfile.open(path, 'r:') as tar:
            for start, length in regions(data, tar):
                chunk = memoryview(data)[start:start + length]
                key = hashlib.sha256(chunk).hexdigest()
                if key in offsets:
                    chunks.append([offsets[key], length])
                else:
                    if key not in included:
                        archive.writestr('data/'+key, chunk)
                        included.add(key)
                    chunks.append([key, length])
                del chunk
        control = subprocess.check_output(['dpkg-deb','--ctrl-tarfile', str(package)])
        archive.writestr('control.tar.gz', gzip.compress(control, mtime=0))
        plan = {'size':path.stat().st_size,'sha256':sha256_file(path),'chunks':chunks,
                'controlSha256':hashlib.sha256(control).hexdigest()}
        archive.writestr('plan.json', json.dumps(plan,separators=(',',':')))
        archive.write(Path(__file__).with_name('install.py'),'__main__.py')
        archive.write(Path(__file__).with_name('README.txt'),'README.txt')
    print(f"{output.name}: {output.stat().st_size:,} bytes; {len(chunks)} archive segments")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-deb', type=Path, required=True)
    parser.add_argument('--deb', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.upstream_deb, args.deb, args.output)
