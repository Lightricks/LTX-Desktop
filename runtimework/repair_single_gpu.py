"""Apply the release's optional-distributed compatibility patches to pinned wheel sources.

The local Windows ROCm build lacks c10d. Keep distributed-only imports and type
checks behind torch.distributed.is_available(), while retaining the full path
when it is provided by PyTorch. Each file accepts only its original release
digest or its exact repaired digest. A second run can resume after interruption.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import tempfile

# Input sources are from the hash-locked Windows release wheels. Each entry
# records original SHA-256, patched SHA-256 and the exact textual edit(s).
PATCHES = {
    'transformers/distributed/sharding_utils.py': (
        'a058c5bfde36d36e664b00bd747f02922f35e6bab3831acdcb1f4c2b9c1e6f5b',
        'a2ac123d7c1a809ea3f64ab40c03bb3bc207dc4cc249b40bba1fdc23c3203dfc',
        (
            ('if is_torch_available():\n    import torch\n    from torch.distributed.tensor import DTensor',
             'if is_torch_available():\n    import torch\n\nif is_torch_available() and torch.distributed.is_available():\n    from torch.distributed.tensor import DTensor'),
        ),
    ),
    'transformers/distributed/fsdp.py': (
        'c9759ff5e32b68f8cdf1b8ce49c1efa5c686b58c32cc0c8708754cd8b2b4eb47',
        '725f330b890a5007b368ca6819886b0249191e8bae1ba79baba449cb63ef3601',
        (
            ('if is_torch_available() and is_torch_greater_or_equal("2.6"):',
             'if is_torch_available() and torch.distributed.is_available() and is_torch_greater_or_equal("2.6"):'),
            ('    if not is_torch_greater_or_equal("2.6"):\n        raise OSError("FSDP2 requires torch>=2.6")',
             '    if not torch.distributed.is_available():\n        raise RuntimeError("FSDP requires PyTorch distributed support")\n\n    if not is_torch_greater_or_equal("2.6"):\n        raise OSError("FSDP2 requires torch>=2.6")'),
        ),
    ),
    'transformers/core_model_loading.py': (
        'bdba8ed71e303c626f91972b4787922f5f23d17060c37145adbbee1ae36599c6',
        '011dab59bc9f775aeb7caae07af2c94d25b18403679ecf3e90b0c6d4d8d09d49',
        (
            ('    from torch.distributed.tensor import DTensor\n',
             '    from torch.distributed.tensor import DTensor\n\n\ndef _is_dtensor(value):\n    return _torch_distributed_available and isinstance(value, DTensor)\n'),
            ('isinstance(ref, DTensor)',
             '_is_dtensor(ref)'),
            ('isinstance(empty_param, DTensor)',
             '_is_dtensor(empty_param)'),
        ),
    ),
    'diffusers/models/_modeling_parallel.py': (
        'be790a2cd4dd7be50705de2c3d1ab0d6cb7d44bea540cf2cefe10d32c2dc30ca',
        'd5c728a15c0eebf3c1bca639489738c50fadf133db7f4d554f3aad2270819085',
        (
            ('from dataclasses import dataclass',
             'from __future__ import annotations\n\nfrom dataclasses import dataclass'),
        ),
    ),
    'diffusers/hooks/context_parallel.py': (
        'd58fac70f989d5aa82ea38011f62b59e3220898bbc22bd59a0acff9b64ff4481',
        '0edcc0e74fe3e650c935749434acc73c625dbd094b774f71af8c7a845f719559',
        (
            ('import copy\n',
             'from __future__ import annotations\n\nimport copy\n'),
        ),
    ),
}

VERSIONS = {'transformers': '5.14.1', 'diffusers': '0.39.0'}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_hash(hex_digest: str) -> str:
    return 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(hex_digest)).rstrip(b'=').decode('ascii')


def plan_updates(site: Path) -> dict[Path, bytes]:
    changes = {}
    records = {}
    for pkg, version in VERSIONS.items():
        metadata = site / f'{pkg}-{version}.dist-info'
        if not (metadata / 'METADATA').is_file():
            raise ValueError(f'Expected the installed {pkg} {version} wheel in {site}')
        record = metadata / 'RECORD'
        rows = list(csv.reader(io.StringIO(record.read_text(encoding='utf-8'), newline='')))
        records[pkg] = (record, rows, {row[0]: row for row in rows})

    for name, (before_hash, after_hash, edits) in PATCHES.items():
        path = site / name
        original = path.read_bytes()
        digest = sha256(original)
        if digest == before_hash:
            text = original.decode('utf-8')
            for old, new in edits:
                text = text.replace(old, new)
            repaired = text.encode('utf-8')
            if sha256(repaired) != after_hash:
                raise ValueError(f'Unexpected repaired source for {name}')
            compile(repaired, str(path), 'exec')
            changes[path] = repaired
        elif digest == after_hash:
            repaired = original
        else:
            raise ValueError(f'{name} differs from the supported release (SHA-256 {digest}).')
        pkg = name.split('/')[0]
        row = records[pkg][2].get(name)
        if row is None or len(row) != 3 or row[1] not in (record_hash(before_hash), record_hash(after_hash)):
            raise ValueError(f'Unexpected wheel RECORD entry for {name}')
        row[1:] = [record_hash(after_hash), str(len(repaired))]

    for record, rows, _ in records.values():
        # Retain the wheel author's original line endings and CSV row order.
        old = record.read_bytes()
        data = io.StringIO(newline='')
        writer = csv.writer(data, lineterminator='\r\n' if b'\r\n' in old else '\n')
        writer.writerows(rows)
        new = data.getvalue().encode('utf-8')
        if old != new:
            changes[record] = new
    return changes


def apply_to_site(site: Path) -> int:
    changes = plan_updates(site)
    for path, data in changes.items():
        with tempfile.NamedTemporaryFile(prefix='.ltx-amd-', dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
            except BaseException:
                stream.close()
                temp.unlink(missing_ok=True)
                raise
        try:
            temp.chmod(path.stat().st_mode)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    return len(changes)


def repair(python_dir: Path) -> int:
    return apply_to_site(python_dir / 'Lib' / 'site-packages')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-dir', type=Path, required=True)
    changed = repair(parser.parse_args().python_dir)
    print(f'Single-GPU runtime compatibility verified ({changed} file updates).')
