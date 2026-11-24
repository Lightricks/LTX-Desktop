"""Regression tests using the two pinned wheels from an installer build cache.

Run: python runtimework/test_repair_single_gpu.py --wheelhouse PATH
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

import repair_single_gpu as patch

HERE = Path(__file__).resolve().parent


class RepairTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.site = Path(folder.name) / 'site-packages'
        manifest = json.loads((HERE / 'runtime-manifest.json').read_text(encoding='utf-8'))
        for pkg, version in patch.VERSIONS.items():
            filename = f'{pkg}-{version}-py3-none-any.whl'
            item = next(w for w in manifest['wheels'] if w['filename'] == filename)
            with (WHEELHOUSE / filename).open('rb') as binary:
                self.assertEqual(hashlib.file_digest(binary, 'sha256').hexdigest(), item['sha256'])
                with zipfile.ZipFile(binary) as wheel:
                    for name in patch.PATCHES:
                        if name.startswith(pkg + '/'):
                            wheel.extract(name, self.site)
                    for basename in ('METADATA', 'RECORD'):
                        wheel.extract(f'{pkg}-{version}.dist-info/{basename}', self.site)

    def assert_patched_records(self):
        for name, (_, expected, _) in patch.PATCHES.items():
            contents = (self.site / name).read_bytes()
            self.assertEqual(patch.sha256(contents), expected)
            pkg = name.split('/')[0]
            record = self.site / f'{pkg}-{patch.VERSIONS[pkg]}.dist-info' / 'RECORD'
            rows = csv.reader(io.StringIO(record.read_text(encoding='utf-8')))
            row = next(row for row in rows if row[0] == name)
            self.assertEqual(row[1:], [patch.record_hash(expected), str(len(contents))])

    def test_interrupted_repair_and_retry(self):
        updates = patch.plan_updates(self.site)
        first = next(iter(updates))
        first.write_bytes(updates[first])
        self.assertEqual(patch.apply_to_site(self.site), len(updates) - 1)
        self.assert_patched_records()
        self.assertEqual(patch.apply_to_site(self.site), 0)

    def test_unrecognized_source_preserved_before_modifying_any_files(self):
        last = list(patch.PATCHES)[-1]
        (self.site / last).write_bytes(b'# unrecognized version\n')
        before = {p: p.read_bytes() for p in self.site.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, 'differs from the supported release'):
            patch.apply_to_site(self.site)
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original, str(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    WHEELHOUSE = parser.parse_args().wheelhouse
    unittest.main(argv=[__file__], verbosity=2)
