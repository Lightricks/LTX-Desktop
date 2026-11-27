"""Linux compact installer regression checks using real, miniature Debian packages."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import build_online
import install


def mini_asar(changed):
    content = [("shared", b"common packed model metadata"), ("branch", changed)]
    entries = {}
    data = bytearray()
    for name, value in content:
        entries[name] = {"offset": str(len(data)), "size": len(value)}
        data.extend(value)
    tree = json.dumps({"files": entries}).encode()
    header = struct.pack("<I", len(tree)) + tree
    header += bytes(-len(header) % 4)
    return struct.pack("<III", 4, 4 + len(header), len(header)) + header + data


def package(directory, name, branch):
    base = directory / name
    control = base / "DEBIAN"
    control.mkdir(parents=True)
    (control / "control").write_text(
        f"Package: {name}\nVersion: 1.3.3-community.4\nArchitecture: amd64\n"
        "Maintainer: Test <test@example.org>\nDescription: Installer QA fixture\n")
    resources = base / "opt" / "Test App"
    resources.mkdir(parents=True)
    executable = resources / "launcher"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    (resources / "launcher-link").symlink_to("launcher")
    (resources / "app.asar").write_bytes(mini_asar(branch))
    deb = directory / (name + ".deb")
    subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(base), str(deb)],
                   check=True, stdout=subprocess.DEVNULL)
    return deb


@unittest.skipUnless(install.sys.platform == "linux" and
                     install.platform.machine().lower() in ("amd64", "x86_64") and shutil.which("dpkg-deb"),
                     "requires x86-64 Linux and dpkg-deb")
class OnlineInstallerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.donor = package(self.work, "ltx-desktop", b"official")
        self.fork = package(self.work, install.PACKAGE, b"community")
        self.release = self.work / "release.zip"
        self.target = self.work / "prepared package.deb"
        self.old = b"previous package"
        self.target.write_bytes(self.old)
        for module in (install, build_online):
            donor_hash = patch.object(module, "DONOR_SHA", install.sha256_file(self.donor))
            donor_hash.start()
            self.addCleanup(donor_hash.stop)
        build_online.build(self.donor, self.fork, self.release)

    def test_regular_user_rebuild_keeps_both_archives_byte_exact(self):
        with patch.object(install.sys, "argv", [str(self.release), "--output", str(self.target),
                                                "--upstream-deb", str(self.donor)]), \
             patch.dict(os.environ, XDG_CACHE_HOME=str(self.work / "cache")), \
             patch.object(subprocess, "run", wraps=subprocess.run) as calls:
            install.main()
            self.assertTrue(all(call.args[0][0] == "dpkg-deb" for call in calls.call_args_list))
        for option in ("--ctrl-tarfile", "--fsys-tarfile"):
            with self.subTest(option=option):
                self.assertEqual(subprocess.check_output(["dpkg-deb", option, str(self.target)]),
                                 subprocess.check_output(["dpkg-deb", option, str(self.fork)]))
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o644)
        with ZipFile(self.release) as release:
            shared_hash = hashlib.sha256(b"common packed model metadata").hexdigest()
            changed_hash = hashlib.sha256(b"community").hexdigest()
            self.assertNotIn("data/" + shared_hash, release.namelist())
            self.assertIn("data/" + changed_hash, release.namelist())

    def test_wrong_donor_leaves_destination_intact(self):
        with self.assertRaisesRegex(ValueError, "upstream package"):
            install.make_package(self.release, self.fork, self.work, self.target)
        self.assertEqual(self.target.read_bytes(), self.old)

    def test_download_rejects_wrong_bytes_and_reuses_verified_cache(self):
        cache = self.work / "downloads"
        with patch.object(install.urllib.request, "urlopen", return_value=io.BytesIO(b"incomplete")):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                install.download(cache)
        self.assertEqual(list(cache.iterdir()), [])
        with patch.object(install.urllib.request, "urlopen", return_value=io.BytesIO(self.donor.read_bytes())):
            cached = install.download(cache)
        with patch.object(install.urllib.request, "urlopen", side_effect=AssertionError("unnecessary download")):
            self.assertEqual(install.download(cache), cached)
        self.assertEqual(cached.read_bytes(), self.donor.read_bytes())

    def test_damaged_resource_control_or_output_digest_leaves_destination_intact(self):
        chunk = "data/" + hashlib.sha256(b"community").hexdigest()
        for entry, error in [(chunk, "resource"), ("plan.json", "SHA-256"), ("control.tar.gz", "control")]:
            with self.subTest(entry=entry):
                bad = self.work / "damaged.zip"
                with ZipFile(self.release) as source, ZipFile(bad, "w") as out:
                    for name in source.namelist():
                        data = source.read(name)
                        if name == entry:
                            if name == chunk:
                                data = b"corrupted"
                            elif name == "plan.json":
                                data = json.dumps({**json.loads(data), "sha256": "0" * 64}).encode()
                            else:
                                data = install.gzip.compress(b"bad control")
                        out.writestr(name, data)
                with self.assertRaisesRegex(ValueError, error):
                    install.make_package(bad, self.donor, self.work, self.target)
                self.assertEqual(self.target.read_bytes(), self.old)
                self.assertEqual(list(self.work.glob(".ltx-*.deb")), [])


if __name__ == "__main__":
    unittest.main()
