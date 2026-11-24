"""Small media regressions. No compilers, GPUs or network needed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from build import seal_setup
from src.setup_engine import MARKER, preserve_launcher_alias, unpack_media
from unittest.mock import patch


class SetupMediaTests(unittest.TestCase):
    def test_both_modes_bootstrap_from_their_own_embedded_index(self):
        for mode in ("online", "offline"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as base:
                base = Path(base)
                bootstrap = base / "extracted"
                bootstrap.mkdir()
                stub = base / "stub"
                stub.write_bytes(b"pretend PE stub")
                files = [base / "desktop.zip"]
                if mode == "offline":
                    files.append(base / "runtime.tar.xz")
                for path in files:
                    path.write_bytes(path.name.encode())
                index = {p.name: {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in files}
                (bootstrap / f"{mode}-index.json").write_text(json.dumps(index), encoding="utf8")
                setup = base / "installer.exe"
                seal_setup(stub, setup, files)
                unpack_media(setup, bootstrap)
                for path in files:
                    self.assertEqual(path.read_bytes(), (bootstrap / path.name).read_bytes())

                index[files[0].name]["sha256"] = "0" * 64
                (bootstrap / f"{mode}-index.json").write_text(json.dumps(index), encoding="utf8")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    unpack_media(setup, bootstrap)

    def test_legacy_launcher_only_for_owned_upgrades(self):
        with tempfile.TemporaryDirectory() as work:
            current, stage = Path(work) / "application", Path(work) / "installing"
            current.mkdir()
            stage.mkdir()
            old = "LTX Desktop AMD.exe"
            new = stage / "LTX Desktop - AMD 395 MAX.exe"
            new.write_bytes(b"new launcher")
            preserve_launcher_alias(current, stage)
            self.assertFalse((stage / old).exists())
            (current / old).write_bytes(b"legacy launcher")
            preserve_launcher_alias(current, stage)
            self.assertFalse((stage / old).exists())
            (current / MARKER).touch()
            with patch("src.setup_engine.os.link", side_effect=OSError("unsupported")), self.assertLogs(level="WARNING"):
                preserve_launcher_alias(current, stage)
            self.assertFalse((stage / old).exists())
            preserve_launcher_alias(current, stage)
            self.assertTrue((stage / old).samefile(new))

    def test_unexpected_overlay_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as base:
            base = Path(base)
            (base / "online-index.json").write_text('{"desktop.zip": {}}')
            media = base / "setup.exe"
            with zipfile.ZipFile(media, "w") as archive:
                archive.writestr("desktop.zip", b"")
                archive.writestr("extra.exe", b"")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                unpack_media(media, base)


if __name__ == "__main__":
    unittest.main()
