"""Exercise the Mac online installer's staged resource and app activation checks."""

import errno
import hashlib
import json
from pathlib import Path
import plistlib
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import materialize


class InstallTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.apps = Path(temporary.name) / "Test Applications"
        self.stage = self.apps / "staging"
        self.stage.mkdir(parents=True)
        self.target = self.apps / (materialize.PRODUCT + ".app")
        self.candidate = self.stage / self.target.name
        self.create_bundle(self.candidate, "new")

    def create_bundle(self, path, version, identifier=materialize.IDENTIFIER):
        info = path / "Contents" / "Info.plist"
        info.parent.mkdir(parents=True)
        info.write_bytes(plistlib.dumps({"CFBundleIdentifier": identifier, "SpaceSpeakCommunityVersion": version}))

    def test_new_app_moves_to_user_applications(self):
        materialize.activate(self.candidate, self.apps)
        self.assertTrue((self.target / "Contents/Info.plist").is_file())
        self.assertFalse(self.candidate.exists())

    def test_unowned_application_and_symbolic_link_survive(self):
        self.create_bundle(self.target, "other", "com.example.another-app")
        with self.assertRaisesRegex(RuntimeError, "different app"):
            materialize.activate(self.candidate, self.apps)
        original = self.stage / "unrelated.app"
        self.target.rename(original)
        self.target.symlink_to(original)
        with self.assertRaisesRegex(RuntimeError, "symbolic link"):
            materialize.activate(self.candidate, self.apps)
        self.assertTrue(self.target.is_symlink())
        self.assertTrue(self.candidate.is_dir())

    def test_failed_swap_preserves_installed_app_and_candidate(self):
        self.create_bundle(self.target, "previous")
        previous = (self.target / "Contents/Info.plist").read_bytes()
        with patch.object(materialize.ctypes, "CDLL") as lib, patch.object(materialize.ctypes, "get_errno", return_value=errno.EACCES):
            lib.return_value.renamex_np.return_value = -1
            with self.assertRaises(PermissionError):
                materialize.activate(self.candidate, self.apps)
            self.assertEqual(lib.return_value.renamex_np.call_args.args[2], 2)
        self.assertEqual((self.target / "Contents/Info.plist").read_bytes(), previous)
        self.assertTrue(self.candidate.exists())

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS renamex_np")
    def test_completed_upgrade_exchanges_both_complete_apps(self):
        self.create_bundle(self.target, "previous")
        materialize.activate(self.candidate, self.apps)
        for path, version in ((self.target, "new"), (self.candidate, "previous")):
            meta = plistlib.loads((path / "Contents/Info.plist").read_bytes())
            self.assertEqual(meta["SpaceSpeakCommunityVersion"], version)

    def test_assemble_preserves_signed_framework_version_and_helper_lookup(self):
        shell = self.stage / "original.app/Contents"
        donor = shell / "Resources"
        (donor / "koffi").mkdir(parents=True)
        (donor / "python-deps-hash.txt").write_text("runtime")
        (donor / "app.asar").write_bytes(b"")
        (shell / "PkgInfo").write_bytes(b"APPL????")
        (shell / "MacOS").mkdir()
        (shell / "MacOS/LTX Desktop").write_bytes(b"launcher")
        (shell / "Info.plist").write_bytes(plistlib.dumps(
            {"CFBundleExecutable": "LTX Desktop", "CFBundleName": "LTX Desktop"}))
        framework = Path("Frameworks/Electron Framework.framework")
        version = shell / framework / "Versions/A"
        (version / "_CodeSignature").mkdir(parents=True)
        (version / "_CodeSignature/CodeResources").write_bytes(b"original seal")
        (version.parent / "Current").symlink_to("A", target_is_directory=True)
        helper = Path("Frameworks/LTX Desktop Helper.app/Contents/MacOS/LTX Desktop Helper")
        (shell / helper.parent).mkdir(parents=True)
        (shell / helper).write_bytes(b"signed helper")

        payload = self.stage / "community.zip"
        with ZipFile(payload, "w") as z:
            z.writestr("plan.json", json.dumps({"parts": [], "files": [], "blobs": {},
                                                "version": "1.3.3-community.4", "headerHash": "test"}))
        contents = self.candidate / "Contents"
        (contents / "Resources").mkdir()
        materialize.assemble(payload, donor, self.candidate)
        self.assertEqual((contents / framework / "Versions/Current").readlink(), Path("A"))
        self.assertEqual((contents / framework / "Versions/Current/_CodeSignature/CodeResources").read_bytes(), b"original seal")
        metadata = plistlib.loads((contents / "Info.plist").read_bytes())
        self.assertEqual(metadata["CFBundleName"], "LTX Desktop")
        self.assertTrue((contents / helper).is_file())
        self.assertEqual(metadata["CFBundleDisplayName"], materialize.PRODUCT)
        self.assertEqual(metadata["CFBundleIdentifier"], materialize.IDENTIFIER)

    def test_corrupt_payload_or_donor_stops_before_activation(self):
        self.create_bundle(self.target, "previous")
        donor = self.stage / "donor"
        donor.mkdir()
        (donor / "app.asar").write_bytes(b"bad!")
        (self.candidate / "Contents/Resources").mkdir()
        correct = hashlib.sha256(b"good").hexdigest()
        for source in (["span", 0], ["payload", correct]):
            with self.subTest(source=source[0]):
                package = self.stage / "payload.zip"
                with ZipFile(package, "w") as z:
                    z.writestr("plan.json", json.dumps({"blobs": {correct: [4, *source]}, "parts": [correct], "files": []}))
                    z.writestr("blobs/" + correct, b"bad!")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    materialize.assemble(package, donor, self.candidate)
                self.assertEqual(plistlib.loads((self.target / "Contents/Info.plist").read_bytes())["SpaceSpeakCommunityVersion"], "previous")


if __name__ == "__main__":
    unittest.main()
