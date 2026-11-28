"""Regression coverage for the Windows installer's pinned component downloader."""
import hashlib
import io
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "setupwork/src"))
import setup_engine as setup


def certificate_error(code, message):
    error = ssl.SSLCertVerificationError(1, message)
    error.verify_code = code
    error.verify_message = message
    return error


class SetupDownloads(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.target = Path(temp.name) / "electron.zip"
        self.partial = self.target.with_suffix(".zip.partial")
        self.data = b"expected release bytes"
        self.item = {"url": "https://github.com/example/release/electron.zip", "size": len(self.data),
                     "sha256": hashlib.sha256(self.data).hexdigest()}
        self.chain_error = certificate_error(20, "unable to get local issuer certificate")
        self.enterContext(patch.object(setup.time, "sleep"))
        self.enterContext(patch.object(setup, "sys", SimpleNamespace(platform="win32")))

    def test_verified_cache_and_completed_partial_do_not_use_network(self):
        with patch.object(setup.urllib.request, "urlopen") as primary, \
                patch.object(setup, "_download_with_windows_trust") as fallback:
            for cached_path in (self.partial, self.target):
                with self.subTest(cached_path=cached_path):
                    cached_path.write_bytes(self.data)
                    setup.fetch(self.item, self.target)
                    self.assertEqual(self.target.read_bytes(), self.data)
                    self.assertFalse(self.partial.exists())
            primary.assert_not_called()
            fallback.assert_not_called()

    def test_reported_issuer_error_uses_native_trust_and_checks_completed_file(self):
        def native_download(url, partial):
            self.assertEqual(url, self.item["url"])
            self.assertEqual(partial.read_bytes(), self.data[:4])
            with partial.open("ab") as f:
                f.write(self.data[4:])

        for error in (URLError(self.chain_error), self.chain_error):
            with self.subTest(error=error):
                self.partial.write_bytes(self.data[:4])
                with patch.object(setup.urllib.request, "urlopen", side_effect=error), \
                        patch.object(setup, "_download_with_windows_trust", side_effect=native_download) as fallback:
                    setup.fetch(self.item, self.target)
                fallback.assert_called_once_with(self.item["url"], self.partial)
                self.assertEqual(self.target.read_bytes(), self.data)
                self.assertFalse(self.partial.exists())
                self.target.unlink()

    def test_native_transfer_with_wrong_hash_is_never_published(self):
        wrong = b"X" * len(self.data)
        with patch.object(setup.urllib.request, "urlopen", side_effect=URLError(self.chain_error)), \
                patch.object(setup, "_download_with_windows_trust",
                             side_effect=lambda url, partial: partial.write_bytes(wrong)):
            with self.assertRaisesRegex(RuntimeError, "SHA-256|hash|checksum"):
                setup.fetch(self.item, self.target)
        self.assertFalse(self.target.exists())

    def test_unrelated_errors_do_not_select_a_second_trust_path(self):
        for error in (certificate_error(62, "hostname mismatch"),
                      certificate_error(10, "certificate has expired"), ConnectionResetError()):
            with self.subTest(error=error), \
                    patch.object(setup.urllib.request, "urlopen", side_effect=URLError(error)), \
                    patch.object(setup, "_download_with_windows_trust") as fallback:
                with self.assertRaises(RuntimeError):
                    setup.fetch(self.item, self.target)
                fallback.assert_not_called()
                self.assertFalse(self.target.exists())

    def test_resumes_normal_https_after_transient_error(self):
        self.partial.write_bytes(self.data[:4])
        response = io.BytesIO(self.data[4:])
        response.status = 206
        response.headers = {"Content-Range": f"bytes 4-{len(self.data)-1}/{len(self.data)}"}
        with patch.object(setup.urllib.request, "urlopen", side_effect=[URLError("connection reset"), response]) as primary, \
                patch.object(setup, "_download_with_windows_trust") as fallback:
            setup.fetch(self.item, self.target)
        self.assertEqual(primary.call_count, 2)
        self.assertEqual(primary.call_args.args[0].get_header("Range"), "bytes=4-")
        self.assertEqual(self.target.read_bytes(), self.data)
        fallback.assert_not_called()

    def test_system_downloader_uses_verification_and_only_https_redirects(self):
        system_root = self.target.parent / "Windows"
        curl = system_root / "System32/curl.exe"
        curl.parent.mkdir(parents=True)
        curl.touch()
        result = subprocess.CompletedProcess([], 0, b"200", b"")
        with patch.dict(os.environ, {"SystemRoot": str(system_root)}), \
                patch.object(setup.subprocess, "run", return_value=result) as run:
            setup._download_with_windows_trust(self.item["url"], self.partial)
        argv = list(map(str, run.call_args.args[0]))
        self.assertEqual(Path(argv[0]), curl)
        self.assertEqual(argv[1], "--disable")
        self.assertEqual(argv[argv.index("--proto") + 1], "=https")
        self.assertEqual(argv[argv.index("--proto-redir") + 1], "=https")
        self.assertIn("--location", argv)
        self.assertFalse({"-k", "--insecure", "--proxy-insecure", "--ssl-no-revoke"}.intersection(argv))

    def test_range_refused_during_native_fallback_restarts_cleanly(self):
        self.partial.write_bytes(self.data[:4])

        def native(args, **kwargs):
            if "--continue-at" in args:
                self.assertEqual(self.partial.read_bytes(), self.data[:4])
                return subprocess.CompletedProcess(args, 22, b"416", b"")
            self.assertFalse(self.partial.exists())
            self.partial.write_bytes(self.data)
            return subprocess.CompletedProcess(args, 0, b"200", b"")

        with patch.dict(os.environ, {"SystemRoot": str(self.target.parent)}), \
                patch.object(setup.urllib.request, "urlopen", side_effect=URLError(self.chain_error)), \
                patch.object(setup.subprocess, "run", side_effect=native) as run:
            setup.fetch(self.item, self.target)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(self.target.read_bytes(), self.data)

    @unittest.skipUnless(os.name == "nt" and os.environ.get("LTX_TEST_WINDOWS_HTTPS") == "1",
                         "opt-in network smoke on Windows using the OS certificate store")
    def test_real_windows_curl_accepts_valid_chain_and_saves_pinned_payload(self):
        # Exercise the Windows inbox curl and trust store without installing a test root.
        manifest = json.loads((ROOT / "runtimework/runtime-manifest.json").read_text(encoding="utf-8"))
        small_wheel = min((w for w in manifest["wheels"] if (w["url"] or "").startswith("https://files.pythonhosted.org/")),
                          key=lambda w: w["size"])
        with patch.object(setup.urllib.request, "urlopen", side_effect=URLError(self.chain_error)):
            setup.fetch(small_wheel, self.target)
        self.assertTrue(setup.matches(self.target, small_wheel))

    def test_windows_certificate_rejection_is_actionable_and_leaves_no_final_file(self):
        system_root = self.target.parent / "Windows"
        curl = system_root / "System32/curl.exe"
        curl.parent.mkdir(parents=True)
        curl.touch()
        result = subprocess.CompletedProcess([], 60, b"000", b"certificate chain was not trusted")
        with patch.dict(os.environ, {"SystemRoot": str(system_root)}), \
                patch.object(setup.urllib.request, "urlopen", side_effect=URLError(self.chain_error)), \
                patch.object(setup.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError) as failure:
                setup.fetch(self.item, self.target)
        message = str(failure.exception).lower()
        for word in ("certificate", "windows"):
            self.assertIn(word, message)
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
