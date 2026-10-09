"""Offline tests for source-evidence preparation (no large downloads)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "source_evidence", Path(__file__).resolve().parents[1] / "packaging/windows/prepare_source_evidence.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceEvidenceTests(unittest.TestCase):
    def test_reject_unsafe_names(self):
        for name in ("../escape", "/absolute", "C:/escape", "..\\escape", "", "."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                MODULE.safe_name(name)

    def test_include_attribution_referenced_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("src/qt_attribution.json", json.dumps({"LicenseFile": "README.md"}))
                stream.writestr("src/README.md", "Copyright example")
                stream.writestr("src/unused.cpp", "not a notice")
            assets = MODULE.zip_notices(archive)
            self.assertEqual(assets["src/README.md"], b"Copyright example")
            self.assertNotIn("src/unused.cpp", assets)

    def test_missing_reference_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("src/qt_attribution.json", '{"LicenseFile": "MISSING"}')
            with self.assertRaisesRegex(ValueError, "Missing Qt"):
                MODULE.zip_notices(archive)

    def test_reference_escape_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("qt_attribution.json", '{"LicenseFile": "../escape"}')
            with self.assertRaises(ValueError):
                MODULE.zip_notices(archive)

    def test_checksum_failure_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "bad.zip").write_bytes(b"invalid")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"archives": [{"filename": "bad.zip", "sha256": "0" * 64}]}))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.prepare(source, root / "output", manifest)
            self.assertFalse((root / "output").exists())

    def test_prepare_retains_original_archive_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            archive = source / "test.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("src/LICENSE", "example license")
                stream.writestr("src/code.cpp", "example source")
            original = archive.read_bytes()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"archives": [{"filename": "test.zip", "sha256": hashlib.sha256(original).hexdigest()}]}))
            output = root / "output"
            report = MODULE.prepare(source, output, manifest)
            self.assertFalse(report["legal_clearance"])
            with zipfile.ZipFile(output / "third-party-source-evidence.zip") as stream:
                self.assertEqual(stream.read("test.zip"), original)
            with self.assertRaises(FileExistsError):
                MODULE.prepare(source, output, manifest)

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaisesRegex(ValueError, "overlap"):
                MODULE.prepare(source, source / "output")


if __name__ == "__main__":
    unittest.main()
