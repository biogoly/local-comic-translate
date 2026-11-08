"""Windows ZIP paths and the original upstream notice relationships."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


WINDOWS = Path(__file__).resolve().parents[1] / "packaging/windows"
SPEC = importlib.util.spec_from_file_location("windows_build", WINDOWS / "build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class SourceNoticePathTests(unittest.TestCase):
    def test_pinned_notices_and_attribution_links_survive_shortening(self):
        sources = WINDOWS / "licenses/source-notices"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "src"
            build.copy_source_notices(sources, target)
            rows = (target / "INDEX.txt").read_text(encoding="utf-8").splitlines()[1:]
            groups = [row.split("\t") for row in rows]
            self.assertCountEqual((original.split("/")[0] for _, original in groups),
                                  (archive.name for archive in sources.iterdir()))
            mirrored = set()
            upward_references = 0
            for alias, original in groups:
                upstream = sources / original
                self.assertTrue(upstream.is_dir(), original)
                for file in upstream.rglob("*"):
                    if not file.is_file():
                        continue
                    copy = target / alias / file.relative_to(upstream)
                    self.assertEqual(copy.read_bytes(), file.read_bytes(), copy)
                    mirrored.add(copy.relative_to(target))
                    if file.name == "qt_attribution.json":
                        records = json.loads(file.read_text(encoding="utf-8"), strict=False)
                        for record in records if isinstance(records, list) else [records]:
                            refs = record.get("LicenseFile", [])
                            for ref in [refs] if isinstance(refs, str) else refs:
                                self.assertEqual((file.parent / ref).read_bytes(),
                                                 (copy.parent / ref).read_bytes(), ref)
                                upward_references += ".." in Path(ref).parts
            self.assertGreater(upward_references, 0)
            self.assertEqual(mirrored | {Path("INDEX.txt")},
                             {p.relative_to(target) for p in target.rglob("*") if p.is_file()})

    def test_unexpected_sibling_and_colliding_aliases_fail_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "src"
            upstream = source / "qtbase-1.zip" / "qtbase-1"
            upstream.mkdir(parents=True)
            (upstream / "LICENSE").write_bytes(b"first license")
            extra = upstream.parent / "UNEXPECTED-NOTICE"
            extra.write_bytes(b"must not disappear")
            with self.assertRaises(ValueError):
                build.copy_source_notices(source, target)
            self.assertFalse(target.exists())
            extra.unlink()
            second = source / "QtBase-2.zip" / "QtBase-2"
            second.mkdir(parents=True)
            (second / "LICENSE").write_bytes(b"second license")
            with self.assertRaises(ValueError):
                build.copy_source_notices(source, target)
            self.assertFalse(target.exists())


class WindowsArchivePathTests(unittest.TestCase):
    def test_exact_utf16_limit_short_name_root_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "LocalComicTranslate"
            bundle.mkdir()
            (bundle / "LocalComicTranslate.exe").write_bytes(b"program")
            # Including the product root this takes 145 UTF-16 units but only 144 code points.
            name = "x" * (145 - len(bundle.name) - 1 - 2 - len(".txt")) + "\U0001f680.txt"
            file = bundle / name
            member = f"{bundle.name}/{name}"
            self.assertEqual(len(member.encode("utf-16-le")) // 2, 145)
            file.write_bytes(b"boundary")
            archive = build.archive_verified_bundle(bundle, Path(temp), "0.1.0b3")
            self.assertEqual(archive.name, "LCT-0.1.0b3-x64.zip")
            with zipfile.ZipFile(archive) as z:
                self.assertCountEqual(z.namelist(), ["LocalComicTranslate/LocalComicTranslate.exe", member])
                self.assertEqual(z.read(member), b"boundary")
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(archive.with_suffix(".zip.sha256").read_text().split(),
                             [checksum, archive.name])
            long_file = file.with_name("x" + file.name)
            file.rename(long_file)
            other_output = Path(temp) / "fresh"
            other_output.mkdir()
            with self.assertRaises(ValueError):
                build.archive_verified_bundle(bundle, other_output, "0.1.0b3")
            self.assertEqual(list(other_output.iterdir()), [])

    def test_shorter_archive_still_excludes_models_and_projects(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "LocalComicTranslate"
            bundle.mkdir()
            (bundle / "LocalComicTranslate.exe").write_bytes(b"program")
            for suffix in (".gguf", ".safetensors", ".ctpr", ".onnx", ".pt", ".pth"):
                with self.subTest(suffix=suffix):
                    private = bundle / ("z-private" + suffix.upper())
                    private.write_bytes(b"must never be packaged")
                    with self.assertRaises(RuntimeError):
                        build.archive_verified_bundle(bundle, Path(temp), "0.1.0b3")
                    self.assertFalse(list(Path(temp).glob("*.zip")))
                    private.unlink()
