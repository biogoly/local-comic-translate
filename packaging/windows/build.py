"""Build, smoke-test and archive an allowlisted Windows x64 beta."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
# Leave room for Explorer's destination folder and its suggested ZIP-name folder.
MAX_MEMBER_UNITS = 145


def run_source_tests():
    """Run the editor tests with real glyphs in the Windows offscreen Qt session."""
    import unittest
    from PySide6.QtGui import QFont, QFontDatabase, QRawFont
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    # The Windows offscreen platform can enumerate zero installed families.
    # Register the host's fonts for tests only; they are not package assets.
    font_dir = Path(os.environ["SystemRoot"]) / "Fonts"
    for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf",
                 "segoeui.ttf", "segoeuib.ttf", "meiryo.ttc", "meiryob.ttc", "malgun.ttf"):
        if (font_dir / name).is_file():
            QFontDatabase.addApplicationFont(str(font_dir / name))
    app.setFont(QFont("Segoe UI", 9))
    glyphs = QRawFont.fromFont(QFont("Arial", 20)).glyphIndexesForString("Hello")
    if len(set(glyphs)) < 2:
        raise RuntimeError("Qt could not load a usable system font for the editor tests.")
    suite = unittest.defaultTestLoader.discover("tests")
    return unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()


def run(command, *, env=None, cwd=ROOT):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def copy_source_notices(source: Path, destination: Path) -> None:
    """Shorten redundant archive prefixes while preserving upstream relative links."""
    groups = []
    aliases = set()
    for archive in sorted(source.iterdir()):
        roots = list(archive.iterdir()) if archive.is_dir() else []
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError(f"Expected one upstream notice root under {archive.name}")
        alias = archive.name.partition("-")[0]
        if not alias.isascii() or not alias.isalnum() or alias.casefold() in aliases:
            raise ValueError(f"Ambiguous notice directory name: {archive.name}")
        aliases.add(alias.casefold())
        groups.append((alias, archive.name, roots[0]))
    if not groups:
        raise ValueError("The upstream source notice collection is empty.")
    destination.mkdir(parents=True, exist_ok=False)
    index = ["Package folder\tOriginal archive and top-level directory"]
    for alias, archive_name, root in groups:
        shutil.copytree(root, destination / alias)
        index.append(f"{alias}/\t{archive_name}/{root.name}/")
    (destination / "INDEX.txt").write_text("\n".join(index) + "\n", encoding="utf-8")


def prepare_payload(payload: Path) -> str:
    version = re.search(r'__version__ = "([^"]+)"', (ROOT / "app/version.py").read_text()).group(1)
    for name in ("LICENSE", "README.md", "requirements-flux.txt"):
        shutil.copy2(ROOT / name, payload / name)
    for name in ("README_fr.md", "README_ko.md", "README_zh-CN.md", "README_es.md",
                 "COMFYUI_BACKEND.md", "COMFYUI_RUNTIME_VALIDATION.md", "COMFYUI_BACKEND_PLAN.md",
                 "LANGUAGE_SUPPORT.md", "RELEASE_NOTES_0.1.0b2.md", f"RELEASE_NOTES_{version}.md",
                 "TEXT_FIT_M0_M1A.md", "TEXT_FIT_M1B.md", "AUTOMATIC_BUBBLE_FITTING.md",
                 "REVIEW_NAVIGATOR.md", "TRANSLATION_CONTEXT.md", "SPATIAL_TRANSLATION_CONTEXT.md",
                 "GRADIENT_REPAIR.md", "PROTECTED_REPAIRS.md", "LETTERING_STYLES.md",
                 "LETTERING_TRANSFORMS.md", "LETTERING_CHECK.md", "ARTISTIC_EDIT_PREVIEWS.md"):
        (payload / "docs").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "docs" / name, payload / "docs" / name)
    lora_docs = payload / "models/loras/flux2-klein"
    lora_docs.mkdir(parents=True)
    shutil.copy2(ROOT / "models/loras/flux2-klein/README.md", lora_docs / "README.md")
    shutil.copy2(ROOT / "packaging/windows/QUICKSTART.txt", payload / "QUICKSTART.txt")
    shutil.copy2(ROOT / "packaging/windows/THIRD_PARTY_NOTICES.md", payload / "THIRD_PARTY_NOTICES.md")
    (payload / "packaging/windows").mkdir(parents=True)
    shutil.copy2(ROOT / "packaging/windows/README.md", payload / "packaging/windows/README.md")
    for name in ("SOURCE_REPLACEMENT.md", "source-archives.json", "RELEASE_REVIEW.md"):
        shutil.copy2(ROOT / "packaging/windows" / name, payload / "packaging/windows" / name)
    extra_licenses = ROOT / "packaging/windows/licenses"
    if extra_licenses.is_dir():
        shutil.copytree(extra_licenses, payload / "licenses",
                        ignore=lambda at, names: ["source-notices"] if Path(at) == extra_licenses else [])
        copy_source_notices(extra_licenses / "source-notices", payload / "licenses/src")

    packages = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        licenses = []
        for file in dist.files or ():
            # Metadata only: no direct_url.json, environment paths or credentials.
            if any(part.lower().startswith(("license", "copying", "notice", "copyright")) for part in file.parts):
                source = Path(dist.locate_file(file))
                if source.is_file() and ".." not in file.parts:
                    destination = payload / "licenses" / name / str(file)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    licenses.append(destination.relative_to(payload).as_posix())
        packages.append({"name": name, "version": dist.version, "license_files": licenses})
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise RuntimeError("The Python distribution LICENSE.txt is required for the release.")
    (payload / "licenses/Python").mkdir(parents=True, exist_ok=True)
    shutil.copy2(python_license, payload / "licenses/Python/LICENSE.txt")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    manifest = {
        "product": "Local Comic Translate", "version": version, "architecture": "x64",
        "python": platform.python_version(), "source_commit": commit, "source_dirty": dirty,
        "built_utc": datetime.now(timezone.utc).isoformat(), "signed": False,
        "models_bundled": False, "build_environment_packages": packages,
    }
    (payload / "BUILD-INFO.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    numbers = [int(n) for n in re.findall(r"\d+", version)][:4]
    numbers += [0] * (4 - len(numbers))
    windows_version = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={tuple(numbers)!r}, prodvers={tuple(numbers)!r},
    mask=0x3f, flags=0x2, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('CompanyName', 'Local Comic Translate contributors'),
    StringStruct('FileDescription', 'Local Comic Translate Windows x64 beta'),
    StringStruct('FileVersion', {version!r}),
    StringStruct('ProductName', 'Local Comic Translate'),
    StringStruct('ProductVersion', {version!r}),
    StringStruct('OriginalFilename', 'LocalComicTranslate.exe')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])])
"""
    (payload / "windows-version.txt").write_text(windows_version, encoding="utf-8")
    return version


def archive_verified_bundle(bundle: Path, output: Path, version: str) -> Path:
    """Archive only after callers have obtained a passing packaged smoke report."""
    archive = output / f"LCT-{version}-x64.zip"
    members = []
    for file in sorted(bundle.rglob("*")):
        if not file.is_file():
            continue
        member = file.relative_to(bundle.parent).as_posix()
        units = len(member.encode("utf-16-le")) // 2
        if units > MAX_MEMBER_UNITS:
            raise ValueError(f"Windows ZIP path is too long ({units} > {MAX_MEMBER_UNITS}): {member}")
        if file.suffix.lower() in {".gguf", ".safetensors", ".ctpr", ".onnx", ".pt", ".pth"}:
            raise RuntimeError(f"Unexpected model or project in bundle: {file.name}")
        members.append((file, member))
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for file, member in members:
            target.write(file, member)
    if archive.stat().st_size >= 2 * 1024**3:
        raise RuntimeError("ZIP exceeds the per-asset GitHub Release size limit; do not publish it.")
    with archive.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    (output / (archive.name + ".sha256")).write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release-artifacts")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/windows")
    parser.add_argument("--skip-tests", action="store_true", help="Developer iteration only; smoke test is never skipped")
    parser.add_argument("--require-cuda", action="store_true", help="Also require real CUDA inference on this build host")
    args = parser.parse_args()
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"} or sys.maxsize <= 2**32:
        raise SystemExit("Build using Windows and x64 Python.")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Build using Python 3.12.")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="beta-", dir=args.work_dir.resolve()))
    output = args.output_dir.resolve() / work.name
    output.mkdir(parents=True)
    payload = work / "payload"
    payload.mkdir()
    version = prepare_payload(payload)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    # Freeze only the dedicated interpreter/wheels and Windows system DLLs.
    # Arbitrary developer PATH entries can inject older MSVC/Qt libraries.
    windows = Path(os.environ["SystemRoot"])
    environment["PATH"] = os.pathsep.join(map(str, (
        Path(sys.executable).parent, Path(sys.base_prefix), Path(sys.base_prefix) / "DLLs",
        windows / "System32", windows, windows / "System32/Wbem",
    )))
    for variable in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH"):
        environment.pop(variable, None)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["LCT_BUILD_PAYLOAD"] = str(payload)
    if not args.skip_tests:
        # Create QApplication before discovery: some protocol tests create a
        # QCoreApplication otherwise, which cannot later host widget tests.
        run([sys.executable, "-c", "import runpy; check = runpy.run_path('packaging/windows/build.py'); "
             "raise SystemExit(not check['run_source_tests']())"], env=environment)
        run([sys.executable, "resources/translations/check_translations.py"], env=environment)
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--workpath", work / "pyinstaller", "--distpath", work / "dist",
         ROOT / "packaging/windows/local-comic-translate.spec"], env=environment)
    bundle = work / "dist/LocalComicTranslate"
    # A visible quick start sits next to the executable; full docs live inside.
    shutil.copy2(payload / "QUICKSTART.txt", bundle / "QUICKSTART.txt")
    shutil.copy2(payload / "LICENSE", bundle / "LICENSE")
    report = output / "smoke-test.json"
    command = [bundle / "LocalComicTranslate.exe", "--smoke-test", report]
    if args.require_cuda:
        command.append("--require-cuda")
    # Test with an unrelated working directory, not the source checkout.
    run(command, cwd=output, env=environment)
    if not report.is_file() or not json.loads(report.read_text(encoding="utf-8")).get("ok"):
        raise RuntimeError("Frozen smoke test did not produce a passing report.")
    archive = archive_verified_bundle(bundle, output, version)
    shutil.copy2(payload / "BUILD-INFO.json", output / "BUILD-INFO.json")
    print(f"Beta ready for testing (not published): {archive}", flush=True)


if __name__ == "__main__":
    main()
