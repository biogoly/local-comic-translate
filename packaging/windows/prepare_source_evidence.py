"""Verify downloaded source archives and prepare attribution/source evidence.

Does not download, execute source code, or approve a binary for distribution.
Output must be a new directory outside the supplied source directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or ".." in path.parts or not path.parts):
        raise ValueError(f"Unsafe archive name: {name}")
    return path.as_posix()


def notice_name(name: str) -> bool:
    return PurePosixPath(name).name.lower().startswith(
        ("license", "copying", "copyright", "notice", "qt_attribution.json"))


def zip_notices(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        members = {safe_name(i.filename): i for i in archive.infolist() if not i.is_dir()}
        names = {name for name in members if notice_name(name)}
        # Some Qt attributions point to README files or source headers rather
        # than conventionally named LICENSE files. Include those verbatim too.
        for name in sorted(names):
            if PurePosixPath(name).name != "qt_attribution.json":
                continue
            # Qt attribution records can contain literal line breaks in text.
            records = json.loads(archive.read(members[name]), strict=False)
            if isinstance(records, dict):
                records = [records]
            for record in records:
                refs = record.get("LicenseFile", [])
                if isinstance(refs, str):
                    refs = [refs]
                for ref in refs:
                    target = safe_name(posixpath.normpath(posixpath.join(posixpath.dirname(name), ref)))
                    if target not in members:
                        raise ValueError(f"Missing Qt attribution target: {target}")
                    names.add(target)
        return {name: archive.read(members[name]) for name in sorted(names)}


def tar_notices(path: Path) -> dict[str, bytes]:
    result = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = safe_name(member.name)
            if member.isfile() and notice_name(name):
                result[name] = archive.extractfile(member).read()
    return result


def prepare(source: Path, output: Path, manifest: Path = HERE / "source-archives.json") -> dict:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output directories must not overlap")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    verified = []
    for row in data["archives"]:
        name = safe_name(row["filename"])
        if "/" in name:
            raise ValueError("Archive filename must be a basename")
        file = source / name
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
        verified.append((row, file))
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for row, file in verified:
        assets = zip_notices(file) if file.suffix == ".zip" else tar_notices(file)
        for name, content in assets.items():
            destination = output / "notices" / file.name / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        counts[file.name] = len(assets)
    shutil.copy2(manifest, output / "source-archives.json")
    # Original compressed source archives are preserved unchanged. This is
    # evidence for review, not a claim of complete corresponding source.
    with zipfile.ZipFile(output / "third-party-source-evidence.zip", "x", compression=zipfile.ZIP_STORED) as archive:
        for row, file in verified:
            archive.write(file, file.name)
        archive.write(manifest, "source-archives.json")
        archive.write(HERE / "SOURCE_REPLACEMENT.md", "SOURCE_REPLACEMENT.md")
    report = {"source_checksums_verified": True, "legal_clearance": False,
              "notice_asset_counts": counts}
    (output / "preparation-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
