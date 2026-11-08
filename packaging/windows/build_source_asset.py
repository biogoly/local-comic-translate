"""Build a source-based Windows release from an explicit Git commit, without bundling wheels."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BUILD_ONLY = {"altgraph", "pefile", "pyinstaller", "pyinstaller-hooks-contrib"}
TOP = {
    "LICENSE",
    "README.md",
    "comic.py",
    "controller.py",
    "run.bat",
    "requirements.txt",
    "requirements-flux.txt",
    "models/loras/flux2-klein/README.md",
}
SETUP = {
    "packaging/windows/start_source.ps1",
    "packaging/windows/SOURCE_SETUP.md",
    "packaging/windows/DAYU_LICENSE.txt",
}
RESOURCE_SUFFIXES = {".svg", ".png", ".gif", ".ico", ".icns", ".qss", ".json", ".qm"}


def runtime_lock(source):
    """Keep the exact runtime pins and SHA-256 hashes from the committed Windows lock."""
    result = [
        "# Windows x64 / CPython 3.12 runtime, derived from packaging/windows/requirements.lock."
    ]
    seen = set()
    for section in re.split(r"(?m)(?=^[\w.-]+==)", source):
        if (
            not section.strip()
            or section.lstrip().startswith("#")
            and not re.search(r"(?m)^[\w.-]+==", section)
        ):
            continue
        first = section.splitlines()[0]
        pin = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)\s*\\?", first.strip())
        if not pin:
            raise ValueError(f"Unrecognized locked requirement: {first}")
        name = re.sub(r"[-_.]+", "-", pin.group(1)).lower()
        hashes = re.findall(
            r"(?m)^\s+--hash=sha256:([0-9a-f]{64})(?:\s*\\)?\s*$", section
        )
        if not hashes or name in seen:
            raise ValueError(f"Missing hashes or duplicate requirement: {name}")
        seen.add(name)
        if name not in BUILD_ONLY:
            result.append(f"{name}=={pin.group(2)} \\")
            for index, digest in enumerate(hashes):
                result.append(
                    "    --hash=sha256:"
                    + digest
                    + (" \\" if index + 1 < len(hashes) else "")
                )
    if (
        not BUILD_ONLY <= seen
        or not {"packaging", "pywin32-ctypes", "onnxruntime-gpu", "pyside6-essentials"}
        <= seen
    ):
        raise ValueError("Unexpected Windows dependency lock contents")
    return ("\n".join(result) + "\n").encode("ascii")


def included(path):
    if (
        path in TOP
        or path in SETUP
        or path.startswith("docs/")
        and path.endswith(".md")
    ):
        return True
    if path.partition("/")[0] in {"app", "modules", "pipeline", "imkit"}:
        return path.endswith(".py")
    return (
        path.startswith(
            (
                "resources/icons/",
                "resources/static/",
                "resources/translations/compiled/",
            )
        )
        and Path(path).suffix.lower() in RESOURCE_SUFFIXES
    )


def commit_files(repo, commit):
    def git(*args, data=None):
        return subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(repo), *args], input=data
        )

    if (
        not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", commit)
        or git("cat-file", "-t", commit).strip() != b"commit"
    ):
        raise ValueError("Supply a full Git commit SHA")
    blobs = {}
    for row in git("ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if row:
            props, raw_path = row.split(b"\t", 1)
            name = raw_path.decode()
            if included(name) or name == "packaging/windows/requirements.lock":
                mode, kind, oid = props.split()
                if (
                    kind != b"blob"
                    or mode not in (b"100644", b"100755")
                    or "\\" in name
                    or ":" in name
                ):
                    raise ValueError(f"Unexpected release input: {name}")
                blobs[name] = oid
    raw = BytesIO(
        git("cat-file", "--batch", data=b"".join(v + b"\n" for v in blobs.values()))
    )
    files = {}
    for name, expected in blobs.items():
        oid, kind, size = raw.readline().split()
        if oid != expected or kind != b"blob":
            raise ValueError("Git object mismatch")
        files[name] = raw.read(int(size))
        if raw.read(1) != b"\n":
            raise ValueError("Incomplete Git blob")
    return files


def build(repo, commit, output):
    files = commit_files(repo, commit)
    required = TOP | SETUP | {"app/version.py", "packaging/windows/requirements.lock"}
    if required - files.keys():
        raise ValueError(
            "Missing release inputs: " + ", ".join(sorted(required - files.keys()))
        )
    version = (
        re.search(rb'__version__ = "([0-9a-z.]+)"', files["app/version.py"])
        .group(1)
        .decode()
    )
    if (
        len(
            [
                p
                for p in files
                if p.startswith("resources/translations/compiled/")
                and p.endswith(".qm")
            ]
        )
        != 9
    ):
        raise ValueError("Expected nine compiled translation catalogs")
    lock = files.pop("packaging/windows/requirements.lock")
    files["packaging/windows/runtime.lock"] = runtime_lock(lock.decode("utf8"))
    files["START-HERE.md"] = files["packaging/windows/SOURCE_SETUP.md"]
    files["SOURCE-INFO.json"] = (
        json.dumps(
            dict(
                version=version,
                source_commit=commit,
                source_windows_lock_sha256=sha256(lock).hexdigest(),
                files={
                    p: sha256(data).hexdigest() for p, data in sorted(files.items())
                },
            ),
            indent=2,
        )
        + "\n"
    ).encode()
    archive = output / f"LCT-{version}-Windows-setup.zip"
    checksum = archive.with_suffix(".zip.sha256")
    output.mkdir(parents=True, exist_ok=True)
    if archive.exists() or checksum.exists():
        raise FileExistsError("Release asset already exists")
    with tempfile.TemporaryDirectory(dir=output) as temp:
        stage = Path(temp) / archive.name
        with zipfile.ZipFile(
            stage, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as z:
            for path, data in sorted(files.items()):
                entry = zipfile.ZipInfo("LCT/" + path, (2020, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                z.writestr(
                    entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
                )
        with zipfile.ZipFile(stage) as z:
            if z.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
        stage.rename(archive)
    with checksum.open("x", encoding="ascii") as stream:
        stream.write(f"{sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    pack = modes.add_parser("build")
    pack.add_argument("--commit", required=True)
    pack.add_argument("--repository", type=Path, default=ROOT)
    pack.add_argument("--output", type=Path, required=True)
    lock = modes.add_parser("runtime-lock")
    lock.add_argument("source", type=Path)
    lock.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "build":
        print(build(args.repository.resolve(), args.commit, args.output.resolve()))
    else:
        # A second launcher can start while the first prepares the same runtime.
        # Replace atomically so both see a complete file with identical contents.
        data = runtime_lock(args.source.read_text(encoding="utf-8-sig"))
        with tempfile.TemporaryDirectory(dir=args.output.parent) as folder:
            prepared = Path(folder) / "runtime.lock"
            prepared.write_bytes(data)
            prepared.replace(args.output)
