"""Inventory a finished bundle for redistribution review; not legal clearance.

Usage: python packaging/windows/audit_bundle.py BUNDLE OUTPUT.json
The output is deliberately external to the bundle: never change a tested ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inventory(bundle: Path) -> dict:
    bundle = bundle.resolve(strict=True)
    if not (bundle / "LocalComicTranslate.exe").is_file():
        raise ValueError("Expected the extracted LocalComicTranslate bundle directory")
    native, notices = [], []
    for file in sorted(bundle.rglob("*")):
        if not file.is_file():
            continue
        relative = file.relative_to(bundle)
        is_native = file.suffix.lower() in {".exe", ".dll", ".pyd"}
        is_notice = "licenses" in (part.lower() for part in relative.parts)
        if not (is_native or is_notice):
            continue
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entry = {"path": relative.as_posix(), "bytes": file.stat().st_size,
                 "sha256": digest}
        (native if is_native else notices).append(entry)
    return {"schema_version": 1, "legal_clearance": False,
            "description": "Observed files, not a complete SBOM or license determination",
            "native_files": native, "license_files": notices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.bundle.resolve()):
        parser.error("Write the audit outside the tested bundle")
    report = inventory(args.bundle)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Inventoried {len(report['native_files'])} native files and "
          f"{len(report['license_files'])} license assets; no legal clearance implied.")


if __name__ == "__main__":
    main()
