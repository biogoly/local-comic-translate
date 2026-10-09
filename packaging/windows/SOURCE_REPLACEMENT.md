# Third-party source and library replacement

## Status

This is a preparation document for the Windows 0.1.0b1 beta, not a completed
source offer or a statement that every redistribution requirement is met.
`source-archives.json` pins the downloaded original archives. The matching
Qt/PySide source ZIP checksums were checked against Qt's published `.sha256`
files; Shapely's against PyPI metadata. GEOS's checksum was calculated locally
from the official HTTPS download. Archives are not modified.

Prepare local review evidence with Python 3.12:

```powershell
python packaging/windows/prepare_source_evidence.py SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY
```

The script verifies every checksum, retains original archives in a companion
ZIP, and extracts notices plus files referenced by Qt attribution records.
The extracted set is conservative: test/example/build-tool notices can appear
without those components being present in the executable. It is not a runtime
SBOM. Do not treat the source-evidence ZIP as complete corresponding source
until the review below is finished.

## Replacement layout

Work on a separate copy of the extracted application and close it first.
Never overwrite the only copy of an installation or a user's projects.

| Library | Location in one-folder package |
| --- | --- |
| Qt | `_internal/PySide6/Qt6*.dll` and `_internal/PySide6/plugins/` |
| PySide | `_internal/PySide6/*.pyd`, `pyside6.abi3.dll` |
| Shiboken | `_internal/shiboken6/` |
| GEOS | `_internal/Shapely.libs/geos-*.dll`, `geos_c-*.dll` |

Build replacement libraries for Windows x64 with the matching interfaces,
Python 3.12 ABI where applicable, and compatible MSVC runtime. Keep dependent
libraries and plugins consistent. For GEOS, the wheel repair step gives the
DLLs hashed names and rewrites their imports: replacing just one file with an
arbitrarily renamed DLL may not work. Rebuild/repair the Shapely wheel against
the replacement GEOS and repackage the application when necessary.

The application adds no license prohibition on modifying these LGPL libraries
or reverse-engineering to debug such modifications. There is no application
signature check that rejects replacement Qt/GEOS libraries. This does not
override independent NVIDIA or other proprietary component terms.

After replacing libraries, run:

```powershell
.\LocalComicTranslate.exe --smoke-test C:\Temp\replacement-smoke.json
```

Use a writable report location. A pass checks startup and core operations;
also test manual editing, loading/saving projects, and image export. Add
`--require-cuda` only on a compatible NVIDIA system. The automated smoke test
uses isolated settings; ordinary interactive startup shares the fork's profile.

## Build provenance still requiring completion

- Qt 6.11.2 sources: qtbase, qtsvg, qtimageformats; PySide/Shiboken: matching
  pyside-setup archive, including build scripts. Establish the wheel producer's
  configuration/patches and software-OpenGL provenance; source version equality
  alone does not establish every build detail.
- GEOS 3.13.1 and Shapely 2.1.2: retain the Windows wheel build/repair recipe
  along with source; verify a replacement end to end.
- PhotoshopAPI 0.9.1 tag resolves to
  `055cad5254390d164b9a2ee8591d5b8f4f6758cc`. Its pinned vcpkg baseline is
  `e3ed41868d5034bc608eaaa58383cd6ecdbb5ffb`; its manifest overrides some
  baseline versions. A source archive without its submodules is insufficient
  to reconstruct the complete build. PyPI supplied no sdist for this version
  when checked on 2026-09-21.
- Before public release, supply the applicable complete sources and necessary
  build/installation information alongside the binary through an appropriate
  distribution method. Do not rely solely on third-party links remaining live.

See `RELEASE_REVIEW.md` and `THIRD_PARTY_NOTICES.md` for unresolved items.
