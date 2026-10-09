# Source-derived notices (collected 2026-09-21)

These supplemental assets are copied byte-for-byte from upstream sources, not
rewritten license summaries. The build copies this directory into the package.
They supplement, rather than replace, the notices supplied by installed wheels.

## Qt, PySide, Shapely and GEOS

`source-notices/` contains 188 assets from the six archives pinned in
`packaging/windows/source-archives.json`. Generated with
`prepare_source_evidence.py`, it includes conventionally named license files,
Qt attribution records, and the files those records identify as LicenseFile.
Some referenced files are source headers or README files.

This is a conservative source attribution collection, not a statement that
every test/example dependency in the source archives is shipped at runtime.
The original source archives are preserved separately for source distribution
review, not embedded in the executable.

## PhotoshopAPI dependencies

`photoshop-native-notices/sources.json` records 18 dependency source URLs,
versions and SHA-256 values. The versions come from PhotoshopAPI v0.9.1's
vcpkg baseline `e3ed41868d5034bc608eaaa58383cd6ecdbb5ffb`, with its manifest
overrides applied. Fifteen downloaded archives also matched SHA-512 values in
the baseline's port recipes. libdeflate 1.22, fmt 11.0.2 and Eigen 3.4.0 are
overrides: their source hashes are locally recorded, not verified against the
baseline's different-version checksums.

The 55 notice assets include OpenImageIO's THIRD-PARTY.md and libjpeg-turbo's
README.ijg. Runtime correspondence is inferred from the tagged build recipes
and available DLL version metadata, not independently confirmed by the wheel
publisher. Original source archives, including Eigen's MPL-covered code, have
been retained outside the repository. Preserve the sources and applicable
build recipes when preparing the release's source companion.

For liblzma 5.8.1, upstream COPYING identifies the library as 0BSD. The GPL/LGPL
texts present in the XZ source collection also cover other source components;
their inclusion is not a claim that the bundled liblzma DLL is GPL/LGPL.

`photoshop-submodules/` preserves notices from these exact submodule revisions:

- mio: `8b6b7d878c89e81614d05edca7936de41ccdd2da`
- pybind11: `8a099e44b3d5f85b20f05828d919d2332a8de841`
- simdutf: `c00580bde0bcc5a8299281910f42f83b7c8ff64c`
- compressed-image: `f424ccfd9299811e70b3b611a8b0a85db7e1aabb`

Build provenance and redistribution questions still open in RELEASE_REVIEW.md
are not resolved solely by copying these notices.
