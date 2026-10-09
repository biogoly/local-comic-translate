# Third-party components

Local Comic Translate is an Apache-2.0 fork of
https://github.com/ogkalu2/comic-translate. Its LICENSE does not replace the
licenses of bundled components. See `licenses/` and `BUILD-INFO.json` for the
license files supplied by the installed wheels and the exact build versions.
In a Windows folder build these files are in `_internal/`, beside this notice.

The bundle includes CPython, PySide6/Qt, ONNX Runtime, NVIDIA CUDA/cuDNN runtime
libraries, NumPy, Pillow, PDFium/pypdfium2, PhotoshopAPI and its native libraries,
and other dependencies listed in the build manifest. Build tools are listed in
the manifest as well; that inventory is not a claim that every package is used
at runtime. Python and Qt shared libraries remain separate from the executable.

## Qt, PySide6 and Shiboken6

This application uses Qt, PySide6 and Shiboken6, copyright The Qt Company Ltd.
and other contributors. Their open-source licensing is separate from the
application's Apache-2.0 license. The intended option for the LGPL-covered
libraries in this build is LGPL version 3. The LGPL and the GPL version 3 text
it incorporates are included as `licenses/Qt/LGPL-3.0-only.txt` and
`licenses/Qt/GPL-3.0-only.txt`. See `licenses/Qt/README.md` for their provenance.

The pinned Windows build uses PySide6 Essentials and Shiboken6 6.11.2, with Qt
6.11.2. Those wheels declare open-source license options in package metadata
but provide only `LicenseRef-Qt-Commercial.txt` in their license directories.
That file is preserved as supplied; its presence is not a claim that this
application or its users hold a commercial Qt license. The additional license
texts in `licenses/Qt/` address that missing-text issue, not every distribution
requirement or every third-party component in Qt.

Matching upstream source locations:

- PySide6 and Shiboken6: [pyside-setup-everywhere-src-6.11.2.zip](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/pyside-setup-everywhere-src-6.11.2.zip).
- Qt Base (Core, GUI, Widgets, Network and related libraries): [qtbase-everywhere-src-6.11.2.zip](https://download.qt.io/archive/qt/6.11/6.11.2/submodules/qtbase-everywhere-src-6.11.2.zip).
- Qt SVG: [qtsvg-everywhere-src-6.11.2.zip](https://download.qt.io/archive/qt/6.11/6.11.2/submodules/qtsvg-everywhere-src-6.11.2.zip).
- Qt Image Formats: the finished beta includes ICNS, TGA, TIFF, WBMP and WebP
  plugins from this module. Include the matching `qtimageformats` 6.11.2 source
  from the Qt source module archive below, plus its TIFF/WebP acknowledgements.
- Other Qt libraries/plugins, if collected: [Qt 6.11.2 source module archive](https://download.qt.io/archive/qt/6.11/6.11.2/submodules/).

These links identify upstream source; they are not a statement that the complete
corresponding source and all wheel build details have already been assembled
for redistribution. Qt third-party acknowledgements and source/replacement
review remain on the public-release checklist below.

For testing library replacement, work in a copy of the extracted application
folder. Qt libraries/plugins are under `_internal/PySide6/`, and Shiboken under
`_internal/shiboken6/`; keep the matching Python extension modules and dependent
DLLs together. Use an interface-compatible build for the same architecture and
Python ABI. The source packaging instructions are in `packaging/windows/README.md`
in the application repository. No additional restriction on modifying these
LGPL libraries or reverse-engineering to debug those modifications is imposed
by this application's license. Replacement of the finished bundle still needs
an end-to-end validation before public release.

## PDFium and pypdfium2

The pinned pypdfium2 5.13.0 wheel includes PDFium 153.0.7999.0. Its collected
license directory contains the pypdfium2 Apache-2.0/BSD-3-Clause texts and its
documentation license, plus `data/windows_x64/BUILD_LICENSES/` with PDFium and
the native dependency acknowledgements. Keep that complete directory, including
`pdfium.txt`, `pdfium-binaries.txt`, and every adjacent dependency notice, with
the binaries. The [pypdfium2 licensing documentation](https://pypdfium2.readthedocs.io/en/stable/readme.html#licensing)
specifically describes those binary-distribution notices.

## NVIDIA runtime libraries

NVIDIA runtime DLLs remain subject to NVIDIA's own terms, not Apache-2.0 or the
Qt licenses. The build preserves each installed NVIDIA package's `License.txt`,
including its bundled third-party notices. The pinned wheels cover CUDA runtime,
cuBLAS, cuFFT, cuRAND, NVRTC, nvJitLink and cuDNN; versions are recorded in
`BUILD-INFO.json`. GPU drivers, developer tools, and model weights are not
intentionally included.

Before redistribution, compare the actual DLL inventory with the applicable
[CUDA redistribution list and terms](https://docs.nvidia.com/cuda/eula/index.html)
and [cuDNN terms](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html).
In particular, the cuDNN 9.26.0.51 wheel inspected for this beta includes a
January 2020 license supplement naming `cudnn64_7.dll`, while the current cuDNN
documentation describes runtime `.dll` files generally. Preserve the supplied
file and resolve the version-specific terms before a public release; do not
treat the wheel's availability as the entire redistribution review.

## Shapely and GEOS

The inspected Shapely 2.1.2 wheel reports GEOS 3.13.1. Shapely's BSD license
does not replace GEOS's LGPL license. The bundle preserves `LICENSE_GEOS`
(LGPL 2.1), `LICENSE_win32` and Shapely's `LICENSE.txt`. GEOS shared libraries
remain separate under `_internal/Shapely.libs/`. Include GEOS in the
corresponding-source and compatible-library replacement review, including
wheel build/repair details; do not limit that review to Qt.

## PhotoshopAPI native dependencies

The inspected PhotoshopAPI 0.9.1 license directory contains its own license,
but no separate dependency-notice collection. Its `.libs` directory includes
OpenImageIO, OpenEXR, Imath, OpenColorIO, libjpeg, Expat, liblzma, libpng,
OpenJPH, TIFF, yaml-cpp, zlib and zstd, among others. The application-level
PhotoshopAPI license must not be treated as covering every native dependency.
Obtain the version-matched dependency notices and any required source before
public redistribution. A filename alone is not reliable version provenance.

Primary component source/license information:

- Python: https://www.python.org/downloads/source/
- PySide6/Qt: https://code.qt.io/cgit/pyside/pyside-setup.git/ and https://www.qt.io/licensing/open-source-lgpl-obligations
- ONNX Runtime: https://github.com/microsoft/onnxruntime
- NVIDIA: https://docs.nvidia.com/cuda/eula/index.html and https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html
- PDFium/pypdfium2: https://github.com/pypdfium2-team/pypdfium2
- PhotoshopAPI: https://github.com/EmilDohne/PhotoshopAPI

## Public-release checklist

This beta is a local test artifact. Automated license-file collection is not a
completed legal compliance audit or approval to publish a release. Before
sharing a public binary release:

1. Inventory the actual bundled DLLs, Python extensions and Qt plugins. Review
   each selected Qt module's license; avoid assuming every module is LGPL merely
   because it arrived in an Essentials wheel.
2. Assemble and provide the complete corresponding Qt/PySide/Shiboken source,
   relevant patches and build information through an appropriate distribution
   method. Preserve version-matched archives; upstream links alone are not our
   completed source-availability process.
3. Validate and document replacement/relinking with interface-compatible LGPL
   libraries in the finished package, including any installation information
   needed. Review [Qt's LGPL obligations guidance](https://www.qt.io/development/open-source-lgpl-obligations).
4. Add the applicable Qt native dependency notices and copyrights from the
   [Qt third-party component index](https://doc.qt.io/qt-6/licenses-used-in-qt.html)
   and matching source attribution files. Obtain a build-matched Qt SBOM where
   available; this PySide wheel does not include one. A generic license text
   is not a substitute for a component's attribution notice.
5. Resolve the NVIDIA terms/version discrepancy above and check every shipped
   runtime against its redistribution conditions. Retain PDFium's full
   `BUILD_LICENSES` set and review notices for other native components, including
   PhotoshopAPI dependencies, software OpenGL and Microsoft runtime DLLs.
6. Recheck these items whenever dependency versions or the bundled module set
   changes. Models are downloaded separately and retain their own licenses and
   access terms.
