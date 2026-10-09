# Windows beta redistribution review — 2026-09-21

Status: **tested locally; public binary release not cleared by this review**.
This is an engineering evidence checklist, not legal advice.

## October 9, 2026 beta refresh

Version 0.1.0b2 is being prepared for local testing with the existing locked
runtime dependencies. The open redistribution items below have not been cleared;
no NVIDIA response was present in the licensing inquiry inbox when checked on
October 9. The already-existing GitHub `v0.10b1` release is not evidence that this
engineering review's gates were satisfied. This refresh does not publish another
binary release or alter that existing release.

The new build's exact source revision, package versions, and acceptance results
are recorded in its adjacent `BUILD-INFO.json` and `smoke-test.json`. The artifact
and counts below remain historical evidence for the September review, not claims
about the new ZIP.

## Artifact reviewed

- Version: 0.1.0b1, Windows x64, unsigned one-folder ZIP.
- ZIP SHA-256: `3776301de8892b2ec4024445ca838d7fd4011384261fb56055b63342f47ee57b`
- User reports successful functional testing.
- Existing automated evidence: 295 regression tests, nine translation catalogs,
  and 17 packaged smoke checks including actual ONNX CUDA inference.
- Additional inventory: 214 native files and 142 license assets. This counts
  files, not distinct components; duplicate runtime DLLs are included.
- The tested ZIP has not been rebuilt or modified during this review. Notice
  updates in the repository must be incorporated into a later release build.

Reproduce the inventory with Python 3.12, choosing a new output filename:

```powershell
python packaging/windows/audit_bundle.py PATH_TO_EXTRACTED_BUNDLE OUTPUT.json
```

The report records relative paths, sizes and SHA-256 hashes. It deliberately
does not infer permission to redistribute from filenames or license presence.

## Findings and remaining release gates

| Component | Observed evidence | Work remaining |
| --- | --- | --- |
| Qt/PySide/Shiboken 6.11.2 | Five Qt DLL modules: Core, Gui, Network, Svg, Widgets; plugins include Qt Image Formats; LGPL/GPL texts supplied separately from wheel assets | Assemble matching source/build information for qtbase, qtsvg, qtimageformats and pyside-setup; native dependency attributions; document and verify compatible library replacement |
| Shapely 2.1.2 / GEOS 3.13.1 | GEOS DLLs and LGPL 2.1 text present | Include GEOS source and wheel build/repair information, plus replacement instructions; do not mistake Shapely's BSD license for GEOS's license |
| PhotoshopAPI 0.9.1 | Numerous native dependencies, but only PhotoshopAPI's own license in its collected license directory | Establish exact dependency build revisions and collect their notices; assess any source obligations, including liblzma; do not copy arbitrary latest licenses |
| NVIDIA | 20 runtime DLL files under nvidia; wheel license files preserved | Check each DLL against version-applicable redistribution lists; resolve cuDNN wheel's 2020 supplement vs documented 2022 supplement; confirm distribution terms |
| PDFium | Full wheel-supplied BUILD_LICENSES directory retained | Preserve it unchanged and recheck if the wheel changes |
| Python/OpenSSL/MSVC/software OpenGL | Separate native binaries included | Confirm matching native dependency notices and redistribution provenance, including Qt's opengl32sw.dll; Python's general license is not a substitute for all native notices |

The NVIDIA inventory includes cuBLAS/cuBLASLt/NVBLAS, CUDA runtime, cuFFT/cuFFTW,
cuRAND, NVRTC and its builtins, nvJitLink, and ten cuDNN DLLs. It excludes GPU
drivers. This is an observed inventory, not a redistribution allowlist.

## Primary references checked

- [Qt LGPL obligations](https://www.qt.io/development/open-source-lgpl-obligations)
- [Qt Image Formats licenses and attributions](https://doc.qt.io/qt-6/qtimageformats-index.html#licenses-and-attributions)
- [GEOS licensing and source project](https://libgeos.org/)
- [CUDA agreement and redistribution list](https://docs.nvidia.com/cuda/eula/index.html)
- [cuDNN agreement and supplement](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html)

The online cuDNN supplement permits runtime .dll redistribution subject to its
agreement, but that does not establish which terms govern the differently
worded file shipped in the pinned wheel. Do not silently replace that file.

## Vendor clarification (sent 2026-09-21)

NVIDIA's cuDNN documentation lists
`nvidia-compute-license-questions@nvidia.com` for licensing questions:

> We plan to distribute an open-source Windows x64 desktop application with
> ONNX Runtime and the runtime DLLs from nvidia-cudnn-cu13 9.26.0.51. The wheel's
> License.txt includes a January 2020 supplement naming cudnn64_7.dll, while
> the online documentation has a February 2022 supplement covering runtime
> .dll files generally. Which agreement/supplement applies to this wheel,
> and may its cuDNN 9 runtime DLLs be redistributed with the application?
> Please identify the applicable notice and distribution requirements.

The inquiry was sent to the address above from the authorized agent inbox
`tamegate10@agentmail.to` on 2026-09-21. The email identifies its sender as an
AI assistant acting with the maintainer's authorization and makes no additional
contractual commitments. A response is pending. No GitHub push, tag or public
release was performed. No dependency was removed or downgraded.

## Follow-up preparation completed

- Retained six official Qt/PySide/GEOS/Shapely source archives; checksum origins
  and SHA-256 values are recorded in `source-archives.json`.
- Added a repeatable offline preparation script with seven tests, including
  checksum failure, path traversal, attribution references and overwrite checks.
- Re-ran the complete regression suite after these changes: 302 tests passed.
- Added 188 source attribution assets and 55 PhotoshopAPI native dependency
  notice assets to the packaging payload. See
  `licenses/SOURCE_NOTICE_PROVENANCE.md` for scope and caveats.
- Downloaded source archives for 18 PhotoshopAPI dependencies from its pinned
  recipes; 15 archives matched recipe SHA-512 values. Three overridden versions
  have locally recorded hashes. Their correspondence to the published wheel is
  inferred from the build recipe, not independently publisher-confirmed.
- Checked XZ 5.8.1's COPYING: its liblzma library is 0BSD, so the previously
  flagged possible liblzma copyleft issue does not follow from this source.
- Added `SOURCE_REPLACEMENT.md` with package layout and replacement validation
  instructions. Modified-library validation, remaining native provenance and
  final source completeness review are still open.

These are documentation/build-support changes only; the existing tested ZIP
has not been altered. The source-evidence archive is not yet represented as a
complete corresponding-source release. NVIDIA clarification remains pending,
so the final commit/build/publication sequence has not been reached.

## Recommended next sequence

1. Establish version-matched source/build provenance and collect missing
   native notices; obtain clarification where supplied terms conflict.
2. Finalize the source-distribution/replacement instructions and release terms.
3. Commit the reviewed packaging sources and build from that exact clean commit.
4. Re-run smoke tests and user checks on that final archive, generate a fresh
   checksum, and publish the binary together with the required source assets.

The current build manifest correctly records a dirty checkout. Publishing a
tag at the previous README commit would not reproduce this beta's packaging.
