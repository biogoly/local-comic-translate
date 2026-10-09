# Windows x64 beta builds

This folder builds a one-folder, unsigned Windows x64 beta ZIP with PyInstaller.
The main editor includes Python, Qt, ONNX Runtime and CUDA/cuDNN libraries;
Python and a CUDA Toolkit installation are not required to run it. GPU processing
still requires a compatible NVIDIA driver; CPU fallback remains available.

Model weights, API keys, user settings, llama.cpp and the separate optional FLUX
environment are deliberately not bundled. Never build from a user's virtual
environment or copy the whole checkout into a release.

## Build locally

Use Windows 10/11 x64, Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and Python 3.12 x64 (uv can install it). Allow several GB for the separate build
environment and output; the GPU runtime alone is over 1 GB uncompressed.
From the repository root in PowerShell:

```powershell
./packaging/windows/build.ps1
```

The script creates `.venv-win-build`, installs the hash-locked requirements,
runs regression tests and Qt catalog checks, freezes the application, runs the
offline packaged acceptance checks, and creates a ZIP plus SHA-256 checksum in
`release-artifacts/beta-*/`. It does not modify `.venv` or a FLUX environment.
Build directories are unique; previous builds are not erased.

To also verify actual CUDA node execution on an NVIDIA build machine:

```powershell
./packaging/windows/build.ps1 -RequireCuda
```

`-Python` accepts a Python executable or uv interpreter selector. `-Environment`
can name a dedicated build environment, never the working app/FLUX environment.
`-OutputDirectory` changes the artifact destination. `-SkipTests` is for local
iteration only; the packaged acceptance test is always required. Release/CI
builds must not skip tests. The build checks Windows, Python 3.12 and x64.

When intentionally updating dependencies, regenerate and review the lock:

```powershell
uv pip compile packaging/windows/requirements-build.in --python 3.12 --output-file packaging/windows/requirements.lock --generate-hashes
```

The Windows file version derives from `app/version.py`; `0.1.0b1` is beta 1.
Keep requirements changes separate from optional FLUX requirements. PyInstaller's
one-folder layout permits individual shared libraries to remain accessible and
avoids unpacking a multi-GB runtime every launch. See the official
[PyInstaller packaging documentation](https://pyinstaller.org/en/stable/operating-mode.html).

## What is verified

The frozen executable supports an offline diagnostic mode:

```powershell
$process = Start-Process -FilePath './LocalComicTranslate.exe' -ArgumentList '--smoke-test', 'report.json' -Wait -PassThru -WindowStyle Hidden
$process.ExitCode
```

Add `--require-cuda` to require a real CUDA inference session, with CPU fallback
disabled and the executed node's provider verified. CI runners normally lack
NVIDIA GPUs, so CI verifies CPU inference only.

The diagnostic mode uses disposable settings and data directories, never runs
profile migration, blocks network/model downloads/provider requests/keyring
access, and creates a hidden real main window. It checks bundled icons, SVGs,
themes, all nine translation catalogs, native modules, image codecs,
PNG/CBZ/CB7/PDF/PSD roundtrips, and the shipped optional worker source files.
Reports contain check names and statuses, not settings or credentials; diagnostic
logs can include filesystem paths, so review them before sharing.

This is not a replacement for a clean-machine test. Before inviting beta testers:

- Test extraction and launch on a Windows machine without Python, Qt or CUDA
  Toolkit installed; confirm both CPU-only and NVIDIA-driver configurations.
- Exercise first-use model/font downloads, real OCR/cleanup, project save/reopen,
  export, manual edits/undo, and connecting separately installed llama.cpp.
- Test the optional external FLUX Python and existing cached 4B/9B models/LoRAs.
  Installing its git-pinned Diffusers dependency also needs Git.
- Test cloud requests only with explicit tester consent and their own keys.
- Compressed CBR/RAR needs an external extractor; that is not bundled or verified.
- Back up projects and note that this fork's source and packaged builds share
  its profile. The ZIP is not a self-contained portable profile.
- Check `BUILD-INFO.json` provenance, license notices and model redistribution
  terms. No model weights, projects, API keys, signed executables, installer,
  file associations or automatic beta updates are provided.

## GitHub workflow and publication

See [the release review](RELEASE_REVIEW.md) for current release gates and
[source/replacement instructions](SOURCE_REPLACEMENT.md) for preparing the
checksum-pinned third-party source evidence. Source-derived notices are
included automatically from `packaging/windows/licenses/`; source archives
remain separate release assets and are not embedded in the application ZIP.

The manual **Build Windows x64 beta** Actions workflow uses a clean Windows x64
runner and uploads the tested ZIP, checksum and diagnostics as a workflow artifact.
It has read-only repository permissions and does **not** create a tag or release.
No workflow needs API keys, personal settings or signing credentials.

After testing and reviewing third-party redistribution requirements, build from
a committed revision, create a version tag, and manually publish a GitHub
**pre-release** with the ZIP, SHA-256 file and clear beta notes. Do not mark it as
the latest stable release. A GitHub release asset must be under 2 GiB; the build
rejects oversized ZIPs. See [GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

Code signing and an installer are separate follow-up work. Unsigned builds may
trigger SmartScreen; testers should verify the source and checksum, not disable
system security. macOS builds are outside this Windows beta workflow.
