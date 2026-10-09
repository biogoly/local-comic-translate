# Run through build.py, which prepares an allowlisted payload and version file.
import importlib.metadata
import os
from pathlib import Path
import sys

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata,
    collect_delvewheel_libs_directory,
)

ROOT = Path(SPECPATH).resolve().parents[1]
PAYLOAD = Path(os.environ["LCT_BUILD_PAYLOAD"])
datas = [(str(PAYLOAD), ".")]
for relative in ("resources/icons", "resources/static", "resources/translations/compiled"):
    datas.append((str(ROOT / relative), relative))

# The external FLUX Python cannot import from the executable's PYZ archive.
# Only these worker sources are exposed; never copy the repository or models.
for relative in (
    "modules/__init__.py", "modules/artistic_edit/__init__.py",
    "modules/artistic_edit/contracts.py", "modules/artistic_edit/lora_registry.py",
    "modules/artistic_edit/flux_worker.py",
):
    datas.append((str(ROOT / relative), "flux-worker/" + str(Path(relative).parent)))

binaries = collect_dynamic_libs("onnxruntime")
# Python's _ssl extension must use its own OpenSSL pair, not a machine-local
# OpenSSL installation (some tools even place copies in Windows/System32).
for filename in ("libssl-3-x64.dll", "libcrypto-3-x64.dll"):
    library = Path(sys.base_prefix) / "DLLs" / filename
    if not library.is_file():
        raise RuntimeError(f"Required Python OpenSSL runtime not found: {filename}")
    binaries.append((str(library), "."))
# Prefer the MSVC runtime shipped with the matching Shiboken wheel over a
# developer tool's older copy discovered on PATH. Qt must share these versions.
shiboken_dist = importlib.metadata.distribution("shiboken6")
for file in shiboken_dist.files or ():
    if file.name.lower().startswith(("msvcp140", "vcruntime140")) and file.suffix.lower() == ".dll":
        binaries.append((str(shiboken_dist.locate_file(file)), "."))
# Preserve NVIDIA wheel paths used by ORT's preload_dlls(directory="").
for dist in importlib.metadata.distributions():
    if dist.metadata["Name"].lower().startswith("nvidia-"):
        for file in dist.files or ():
            if str(file).lower().endswith(".dll"):
                binaries.append((str(dist.locate_file(file)), str(file.parent)))

datas += copy_metadata("onnxruntime-gpu")
datas += copy_metadata("keyring")
datas += collect_data_files("photoshopapi")
datas, binaries = collect_delvewheel_libs_directory("photoshopapi", datas=datas, binaries=binaries)

a = Analysis(
    [str(ROOT / "packaging/windows/entrypoint.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["comic", "controller", "photoshopapi", "photoshopapi.lib.photoshopapi",
                   "keyring.backends.Windows", "PIL.ImageQt",
                   "numpy.core.multiarray", "numpy.core._multiarray_umath"] + collect_submodules("win32ctypes"),
    excludes=["torch", "torchvision", "torchaudio", "transformers", "diffusers",
              "accelerate", "peft", "bitsandbytes", "pytest", "tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="LocalComicTranslate", debug=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
    icon=str(ROOT / "resources/icons/icon.ico"),
    version=str(PAYLOAD / "windows-version.txt"),
    contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="LocalComicTranslate")
