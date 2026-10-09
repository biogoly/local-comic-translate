"""Runtime locations shared by source checkouts and packaged Artistic Edit.

The optional FLUX Python cannot import code from PyInstaller's PYZ archive.
Packages therefore carry a small, plain-Python worker tree, without weights or
Torch/Diffusers, separately from the GUI's frozen modules.
"""

from __future__ import annotations

from pathlib import Path
import sys

from modules.utils.paths import get_user_data_dir


WORKER_SOURCE_FILES = (
    "modules/__init__.py",
    "modules/artistic_edit/__init__.py",
    "modules/artistic_edit/contracts.py",
    "modules/artistic_edit/lora_registry.py",
    "modules/artistic_edit/flux_worker.py",
)


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def worker_source_root() -> Path:
    """Return a verified real source tree usable by the separate Python."""
    if is_frozen_app():
        bundle_root = getattr(sys, "_MEIPASS", None)
        if not bundle_root:
            raise RuntimeError("The packaged FLUX worker bundle could not be located.")
        root = Path(bundle_root) / "flux-worker"
    else:
        root = Path(__file__).resolve().parents[2]
    missing = [relative for relative in WORKER_SOURCE_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "The FLUX worker source files are missing. Reinstall the application: "
            + ", ".join(missing)
        )
    return root.resolve(strict=False)


def lora_root() -> Path:
    """Keep source LoRAs in the repo and packaged LoRAs in the writable profile.

    This only chooses a location. Opening the LoRA folder creates it, while
    ordinary startup remains read-only and never copies large model files.
    """
    if is_frozen_app():
        root = Path(get_user_data_dir())
    else:
        root = Path(__file__).resolve().parents[2]
    return root / "models" / "loras" / "flux2-klein"


def is_application_executable(candidate: str | Path) -> bool:
    """The packaged GUI executable is not an optional Python interpreter."""
    return is_frozen_app() and (
        Path(candidate).expanduser().resolve(strict=False)
        == Path(sys.executable).resolve(strict=False)
    )
