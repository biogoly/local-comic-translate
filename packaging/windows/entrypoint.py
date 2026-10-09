"""Frozen desktop entry point; no compiler/runtime is needed by end users."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import sys


def main() -> int:
    multiprocessing.freeze_support()
    smoke = "--smoke-test" in sys.argv
    if smoke:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--smoke-test", required=True, metavar="REPORT.json")
        parser.add_argument("--require-cuda", action="store_true")
        args = parser.parse_args()
        log_dir = Path(args.smoke_test).resolve().parent
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
        from modules.utils.paths import get_user_data_dir

        log_dir = Path(get_user_data_dir()) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("package-smoke.log" if smoke else "desktop.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")],
        force=True,
    )
    # A windowed PyInstaller executable has no console streams. Native wrappers
    # and dependencies may still print diagnostics, so provide valid handles.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(log_path, "a", encoding="utf-8", buffering=1)

    def exception_hook(exc_type, exc, traceback):
        logging.critical("Unhandled desktop exception", exc_info=(exc_type, exc, traceback))

    sys.excepthook = exception_hook
    try:
        if smoke:
            from app.package_smoke import run_smoke_test

            return run_smoke_test(args.smoke_test, require_cuda=args.require_cuda)
        from comic import main as run_desktop

        run_desktop()
        return 0
    except Exception:
        logging.exception("Desktop startup failed; see %s", log_path)
        if not smoke:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None, f"Local Comic Translate could not start.\n\nDiagnostic log:\n{log_path}",
                "Local Comic Translate", 0x10,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
