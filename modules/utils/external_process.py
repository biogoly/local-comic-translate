"""Keep external runtimes independent from a frozen Windows application's DLLs.

PyInstaller's DLL search directory and package-hook PATH entries are useful for
the GUI, but must not leak into an independently installed llama.cpp or Python.
See https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html
#launching-external-programs-from-the-frozen-application
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import ntpath
import sys
import threading


_DLL_DIRECTORY_LOCK = threading.RLock()
_PATH_VARIABLES = frozenset({
    "PATH", "PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "QML_IMPORT_PATH",
    "QML2_IMPORT_PATH", "QTWEBENGINEPROCESS_PATH", "QTWEBENGINE_RESOURCES_PATH",
    "QTWEBENGINE_LOCALES_PATH",
})


def _is_frozen_windows() -> bool:
    return sys.platform == "win32" and bool(getattr(sys, "frozen", False))


def _is_bundled_path(value: str, bundle_root: str) -> bool:
    # Windows paths are case-insensitive; commonpath also distinguishes a real
    # child from an unrelated sibling such as "_internal-other". Do not resolve
    # against the working directory or remove users' relative search entries.
    candidate = ntpath.normcase(ntpath.normpath(value.strip().strip('"')))
    root = ntpath.normcase(ntpath.normpath(bundle_root))
    if not ntpath.isabs(candidate):
        return False
    try:
        return ntpath.commonpath([candidate, root]) == root
    except ValueError:
        return False  # Different drives, or an invalid path: preserve it.


def sanitized_external_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Copy an environment, removing only frozen Windows bundle search paths.

    User CUDA, Python, Qt and other external paths remain intact, including their
    original spelling and ordering. This function never mutates ``os.environ``
    or its argument. Non-frozen and non-Windows applications receive a copy with
    identical contents.
    """
    result = dict(environment)
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not _is_frozen_windows() or not bundle_root:
        return result

    for name, value in environment.items():
        if name.upper() not in _PATH_VARIABLES:
            continue
        entries = value.split(";")
        kept = [entry for entry in entries if not _is_bundled_path(entry, str(bundle_root))]
        if kept == entries:
            continue
        if kept:
            result[name] = ";".join(kept)
        else:
            result.pop(name, None)
    return result


def _windows_dll_api():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetDllDirectoryW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
    kernel32.GetDllDirectoryW.restype = wintypes.DWORD
    kernel32.SetDllDirectoryW.argtypes = [wintypes.LPCWSTR]
    kernel32.SetDllDirectoryW.restype = wintypes.BOOL
    return kernel32


def _read_dll_directory(kernel32) -> str | None:
    capacity = 256
    while True:
        buffer = ctypes.create_unicode_buffer(capacity)
        ctypes.set_last_error(0)
        length = kernel32.GetDllDirectoryW(capacity, buffer)
        if not length:
            error = ctypes.get_last_error()
            if error:
                raise ctypes.WinError(error)
            return None
        if length < capacity:
            return buffer.value
        capacity = length + 1


def _set_dll_directory(kernel32, directory: str | None) -> None:
    if not kernel32.SetDllDirectoryW(directory):
        raise ctypes.WinError(ctypes.get_last_error())


@contextmanager
def external_process_environment() -> Iterator[None]:
    """Temporarily restore Windows DLL lookup while spawning an external child.

    Pair this with ``sanitized_external_environment`` passed to the child. Keep
    the scope short: around Popen, or QProcess.start plus waitForStarted (Qt may
    create the child asynchronously). This is not for bundled Qt helper programs
    or multiprocessing workers that require the application's own libraries.

    Only the DLL search directory is temporarily changed, never os.environ. A
    shared reentrant lock serializes the application's external launch sites,
    and the actual previous directory is restored even if child startup fails.
    """
    if not _is_frozen_windows():
        yield
        return

    with _DLL_DIRECTORY_LOCK:
        kernel32 = _windows_dll_api()
        previous = _read_dll_directory(kernel32)
        _set_dll_directory(kernel32, None)
        try:
            yield
        finally:
            _set_dll_directory(kernel32, previous)
