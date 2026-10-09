import ctypes
from ctypes import wintypes
import os
from pathlib import PureWindowsPath
import sys
import unittest
from unittest.mock import MagicMock, patch

from modules.utils import external_process
from modules.translation.llm.llama_server import LlamaServerConfig, LlamaServerRuntime


BUNDLE = r"C:\Apps\Comic Translate\_internal"


class ExternalEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(sys, "frozen", True, create=True))
        self.enterContext(patch.object(sys, "_MEIPASS", BUNDLE, create=True))
        self.enterContext(patch.object(sys, "platform", "win32"))

    def test_removes_only_bundle_paths_preserving_order_and_case(self):
        original = {
            "Path": (
                r'C:\Windows\System32;"C:\Apps\Comic Translate\_internal\PySide6";'
                r'c:/apps/comic translate/_internal/torch/lib;'
                r'C:\Apps\Comic Translate\_internal-other;D:\CUDA\bin;relative\bin'
            ),
            "CUSTOM": "unchanged",
        }
        before = dict(original)

        result = external_process.sanitized_external_environment(original)

        self.assertEqual(result["Path"], (
            r'C:\Windows\System32;C:\Apps\Comic Translate\_internal-other;'
            r'D:\CUDA\bin;relative\bin'
        ))
        self.assertEqual(result["CUSTOM"], "unchanged")
        self.assertEqual(original, before)

    def test_removes_bundle_qt_and_python_injections(self):
        original = {
            "QT_PLUGIN_PATH": BUNDLE + r"\PySide6\plugins",
            "QT_QPA_PLATFORM_PLUGIN_PATH": BUNDLE + r"\PySide6\plugins\platforms",
            "QML2_IMPORT_PATH": BUNDLE + r"\PySide6\qml",
            "QTWEBENGINEPROCESS_PATH": BUNDLE + r"\QtWebEngineProcess.exe",
            "PYTHONHOME": BUNDLE,
            "PYTHONPATH": BUNDLE + r"\base_library.zip;D:\ExternalPython\packages",
            "QT_QPA_PLATFORM": "offscreen",
        }

        result = external_process.sanitized_external_environment(original)

        self.assertEqual(result, {
            "PYTHONPATH": r"D:\ExternalPython\packages",
            "QT_QPA_PLATFORM": "offscreen",
        })

    def test_preserves_user_python_and_qt_settings(self):
        original = {
            "PYTHONHOME": r"D:\ExternalPython",
            "PYTHONPATH": r"D:\ExternalPython\packages",
            "QT_PLUGIN_PATH": r"D:\OtherApp\plugins",
            "QT_API": "pyside6",
            "PATH": r"D:\CUDA\bin;;C:\Windows\System32",
        }
        self.assertEqual(external_process.sanitized_external_environment(original), original)

    def test_preserves_path_with_parent_components_outside_bundle(self):
        original = {"PATH": BUNDLE + r"\..\external\bin;" + BUNDLE + r"\torch\..\bin"}
        self.assertEqual(
            external_process.sanitized_external_environment(original),
            {"PATH": BUNDLE + r"\..\external\bin"},
        )

    def test_non_frozen_environment_is_an_unchanged_copy(self):
        original = {"PATH": BUNDLE, "PYTHONHOME": BUNDLE}
        with patch.object(sys, "frozen", False):
            result = external_process.sanitized_external_environment(original)
        self.assertEqual(result, original)
        self.assertIsNot(result, original)

    def test_non_windows_environment_is_unchanged(self):
        original = {"PATH": BUNDLE, "PYTHONHOME": BUNDLE}
        with patch.object(sys, "platform", "linux"):
            self.assertEqual(external_process.sanitized_external_environment(original), original)

    def test_missing_bundle_directory_is_safe(self):
        original = {"PATH": BUNDLE}
        with patch.object(sys, "_MEIPASS", None):
            self.assertEqual(external_process.sanitized_external_environment(original), original)


class _FakeKernel32:
    def __init__(self, directory):
        self.directory = directory
        self.get_calls = []
        self.set_calls = []

    def GetDllDirectoryW(self, capacity, buffer):
        self.get_calls.append(capacity)
        if not self.directory:
            return 0
        if len(self.directory) >= capacity:
            return len(self.directory) + 1
        buffer.value = self.directory
        return len(self.directory)

    def SetDllDirectoryW(self, directory):
        self.set_calls.append(directory)
        self.directory = directory
        return True


class ExternalProcessContextTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(sys, "frozen", True, create=True))
        self.enterContext(patch.object(sys, "platform", "win32"))
        self.enterContext(patch.object(sys, "_MEIPASS", BUNDLE, create=True))
        # Portable mocks: ctypes last-error helpers exist only on Windows.
        self.enterContext(patch.object(ctypes, "set_last_error", create=True))
        self.enterContext(patch.object(ctypes, "get_last_error", return_value=0, create=True))
        self.kernel32 = _FakeKernel32(r"D:\Actual Previous DLL Directory\日本語")
        self.api = self.enterContext(patch.object(
            external_process, "_windows_dll_api", return_value=self.kernel32,
        ))

    def test_clears_during_spawn_and_restores_actual_unicode_directory(self):
        previous = self.kernel32.directory
        before_environment = dict(os.environ)

        with external_process.external_process_environment():
            self.assertIsNone(self.kernel32.directory)
            self.assertEqual(dict(os.environ), before_environment)

        self.assertEqual(self.kernel32.directory, previous)
        self.assertEqual(self.kernel32.set_calls, [None, previous])
        self.assertEqual(dict(os.environ), before_environment)

    def test_restores_directory_after_spawn_exception(self):
        previous = self.kernel32.directory
        with self.assertRaisesRegex(OSError, "spawn failed"):
            with external_process.external_process_environment():
                raise OSError("spawn failed")
        self.assertEqual(self.kernel32.directory, previous)

    def test_nested_context_restores_each_prior_state(self):
        previous = self.kernel32.directory
        with external_process.external_process_environment():
            with external_process.external_process_environment():
                self.assertIsNone(self.kernel32.directory)
            self.assertIsNone(self.kernel32.directory)
        self.assertEqual(self.kernel32.set_calls, [None, None, None, previous])

    def test_grows_buffer_for_long_unicode_directory(self):
        previous = "C:\\" + "資料\\" * 110
        self.kernel32.directory = previous
        with external_process.external_process_environment():
            self.assertIsNone(self.kernel32.directory)
        self.assertEqual(self.kernel32.directory, previous)
        self.assertGreater(self.kernel32.get_calls[1], 256)

    def test_no_previous_directory_remains_default(self):
        self.kernel32.directory = None
        with external_process.external_process_environment():
            self.assertIsNone(self.kernel32.directory)
        self.assertEqual(self.kernel32.set_calls, [None, None])

    def test_does_not_touch_win32_api_in_source_mode(self):
        with patch.object(sys, "frozen", False):
            with external_process.external_process_environment():
                pass
        self.api.assert_not_called()

    def test_does_not_touch_win32_api_on_other_platforms(self):
        with patch.object(sys, "platform", "darwin"):
            with external_process.external_process_environment():
                pass
        self.api.assert_not_called()

    def test_read_error_does_not_change_directory_or_launch(self):
        self.kernel32.GetDllDirectoryW = MagicMock(return_value=0)
        with (
            patch.object(ctypes, "get_last_error", return_value=5, create=True),
            patch.object(ctypes, "WinError", return_value=OSError("read failed"), create=True),
        ):
            with self.assertRaisesRegex(OSError, "read failed"):
                with external_process.external_process_environment():
                    self.fail("Must not spawn with unknown DLL directory")
        self.assertEqual(self.kernel32.set_calls, [])


class WindowsApiDeclarationTests(unittest.TestCase):
    def test_uses_explicit_unicode_prototypes_and_last_error(self):
        kernel32 = MagicMock()
        with patch.object(ctypes, "WinDLL", return_value=kernel32, create=True) as load:
            self.assertIs(external_process._windows_dll_api(), kernel32)
        load.assert_called_once_with("kernel32", use_last_error=True)
        self.assertEqual(kernel32.GetDllDirectoryW.argtypes, [wintypes.DWORD, wintypes.LPWSTR])
        self.assertIs(kernel32.GetDllDirectoryW.restype, wintypes.DWORD)
        self.assertEqual(kernel32.SetDllDirectoryW.argtypes, [wintypes.LPCWSTR])
        self.assertIs(kernel32.SetDllDirectoryW.restype, wintypes.BOOL)


class LlamaExternalProcessTests(unittest.TestCase):
    def _start_server(self, frozen):
        runtime = LlamaServerRuntime()
        config = LlamaServerConfig(executable_path="llama-server", model_path="model.gguf")
        child = MagicMock()
        child.poll.return_value = None
        context = MagicMock()
        context.__enter__ = MagicMock()
        context.__exit__ = MagicMock(return_value=False)
        with (
            patch.object(sys, "frozen", frozen, create=True),
            patch("modules.translation.llm.llama_server.os.name", "nt"),
            patch("modules.translation.llm.llama_server.Path", PureWindowsPath),
            patch("modules.translation.llm.llama_server.subprocess.CREATE_NO_WINDOW", 0x08000000, create=True),
            patch.object(runtime, "_validate_config", return_value=config),
            patch.object(runtime, "_start_output_reader"),
            patch.object(runtime, "_wait_until_ready"),
            patch("modules.translation.llm.llama_server.external_process_environment", return_value=context),
            patch("modules.translation.llm.llama_server.sanitized_external_environment", return_value={"PATH": "external"}) as sanitize,
            patch("modules.translation.llm.llama_server.subprocess.Popen", return_value=child) as popen,
        ):
            runtime.ensure_running(config)
            context.__enter__.assert_called_once()
            context.__exit__.assert_called_once()
            runtime.stop()
        return sanitize, popen

    def test_frozen_llama_server_gets_sanitized_environment(self):
        sanitize, popen = self._start_server(True)
        sanitize.assert_called_once_with(os.environ)
        self.assertEqual(popen.call_args.kwargs["env"], {"PATH": "external"})

    def test_source_llama_server_keeps_existing_environment_behavior(self):
        sanitize, popen = self._start_server(False)
        sanitize.assert_not_called()
        self.assertNotIn("env", popen.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
