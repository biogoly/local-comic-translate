import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, replace
from unittest.mock import patch
from uuid import uuid4

import requests

from modules.artistic_edit.comfy_runtime import (
    ComfyRuntime, ComfyRuntimeBusyError, ComfyRuntimeCancelled,
    ComfyRuntimeError, resolve_gpu_device, resolve_installation,
)


@dataclass(frozen=True)
class Config:
    mode: str = "managed"
    base_url: str = "http://127.0.0.1:8188"
    installation_root: str = ""
    python_executable: str | None = None
    gpu_index: int = 1
    memory_policy: str = "normal"


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.stdout = None
        self.terminated = False
        self.killed = False
        self.wait_timeouts = []
        self.fail_first_wait = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.fail_first_wait and len(self.wait_timeouts) == 1:
            raise subprocess.TimeoutExpired("owned-comfy", timeout)
        self.returncode = 0
        return 0


class Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status_code = status
        self.closed = False

    def iter_content(self, chunk_size):
        encoded = self.data if isinstance(self.data, bytes) else json.dumps(self.data).encode()
        for offset in range(0, len(encoded), chunk_size):
            yield encoded[offset:offset + chunk_size]

    def close(self):
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.portable = self.root / "portable with spaces é"
        self.comfy = self.portable / "ComfyUI"
        self.python = self.portable / "python_embeded/python.exe"
        self.frontend = self.python.parent / "Lib/site-packages/comfyui_frontend_package/static"
        for folder in (self.comfy / "comfy", self.comfy / "models", self.frontend):
            folder.mkdir(parents=True)
        self.python.touch()
        (self.frontend / "index.html").write_text("frontend", encoding="utf-8")
        (self.comfy / "main.py").write_text("# never executed", encoding="utf-8")
        (self.comfy / "comfyui_version.py").write_text('__version__ = "0.37.0"', encoding="utf-8")
        (self.comfy / "comfy/cli_args.py").write_text(
            "--base-directory --models-directory --input-directory --output-directory "
            "--temp-directory --user-directory --database-url --front-end-root "
            "--disable-all-custom-nodes --disable-api-nodes --disable-auto-launch "
            "--cuda-device --cache-none", encoding="utf-8",
        )
        self.config = Config(installation_root=str(self.portable))
        self.process = FakeProcess()
        self.commands = []
        self.gets = []
        self.sessions = []
        self.queue = {"queue_running": [], "queue_pending": []}
        self.foreign_identity = False
        self.http_error = None
        self.connection_failures = 0
        self.elapsed = 0

        def spawn(command, **kwargs):
            self.commands.append((command, kwargs))
            return self.process

        owner = self

        class Session:
            trust_env = True

            def __init__(self):
                self.closed = False
                owner.sessions.append(self)

            def get(self, url, **kwargs):
                owner.gets.append((url, kwargs, self.trust_env))
                if owner.http_error:
                    raise owner.http_error
                if owner.connection_failures:
                    owner.connection_failures -= 1
                    raise requests.ConnectionError("not listening yet")
                if url.endswith("/system_stats"):
                    argv = [] if owner.foreign_identity else owner.commands[-1][0][4:]
                    return Response({"system": {"argv": argv, "comfyui_version": "0.37.0"}})
                if url.endswith("/queue"):
                    return Response(owner.queue)
                raise AssertionError("Unexpected request: " + url)

            def close(self):
                self.closed = True

        def wait(seconds):
            self.elapsed += seconds

        self.runtime = ComfyRuntime(
            self.root / "state", process_factory=spawn, http_factory=Session,
            gpu_resolver=lambda index: "GPU-aaaa-bbbb" if index == 1 else "GPU-cccc-dddd",
            port_factory=lambda: 48177, clock=lambda: self.elapsed, wait=wait,
            startup_timeout=1,
        )

    def start(self):
        return self.runtime.ensure_started(self.config, threading.Event())

    def test_connect_mode_never_spawns_reads_installation_or_owns_server(self):
        config = Config(mode="connect")
        self.assertEqual(self.runtime.ensure_started(config), config.base_url)
        self.runtime.disconnect()
        self.runtime.shutdown()
        self.assertEqual(self.commands, [])
        self.assertEqual(self.gets, [])
        self.assertFalse((self.root / "state").exists())

    def test_isolated_launch_preserves_installation_and_binds_uuid_gpu(self):
        before = {str(p.relative_to(self.portable)): p.read_bytes() for p in self.portable.rglob("*") if p.is_file()}
        with patch.dict("os.environ", {"PYTHONPATH": "bad", "PYTHONHOME": "bad", "CUDA_VISIBLE_DEVICES": "0"}):
            self.assertEqual(self.start(), "http://127.0.0.1:48177")
        command, kwargs = self.commands[0]
        self.assertEqual(command[:4], [str(self.python), "-B", "-s", "-u"])
        self.assertEqual(kwargs["cwd"], str(self.comfy))
        flags = dict(zip(command[5:], command[6:]))
        run_root = Path(flags["--base-directory"])
        self.assertTrue(run_root.is_relative_to(self.root / "state"))
        self.assertEqual(flags["--models-directory"], str(self.comfy / "models"))
        self.assertEqual(flags["--database-url"], "sqlite:///" + (run_root / "user/comfyui.db").as_posix())
        self.assertEqual(flags["--front-end-root"], str(self.frontend))
        self.assertEqual(flags["--cuda-device"], "GPU-aaaa-bbbb")
        self.assertIn("--disable-all-custom-nodes", command)
        self.assertIn("--disable-api-nodes", command)
        self.assertIn("--disable-auto-launch", command)
        self.assertIn("--windows-standalone-build", command)
        self.assertNotIn("--enable-manager", command)
        environment = kwargs["env"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-aaaa-bbbb")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertTrue(Path(environment["TEMP"]).is_relative_to(run_root))
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(all(not trust for _, _, trust in self.gets))
        self.assertTrue(all(options["allow_redirects"] is False for _, options, _ in self.gets))
        self.assertTrue(all(session.closed for session in self.sessions))
        after = {str(p.relative_to(self.portable)): p.read_bytes() for p in self.portable.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_low_cache_is_explicit_and_reuse_never_starts_second_process(self):
        self.config = replace(self.config, memory_policy="low_cache")
        first = self.start()
        self.assertIn("--cache-none", self.commands[0][0])
        self.assertEqual(self.start(), first)
        self.assertEqual(len(self.commands), 1)
        with self.assertRaises(ComfyRuntimeBusyError):
            self.runtime.ensure_started(replace(self.config, gpu_index=0))
        self.assertFalse(self.process.terminated)

    def test_startup_tolerates_connection_refusal_but_not_foreign_identity(self):
        self.connection_failures = 2
        self.start()
        self.assertAlmostEqual(self.elapsed, 0.4)
        self.foreign_identity = True
        self.assertFalse(self.runtime.can_shutdown())
        with self.assertRaises(ComfyRuntimeBusyError):
            self.runtime.stop()
        self.assertFalse(self.process.terminated)

    def test_port_collision_stops_only_child_and_never_returns_foreign_url(self):
        self.foreign_identity = True
        with self.assertRaisesRegex(ComfyRuntimeError, "refusing to attach"):
            self.start()
        self.assertTrue(self.process.terminated)
        self.assertIsNone(self.runtime.base_url)
        self.assertFalse(self.runtime.is_running())
        self.assertTrue(all(url.endswith("/system_stats") for url, _, _ in self.gets))

    def test_cancel_before_start_and_during_readiness(self):
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(ComfyRuntimeCancelled):
            self.runtime.ensure_started(self.config, cancelled)
        self.assertFalse(self.commands)
        cancelled.clear()
        self.connection_failures = 2

        def wait(_seconds):
            cancelled.set()

        self.runtime._wait = wait
        with self.assertRaises(ComfyRuntimeCancelled):
            self.runtime.ensure_started(self.config, cancelled)
        self.assertTrue(self.process.terminated)

    def test_timeout_and_early_exit_are_bounded_and_reset_ownership(self):
        self.connection_failures = 100
        with self.assertRaisesRegex(ComfyRuntimeError, "did not become ready"):
            self.start()
        self.assertTrue(self.process.terminated)
        self.assertLess(self.elapsed, 1.5)
        self.process = FakeProcess()
        self.process.returncode = 7
        with self.assertRaisesRegex(ComfyRuntimeError, "code 7"):
            self.start()
        self.assertFalse(self.runtime.is_running())

    def test_idle_stop_checks_identity_and_queue_and_has_bounded_kill_fallback(self):
        self.start()
        run_root = self.runtime._run_root
        marker = run_root / "output/result.png"
        marker.write_bytes(b"retained evidence")
        self.process.fail_first_wait = True
        self.runtime.shutdown()
        self.assertTrue(self.process.terminated)
        self.assertTrue(self.process.killed)
        self.assertEqual(self.process.wait_timeouts, [5, 2])
        self.assertFalse(self.runtime.is_running())
        self.assertEqual(marker.read_bytes(), b"retained evidence")
        self.assertFalse(any("/free" in url or "/interrupt" in url for url, _, _ in self.gets))

    def test_foreign_and_owned_jobs_both_block_stop_until_idle(self):
        self.start()
        ours, foreign = str(uuid4()), str(uuid4())
        self.runtime.register_prompt(ours)
        self.queue["queue_running"] = [[1, foreign]]
        with self.assertRaisesRegex(ComfyRuntimeBusyError, "Another client"):
            self.runtime.stop()
        self.queue["queue_running"] = [[1, ours]]
        self.assertFalse(self.runtime.can_shutdown())
        self.queue["queue_running"] = []
        self.assertFalse(self.runtime.can_shutdown())
        self.runtime.finish_prompt(ours)
        self.assertTrue(self.runtime.can_shutdown())
        self.runtime.stop()
        self.assertTrue(self.process.terminated)

    def test_unreachable_or_malformed_queue_never_justifies_killing_live_server(self):
        self.start()
        self.http_error = requests.Timeout("timeout")
        self.assertFalse(self.runtime.can_shutdown())
        self.http_error = None
        self.queue = {"queue_running": [], "queue_pending": "bad"}
        with self.assertRaises(ComfyRuntimeBusyError):
            self.runtime.stop()
        self.assertFalse(self.process.terminated)
        self.process.returncode = 1
        self.assertTrue(self.runtime.can_shutdown())

    def test_shutdown_does_not_wait_on_an_in_progress_start(self):
        self.runtime._operation_lock.acquire()
        try:
            with self.assertRaises(ComfyRuntimeBusyError):
                self.runtime.stop()
        finally:
            self.runtime._operation_lock.release()

    def test_external_dll_environment_restores_when_spawn_fails(self):
        events = []

        @contextmanager
        def context():
            events.append("restore-external")
            try:
                yield
            finally:
                events.append("restore-app")

        def fail(*args, **kwargs):
            events.append("spawn")
            raise OSError("broken python")

        self.runtime._process_factory = fail
        with patch("modules.artistic_edit.comfy_runtime.external_process_environment", context):
            with self.assertRaises(OSError):
                self.start()
        self.assertEqual(events, ["restore-external", "spawn", "restore-app"])
        self.assertTrue(self.runtime.can_shutdown())

    def test_resolver_rejects_unsupported_missing_and_ambiguous_installations(self):
        resolved = resolve_installation(self.config)
        self.assertEqual(resolved.python, self.python)
        self.assertEqual(resolve_installation(replace(self.config, installation_root=str(self.comfy))), resolved)
        (self.comfy / "comfyui_version.py").write_text('__version__ = "0.36.9"')
        with self.assertRaisesRegex(ComfyRuntimeError, "0.37.0"):
            resolve_installation(self.config)
        (self.comfy / "comfyui_version.py").write_text('__version__ = "0.37.0"')
        (self.portable / "main.py").touch()
        with self.assertRaisesRegex(ComfyRuntimeError, "unambiguous"):
            resolve_installation(self.config)
        (self.portable / "main.py").unlink()
        self.runtime.state_root = self.comfy / "user/new-state"
        with self.assertRaisesRegex(ComfyRuntimeError, "outside"):
            self.start()
        self.assertFalse(self.commands)

    def test_resolver_rejects_gui_executable_and_missing_frontend(self):
        with patch("modules.artistic_edit.comfy_runtime.is_application_executable", return_value=True):
            with self.assertRaisesRegex(ComfyRuntimeError, "editor is not Python"):
                resolve_installation(self.config)
        (self.frontend / "index.html").unlink()
        with self.assertRaisesRegex(ComfyRuntimeError, "Nothing was installed"):
            resolve_installation(self.config)

    def test_status_response_size_and_redirects_are_rejected(self):
        class Session:
            trust_env = True

            def get(self, *args, **kwargs):
                return Response(b"x" * (1024 * 1024 + 1))

            def close(self):
                pass

        self.runtime._http_factory = Session
        with self.assertRaisesRegex(ComfyRuntimeError, "size limit"):
            self.start()
        self.assertTrue(self.process.terminated)

    def test_gpu_identity_query_is_read_only_bounded_and_rejects_missing_index(self):
        result = subprocess.CompletedProcess([], 0, "0, GPU-aaaa-bbbb\n1, GPU-cccc-dddd\n", "")
        with (
            patch("modules.artistic_edit.comfy_runtime.shutil.which", return_value="nvidia-smi"),
            patch("modules.artistic_edit.comfy_runtime.subprocess.run", return_value=result) as run,
        ):
            self.assertEqual(resolve_gpu_device(1), "GPU-cccc-dddd")
            with self.assertRaisesRegex(ComfyRuntimeError, "not found"):
                resolve_gpu_device(2)
        self.assertEqual(run.call_args.args[0][1:], ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)


if __name__ == "__main__":
    unittest.main()
