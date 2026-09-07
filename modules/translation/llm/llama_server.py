"""Lifecycle management for an optional llama.cpp ``llama-server`` child process."""

from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


logger = logging.getLogger(__name__)
MANAGED_MODEL_ALIAS = "local-comic-translate"


@dataclass(frozen=True)
class LlamaServerConfig:
    executable_path: str = ""
    model_path: str = ""
    mmproj_path: str = ""
    context_size: int = 8192
    gpu_layers: int = 999
    startup_timeout: int = 90


def resolve_llama_server(explicit_path: str = "") -> Path:
    """Resolve llama-server from a configured path, PATH, or app-local bundle."""
    if explicit_path.strip():
        explicit = Path(os.path.expandvars(os.path.expanduser(explicit_path.strip())))
        if explicit.is_file():
            return explicit.resolve()
        raise FileNotFoundError(f"llama-server was not found at: {explicit}")

    executable_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    on_path = shutil.which("llama-server") or shutil.which(executable_name)
    if on_path:
        return Path(on_path).resolve()

    roots = [Path(__file__).resolve().parents[3]]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent)
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        roots.insert(0, Path(bundle_root))

    relative_candidates = (
        Path(executable_name),
        Path("bin") / executable_name,
        Path("llama.cpp") / executable_name,
    )
    for root in roots:
        for relative in relative_candidates:
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError(
        "llama-server was not found. Select the executable in Settings > LLMs or add it to PATH."
    )


class LlamaServerRuntime:
    """Own one app-scoped llama-server process and restart it when settings change."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._config: LlamaServerConfig | None = None
        self._base_url: str | None = None
        self._output_tail: deque[str] = deque(maxlen=40)
        self._output_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()

    def ensure_running(self, config: LlamaServerConfig) -> str:
        """Start the configured server if needed and return its ``/v1`` URL."""
        validated = self._validate_config(config)
        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is None
                and self._config == validated
                and self._base_url
            ):
                return self._base_url

            self._stop_locked()
            self._stop_requested.clear()
            port = self._find_available_port()
            executable = Path(validated.executable_path)
            command = self.build_command(validated, port)
            popen_kwargs = {
                "cwd": str(executable.parent),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            logger.info("Starting managed llama-server on 127.0.0.1:%s", port)
            self._output_tail.clear()
            try:
                self._process = subprocess.Popen(command, **popen_kwargs)
            except OSError as exc:
                self._process = None
                raise RuntimeError(f"Could not start llama-server: {exc}") from exc

            self._config = validated
            self._base_url = f"http://127.0.0.1:{port}/v1"
            self._start_output_reader()

            try:
                self._wait_until_ready(port, validated.startup_timeout)
            except Exception:
                output = self._formatted_output_tail()
                self._stop_locked()
                suffix = f"\nRecent llama-server output:\n{output}" if output else ""
                raise RuntimeError(f"llama-server failed to become ready.{suffix}")

            return self._base_url

    @staticmethod
    def build_command(config: LlamaServerConfig, port: int) -> list[str]:
        command = [
            config.executable_path,
            "-m", config.model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--alias", MANAGED_MODEL_ALIAS,
            "--ctx-size", str(config.context_size),
            "--n-gpu-layers", str(config.gpu_layers),
            "--jinja",
        ]
        if config.mmproj_path:
            command.extend(["--mmproj", config.mmproj_path])
        return command

    def stop(self) -> None:
        # Set this before taking the lock so a concurrent startup health poll
        # exits promptly instead of making the UI wait for startup_timeout.
        self._stop_requested.set()
        with self._lock:
            self._stop_locked()

    def _validate_config(self, config: LlamaServerConfig) -> LlamaServerConfig:
        executable = resolve_llama_server(config.executable_path)
        model = Path(os.path.expandvars(os.path.expanduser(config.model_path.strip())))
        if not model.is_file():
            raise FileNotFoundError(f"GGUF model was not found at: {model}")

        mmproj = ""
        if config.mmproj_path.strip():
            mmproj_path = Path(os.path.expandvars(os.path.expanduser(config.mmproj_path.strip())))
            if not mmproj_path.is_file():
                raise FileNotFoundError(f"Multimodal projector was not found at: {mmproj_path}")
            mmproj = str(mmproj_path.resolve())

        return LlamaServerConfig(
            executable_path=str(executable),
            model_path=str(model.resolve()),
            mmproj_path=mmproj,
            context_size=max(1024, int(config.context_size)),
            gpu_layers=max(0, int(config.gpu_layers)),
            startup_timeout=max(5, int(config.startup_timeout)),
        )

    def _start_output_reader(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        def read_output() -> None:
            try:
                for line in process.stdout:
                    clean = line.rstrip()
                    if clean:
                        self._output_tail.append(clean)
                        logger.debug("llama-server: %s", clean)
            except (OSError, ValueError):
                pass

        self._output_thread = threading.Thread(
            target=read_output,
            name="llama-server-output",
            daemon=True,
        )
        self._output_thread.start()

    def _wait_until_ready(self, port: int, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        health_url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if self._stop_requested.is_set():
                raise RuntimeError("llama-server startup was cancelled.")
            process = self._process
            if process is None or process.poll() is not None:
                exit_code = process.returncode if process is not None else "unknown"
                raise RuntimeError(f"llama-server exited with code {exit_code}.")
            try:
                with urlopen(health_url, timeout=1) as response:
                    if 200 <= response.status < 300:
                        return
            except (HTTPError, URLError, TimeoutError, OSError):
                pass
            time.sleep(0.2)
        raise TimeoutError(f"llama-server did not become ready within {timeout} seconds.")

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        self._config = None
        self._base_url = None
        if process is None or process.poll() is not None:
            return

        logger.info("Stopping managed llama-server")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _formatted_output_tail(self) -> str:
        return "\n".join(self._output_tail)

    @staticmethod
    def _find_available_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


_RUNTIME = LlamaServerRuntime()


def get_llama_server_runtime() -> LlamaServerRuntime:
    return _RUNTIME


def shutdown_local_runtime() -> None:
    _RUNTIME.stop()


atexit.register(shutdown_local_runtime)
