"""Own an optional ComfyUI child without changing its installation or profile."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Callable, TYPE_CHECKING
from uuid import UUID, uuid4

import requests

from modules.utils.external_process import (
    external_process_environment,
    sanitized_external_environment,
)
from .runtime_paths import is_application_executable

if TYPE_CHECKING:
    from .comfy_config import ComfyConnectionConfig


class ComfyRuntimeError(RuntimeError):
    pass


class ComfyRuntimeBusyError(ComfyRuntimeError):
    """The owned process cannot safely be stopped yet."""


class ComfyRuntimeCancelled(ComfyRuntimeError):
    pass


@dataclass(frozen=True)
class ComfyInstallation:
    root: Path
    python: Path
    main: Path
    models: Path
    frontend: Path
    version: str
    portable: bool


_REQUIRED_FLAGS = (
    "--base-directory", "--models-directory", "--input-directory",
    "--output-directory", "--temp-directory", "--user-directory",
    "--database-url", "--front-end-root", "--disable-all-custom-nodes",
    "--disable-api-nodes", "--disable-auto-launch", "--cuda-device",
)


def _source_text(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > 512 * 1024:
        raise ComfyRuntimeError(f"Required ComfyUI source file is missing or invalid: {path}")
    return path.read_text(encoding="utf-8")


def resolve_installation(config: ComfyConnectionConfig) -> ComfyInstallation:
    root = Path(config.installation_root).expanduser().resolve()
    candidates = [path for path in (root, root / "ComfyUI") if (path / "main.py").is_file()]
    if len(candidates) != 1:
        raise ComfyRuntimeError("Select an unambiguous ComfyUI portable root or ComfyUI source folder.")
    comfy = candidates[0]
    portable_python = comfy.parent / "python_embeded" / "python.exe"
    configured = config.python_executable
    python = Path(configured).expanduser().resolve() if configured else portable_python
    if not python.is_file() or is_application_executable(python):
        raise ComfyRuntimeError("Select the separate ComfyUI Python executable; the editor is not Python.")
    portable = python == portable_python
    version_source = _source_text(comfy / "comfyui_version.py")
    match = re.search(r'__version__\s*=\s*[\'"](\d+)\.(\d+)\.(\d+)[\'"]', version_source)
    if not match or tuple(map(int, match.groups())) < (0, 37, 0):
        raise ComfyRuntimeError("Managed ComfyUI requires version 0.37.0 or newer with isolated directory support.")
    flags = _source_text(comfy / "comfy" / "cli_args.py")
    required = (*_REQUIRED_FLAGS, *(("--cache-none",) if config.memory_policy == "low_cache" else ()))
    if any(flag not in flags for flag in required):
        raise ComfyRuntimeError("This ComfyUI installation lacks the required safe startup flags. Use connect mode.")
    frontend_candidates = (
        python.parent / "Lib" / "site-packages" / "comfyui_frontend_package" / "static",
        python.parent.parent / "Lib" / "site-packages" / "comfyui_frontend_package" / "static",
        comfy / "web",
    )
    frontend = next((p for p in frontend_candidates if (p / "index.html").is_file()), None)
    if frontend is None:
        raise ComfyRuntimeError("The selected ComfyUI Python has no installed local frontend. Nothing was installed.")
    models = comfy / "models"
    if not models.is_dir():
        raise ComfyRuntimeError("The selected ComfyUI installation has no models directory.")
    return ComfyInstallation(
        comfy.parent if portable else comfy, python, comfy / "main.py", models,
        frontend.resolve(), ".".join(match.groups()), portable,
    )


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_gpu_device(index: int) -> str:
    """Resolve the physical NVIDIA index before Comfy initializes native code."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise ComfyRuntimeError("nvidia-smi was not found. Install the NVIDIA driver or use connect mode.")
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        with external_process_environment():
            result = subprocess.run(
                [executable, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
                env=sanitized_external_environment(os.environ), **kwargs,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ComfyRuntimeError("Could not identify the selected NVIDIA GPU. Use connect mode or check the driver.") from exc
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[0] == str(index) and re.fullmatch(r"GPU-[0-9a-fA-F-]+", fields[1]):
            return fields[1]
    raise ComfyRuntimeError(f"NVIDIA GPU {index} was not found.")


def _cancelled(value) -> bool:
    if value is None:
        return False
    return bool(value() if callable(value) else value.is_set())


class ComfyRuntime:
    """One persistent dedicated child; connect-mode servers never become owned."""

    def __init__(
        self, state_root: str | Path, *, process_factory=subprocess.Popen,
        http_factory=requests.Session, gpu_resolver=resolve_gpu_device,
        port_factory=_available_port, clock=time.monotonic,
        wait: Callable[[float], None] = time.sleep, startup_timeout: float = 120,
    ) -> None:
        self.state_root = Path(state_root).expanduser().resolve()
        self._process_factory = process_factory
        self._http_factory = http_factory
        self._gpu_resolver = gpu_resolver
        self._port_factory = port_factory
        self._clock = clock
        self._wait = wait
        self._startup_timeout = max(1.0, float(startup_timeout))
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._process = None
        self._config = None
        self._base_url: str | None = None
        self._run_root: Path | None = None
        self._argv: list[str] = []
        self._starting = False
        self._ready = False
        self._submitted_prompts: set[str] = set()
        self._active_prompts: set[str] = set()
        self._output_tail: deque[str] = deque(maxlen=80)
        self._stop_reason = ""

    @property
    def base_url(self) -> str | None:
        with self._lock:
            return self._base_url if self._ready and self.is_running() else None

    @property
    def shutdown_reason(self) -> str:
        with self._lock:
            return self._stop_reason

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def recent_output(self) -> str:
        with self._lock:
            return "\n".join(self._output_tail)

    def register_prompt(self, prompt_id: str) -> None:
        UUID(prompt_id)
        with self._lock:
            if self._ready and self.is_running():
                self._submitted_prompts.add(prompt_id)
                self._active_prompts.add(prompt_id)

    def finish_prompt(self, prompt_id: str) -> None:
        with self._lock:
            self._active_prompts.discard(prompt_id)

    def _json_get(self, url: str):
        session = self._http_factory()
        session.trust_env = False
        try:
            response = session.get(url, timeout=(1, 2), allow_redirects=False, stream=True)
            try:
                if response.status_code != 200:
                    raise ComfyRuntimeError(f"ComfyUI returned HTTP {response.status_code}.")
                data = bytearray()
                for chunk in response.iter_content(chunk_size=65536):
                    data.extend(chunk)
                    if len(data) > 1024 * 1024:
                        raise ComfyRuntimeError("ComfyUI status response exceeded the size limit.")
                result = json.loads(data)
                if not isinstance(result, dict):
                    raise ComfyRuntimeError("ComfyUI returned invalid status data.")
                return result
            finally:
                response.close()
        finally:
            session.close()

    def _verify_identity(self, data: dict, argv: list[str]) -> None:
        system = data.get("system")
        if not isinstance(system, dict) or system.get("argv") != argv:
            raise ComfyRuntimeError("The responding ComfyUI server is not this app's process; refusing to attach.")

    def ensure_started(self, config: ComfyConnectionConfig, cancelled=None, on_status=None) -> str:
        if config.mode == "connect":
            return config.base_url
        if _cancelled(cancelled):
            raise ComfyRuntimeCancelled("ComfyUI startup cancelled.")
        with self._operation_lock:
            with self._lock:
                live = self.is_running()
                existing_config = self._config
                base_url = self._base_url
                argv = self._argv.copy()
            if live:
                if existing_config != config:
                    raise ComfyRuntimeBusyError("Unload the managed ComfyUI runtime before changing its settings.")
                self._verify_identity(self._json_get(base_url + "/system_stats"), argv)
                if not self.is_running():
                    raise ComfyRuntimeError("The owned ComfyUI process exited.")
                return base_url

            installation = resolve_installation(config)
            if self.state_root.is_relative_to(installation.root):
                raise ComfyRuntimeError("Managed runtime data must be outside the ComfyUI installation.")
            gpu = self._gpu_resolver(config.gpu_index)
            if _cancelled(cancelled):
                raise ComfyRuntimeCancelled("ComfyUI startup cancelled.")
            run_root = self.state_root / ("runtime-" + str(uuid4()))
            for relative in ("input", "output", "scratch", "user", "os-temp", "cache/huggingface", "cache/torch", "cache/inductor", "cache/triton", "cache/cuda"):
                (run_root / relative).mkdir(parents=True, exist_ok=False)
            port = self._port_factory()
            server_argv = [
                str(installation.main), "--listen", "127.0.0.1", "--port", str(port),
                "--disable-auto-launch", "--cuda-device", gpu,
                "--base-directory", str(run_root), "--models-directory", str(installation.models),
                "--input-directory", str(run_root / "input"),
                "--output-directory", str(run_root / "output"),
                "--temp-directory", str(run_root / "scratch"),
                "--user-directory", str(run_root / "user"),
                "--database-url", "sqlite:///" + (run_root / "user/comfyui.db").as_posix(),
                "--front-end-root", str(installation.frontend),
                "--disable-all-custom-nodes", "--disable-api-nodes",
            ]
            if installation.portable:
                server_argv.append("--windows-standalone-build")
            if config.memory_policy == "low_cache":
                server_argv.append("--cache-none")
            environment = sanitized_external_environment(os.environ)
            for key in tuple(environment):
                if key.upper() in {"PYTHONPATH", "PYTHONHOME"}:
                    environment.pop(key)
            environment.update({
                "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                "TEMP": str(run_root / "os-temp"), "TMP": str(run_root / "os-temp"),
                "HF_HOME": str(run_root / "cache/huggingface"),
                "TORCH_HOME": str(run_root / "cache/torch"),
                "TORCHINDUCTOR_CACHE_DIR": str(run_root / "cache/inductor"),
                "TRITON_CACHE_DIR": str(run_root / "cache/triton"),
                "CUDA_CACHE_PATH": str(run_root / "cache/cuda"),
            })
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            with self._lock:
                self._starting = True
                self._ready = False
                self._output_tail = deque(maxlen=80)
                self._submitted_prompts.clear()
                self._active_prompts.clear()
            try:
                if on_status:
                    on_status("Starting ComfyUI…")
                with external_process_environment():
                    process = self._process_factory(
                        [str(installation.python), "-B", "-s", "-u", *server_argv],
                        cwd=str(installation.main.parent), env=environment,
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs,
                    )
                with self._lock:
                    self._process = process
                    self._config = config
                    self._argv = server_argv
                    self._base_url = f"http://127.0.0.1:{port}"
                    self._run_root = run_root
                self._read_output(process, self._output_tail)
                deadline = self._clock() + self._startup_timeout
                while self._clock() < deadline:
                    if _cancelled(cancelled):
                        raise ComfyRuntimeCancelled("ComfyUI startup cancelled.")
                    if process.poll() is not None:
                        raise ComfyRuntimeError(f"ComfyUI exited during startup (code {process.returncode}).\n{self.recent_output()}")
                    try:
                        data = self._json_get(self._base_url + "/system_stats")
                    except (requests.RequestException, json.JSONDecodeError):
                        self._wait(0.2)
                        continue
                    self._verify_identity(data, server_argv)
                    if process.poll() is not None:
                        raise ComfyRuntimeError("ComfyUI exited during readiness verification.")
                    with self._lock:
                        self._ready = True
                    return self._base_url
                raise ComfyRuntimeError(f"ComfyUI did not become ready within {self._startup_timeout:g} seconds.\n{self.recent_output()}")
            except BaseException:
                # Until readiness succeeds this child has never been exposed to a job.
                # Its handle, not the responding port, is the only termination target.
                self._terminate_owned()
                raise
            finally:
                with self._lock:
                    self._starting = False

    def _read_output(self, process, tail: deque[str]) -> None:
        if process.stdout is None:
            return

        def read():
            try:
                while True:
                    line = process.stdout.readline(4096)
                    if not line:
                        break
                    with self._lock:
                        tail.append(line.rstrip())
            except (OSError, ValueError):
                pass
            finally:
                process.stdout.close()

        threading.Thread(target=read, name="comfy-output", daemon=True).start()

    def can_shutdown(self) -> bool:
        with self._lock:
            self._stop_reason = ""
            if self._starting:
                self._stop_reason = "Cancel the pending ComfyUI startup before closing."
                return False
            if not self.is_running():
                return True
            base_url, argv = self._base_url, self._argv.copy()
        try:
            self._verify_identity(self._json_get(base_url + "/system_stats"), argv)
            queue = self._json_get(base_url + "/queue")
            running, pending = queue.get("queue_running"), queue.get("queue_pending")
            if not isinstance(running, list) or not isinstance(pending, list):
                raise ComfyRuntimeError("The server returned an invalid queue.")
            ids = []
            for item in running + pending:
                if not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[1], str):
                    raise ComfyRuntimeError("The server returned an invalid queue entry.")
                ids.append(item[1])
            with self._lock:
                foreign = set(ids) - self._submitted_prompts
                if foreign:
                    self._stop_reason = "Another client has queued or running work. Wait for it before closing or unloading ComfyUI."
                elif ids:
                    self._stop_reason = "ComfyUI is still processing this app's job. Cancel it or wait before closing."
                elif self._active_prompts:
                    self._stop_reason = "Wait for this app's ComfyUI job to finish observing or fetching its result."
                return not ids and not self._active_prompts
        except (requests.RequestException, ValueError, ComfyRuntimeError) as exc:
            with self._lock:
                if not self.is_running():
                    return True
                self._stop_reason = f"Cannot verify that the owned ComfyUI server is idle: {exc}"
            return False

    def stop(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise ComfyRuntimeBusyError("ComfyUI startup or shutdown is still in progress.")
        try:
            if not self.can_shutdown():
                raise ComfyRuntimeBusyError(self.shutdown_reason)
            self._terminate_owned()
        finally:
            self._operation_lock.release()

    def shutdown(self) -> None:
        self.stop()

    def disconnect(self) -> None:
        """Connect mode owns no transport/process; each client owns its connection."""

    def _terminate_owned(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        with self._lock:
            self._process = None
            self._config = None
            self._base_url = None
            self._ready = False
            self._argv = []
            self._active_prompts.clear()
            self._submitted_prompts.clear()
        # Retain per-run artifacts. Never sweep an installation's input/output/temp.
