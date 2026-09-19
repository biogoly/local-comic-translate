"""Qt process client for the isolated FLUX Artistic Edit worker."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .contracts import (
    ArtisticEditRequest,
    ArtisticEditResult,
    ContractError,
    DevicePolicy,
    FluxModelSpec,
    LoraSpec,
    MAX_PROTOCOL_LINE_LENGTH,
    encode_json_line,
    message_matches_request,
    parse_protocol_line,
)


class FluxWorkerClient(QObject):
    """Own one worker process and expose validated worker events as Qt signals."""

    ready = Signal(object)
    loading = Signal(str)
    progress = Signal(int, int)
    result = Signal(object)
    cancelled = Signal(str)
    error = Signal(str, str, str)
    diagnostics = Signal(str)
    worker_crashed = Signal(str)

    def __init__(
        self,
        temp_root: str | Path,
        *,
        python_executable: str | Path | None = None,
        python_path_entries: Iterable[str | Path] = (),
        cuda_device_index: int = 0,
        allow_downloads: bool = False,
        process_factory: Callable[[QObject], QProcess] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.temp_root = Path(temp_root).expanduser().resolve(strict=False)
        self.python_executable = str(
            Path(python_executable or sys.executable).expanduser().resolve(strict=False)
        )
        self.python_path_entries = tuple(
            str(Path(entry).expanduser().resolve(strict=False))
            for entry in python_path_entries
            if str(entry).strip()
        )
        self.cuda_device_index = max(0, int(cuda_device_index))
        self.allow_downloads = bool(allow_downloads)
        factory = process_factory or QProcess
        self.process = factory(self)
        self._stdout_buffer = bytearray()
        self._stderr_tail: list[str] = []
        self._pending: dict[str, str] = {}
        self._active_generation_id: str | None = None
        self._startup_policy: DevicePolicy | None = None
        self._expected_shutdown = False
        self._process_modifier = None

        self.process.started.connect(self._on_started)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_process_error)

    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, device_policy: DevicePolicy | str) -> None:
        policy = DevicePolicy(device_policy)
        if self.is_running():
            if policy != self._startup_policy:
                raise RuntimeError("Restart the FLUX worker before changing device policy.")
            return
        if not self.temp_root.is_dir():
            raise FileNotFoundError(
                f"Artistic Edit temporary directory does not exist: {self.temp_root}"
            )
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(
                f"Artistic Edit Python executable does not exist: {self.python_executable}"
            )

        self._stdout_buffer.clear()
        self._stderr_tail.clear()
        self._pending.clear()
        self._active_generation_id = None
        self._startup_policy = policy
        self._expected_shutdown = False

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        # The FLUX runtime is deliberately isolated from the main application.
        # Inheriting either of these variables can make the worker import torch,
        # diffusers, or compiled extensions from the app environment instead of
        # the selected FLUX Python.  On Windows that can leave from_pretrained()
        # waiting indefinitely rather than producing a useful import error.
        environment.remove("PYTHONPATH")
        environment.remove("PYTHONHOME")
        if self.python_path_entries:
            environment.insert("PYTHONPATH", os.pathsep.join(self.python_path_entries))
        if policy == DevicePolicy.SINGLE_GPU_0:
            environment.insert("CUDA_VISIBLE_DEVICES", "0")
        elif policy == DevicePolicy.SINGLE_GPU_1:
            environment.insert("CUDA_VISIBLE_DEVICES", "1")
        elif policy in {
            DevicePolicy.MODEL_CPU_OFFLOAD,
            DevicePolicy.SEQUENTIAL_CPU_OFFLOAD,
        }:
            environment.insert("CUDA_VISIBLE_DEVICES", str(self.cuda_device_index))
        elif policy == DevicePolicy.CPU:
            environment.insert("CUDA_VISIBLE_DEVICES", "")
        elif policy == DevicePolicy.BALANCED:
            environment.remove("CUDA_VISIBLE_DEVICES")
        self.process.setProcessEnvironment(environment)

        repository_root = Path(__file__).resolve().parents[2]
        self.process.setWorkingDirectory(str(repository_root))
        arguments = [
            "-u",
            "-m",
            "modules.artistic_edit.flux_worker",
            "--temp-root",
            str(self.temp_root),
            "--device-policy",
            policy.value,
            "--cuda-device",
            str(self.cuda_device_index),
        ]
        if self.allow_downloads:
            arguments.append("--allow-downloads")
        self.process.setProgram(self.python_executable)
        self.process.setArguments(arguments)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)

        if os.name == "nt" and hasattr(self.process, "setCreateProcessArgumentsModifier"):
            create_no_window = 0x08000000

            def hide_console(arguments_object) -> None:
                arguments_object.flags |= create_no_window

            self._process_modifier = hide_console
            self.process.setCreateProcessArgumentsModifier(hide_console)
        self.process.start()

    def _on_started(self) -> None:
        request_id = str(uuid4())
        self._send({"id": request_id, "op": "ping"}, operation="ping")

    def _send(self, message: Mapping[str, Any], *, operation: str) -> str:
        if not self.is_running():
            raise RuntimeError("FLUX worker is not running.")
        request_id = str(message["id"])
        encoded = encode_json_line(message).encode("utf-8")
        self._pending[request_id] = operation
        written = self.process.write(encoded)
        if written < 0:
            self._pending.pop(request_id, None)
            raise RuntimeError("Could not write to the FLUX worker.")
        return request_id

    def load(
        self,
        model: FluxModelSpec,
        lora: LoraSpec | None,
        device_policy: DevicePolicy | str,
    ) -> str:
        policy = DevicePolicy(device_policy)
        self._require_policy(policy)
        if lora is not None and lora.base_model != model.key:
            raise ContractError("Selected LoRA does not match the selected FLUX model size.")
        request_id = str(uuid4())
        return self._send(
            {
                "id": request_id,
                "op": "load",
                "model": model.to_dict(),
                "lora": lora.to_dict() if lora is not None else None,
                "device_policy": policy.value,
            },
            operation="load",
        )

    def generate(self, request: ArtisticEditRequest) -> str:
        request.validate_paths(self.temp_root)
        self._require_policy(request.device_policy)
        if self._active_generation_id is not None:
            raise RuntimeError("An Artistic Edit generation is already active.")
        self._active_generation_id = request.request_id
        try:
            return self._send(request.to_command(), operation="generate")
        except Exception:
            self._active_generation_id = None
            raise

    def cancel(self) -> bool:
        request_id = self._active_generation_id
        if request_id is None or not self.is_running():
            return False
        encoded = encode_json_line({"id": request_id, "op": "cancel"}).encode("utf-8")
        if self.process.write(encoded) < 0:
            raise RuntimeError("Could not send cancellation to the FLUX worker.")
        return True

    def unload(self) -> str:
        request_id = str(uuid4())
        return self._send({"id": request_id, "op": "unload"}, operation="unload")

    def shutdown(self) -> str | None:
        if not self.is_running():
            return None
        request_id = str(uuid4())
        self._expected_shutdown = True
        return self._send({"id": request_id, "op": "shutdown"}, operation="shutdown")

    def close(self, timeout_ms: int = 5000) -> None:
        if not self.is_running():
            return
        try:
            self.shutdown()
        except RuntimeError:
            pass
        if not self.process.waitForFinished(max(0, int(timeout_ms))):
            self.process.kill()
            self.process.waitForFinished(2000)

    def _require_policy(self, policy: DevicePolicy) -> None:
        if not self.is_running():
            raise RuntimeError("Start the FLUX worker before sending a request.")
        if policy != self._startup_policy:
            raise RuntimeError("Request device policy does not match the running FLUX worker.")

    @staticmethod
    def _read_bytes(value: Any) -> bytes:
        try:
            return bytes(value)
        except TypeError:
            data = getattr(value, "data", None)
            return bytes(data()) if callable(data) else b""

    def _on_stdout(self) -> None:
        self._stdout_buffer.extend(self._read_bytes(self.process.readAllStandardOutput()))
        while b"\n" in self._stdout_buffer:
            raw_line, _, remainder = self._stdout_buffer.partition(b"\n")
            self._stdout_buffer = bytearray(remainder)
            if raw_line.strip():
                try:
                    line = raw_line.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    self.error.emit("protocol_error", str(exc), repr(bytes(raw_line[:1000])))
                    continue
                self._consume_stdout_line(line)
        if len(self._stdout_buffer) > MAX_PROTOCOL_LINE_LENGTH:
            self._stdout_buffer.clear()
            self.error.emit(
                "protocol_error",
                "Worker stdout line exceeded the protocol size limit.",
                "",
            )

    def _consume_stdout_line(self, line: str) -> None:
        try:
            message = parse_protocol_line(line, temp_root=self.temp_root)
            if "event" not in message:
                raise ContractError("Worker stdout contained a command instead of an event.")
        except (ContractError, UnicodeError) as exc:
            self.error.emit("protocol_error", str(exc), line[:1000])
            return

        request_id = message["id"]
        if request_id not in self._pending:
            return
        event = message["event"]
        try:
            if event == "loading":
                text = message.get("message")
                if not isinstance(text, str):
                    raise ContractError("Loading event message must be a string.")
                self.loading.emit(text)
                return
            if event == "progress":
                step = message.get("step")
                total = message.get("total")
                if (
                    isinstance(step, bool)
                    or isinstance(total, bool)
                    or not isinstance(step, int)
                    or not isinstance(total, int)
                    or step < 0
                    or total <= 0
                    or step > total
                ):
                    raise ContractError("Progress event contains invalid step counts.")
                self.progress.emit(step, total)
                return
            if event == "result":
                if not message_matches_request(message, self._active_generation_id):
                    return
                result = ArtisticEditResult.from_dict(
                    {
                        "request_id": request_id,
                        "output_path": message.get("output_path"),
                        "seed": message.get("seed"),
                        "elapsed_seconds": message.get("elapsed_seconds"),
                        "width": message.get("width"),
                        "height": message.get("height"),
                    },
                    temp_root=self.temp_root,
                )
                self._complete(request_id)
                self.result.emit(result)
                return
            if event == "cancelled":
                self._complete(request_id)
                self.cancelled.emit(request_id)
                return
            if event == "error":
                code = message.get("code")
                text = message.get("message")
                details = message.get("details", "")
                if not all(isinstance(value, str) for value in (code, text, details)):
                    raise ContractError("Error event fields must be strings.")
                self._complete(request_id)
                self.error.emit(code, text, details)
                return
            if event == "ready":
                capabilities = message.get("capabilities", {})
                if not isinstance(capabilities, dict):
                    raise ContractError("Ready event capabilities must be an object.")
                self._complete(request_id)
                self.ready.emit(capabilities)
                return
            raise ContractError(f"Unsupported worker event: {event}.")
        except ContractError as exc:
            self._complete(request_id)
            self.error.emit("protocol_error", str(exc), line[:1000])

    def _complete(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
        if self._active_generation_id == request_id:
            self._active_generation_id = None

    def _on_stderr(self) -> None:
        data = self._read_bytes(self.process.readAllStandardError())
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        self._stderr_tail.extend(line for line in text.splitlines() if line.strip())
        self._stderr_tail = self._stderr_tail[-80:]
        self.diagnostics.emit(text)

    def _on_finished(self, exit_code: int, _exit_status: Any) -> None:
        expected = self._expected_shutdown
        details = "\n".join(self._stderr_tail[-40:])
        self._pending.clear()
        self._active_generation_id = None
        self._startup_policy = None
        self._expected_shutdown = False
        if not expected or exit_code != 0:
            message = f"FLUX worker exited unexpectedly with code {exit_code}."
            if details:
                message += f"\nRecent worker output:\n{details}"
            self.worker_crashed.emit(message)

    def _on_process_error(self, process_error: Any) -> None:
        if self._expected_shutdown:
            return
        self.worker_crashed.emit(f"FLUX worker process error: {process_error}")
