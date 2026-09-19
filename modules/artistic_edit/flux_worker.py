"""Isolated JSON-lines worker for optional FLUX.2 Klein image editing.

Torch and Diffusers are imported only after the worker receives a load or
generate command. Importing this module in the normal application process is
therefore safe when the optional FLUX environment is not installed.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from typing import Any, Callable, Mapping, TextIO
from uuid import uuid4

from PIL import Image, ImageOps

from .contracts import (
    ArtisticEditRequest,
    ArtisticEditResult,
    ContractError,
    DevicePolicy,
    FluxModelSpec,
    LoraSpec,
    ModelKey,
    encode_json_line,
    parse_protocol_line,
    validate_request_id,
)
from .lora_registry import LoraCompatibilityError, validate_lora_compatibility


logger = logging.getLogger(__name__)
PROTOCOL_VERSION = 1
ADAPTER_NAME = "artistic_edit"


def configure_protocol_streams() -> None:
    """Make the JSON-lines protocol UTF-8 regardless of the Windows code page."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")


class GenerationCancelled(RuntimeError):
    """Private control-flow exception raised from the Diffusers callback."""


class WorkerDependencyError(RuntimeError):
    """The optional FLUX runtime is missing or incompatible."""


class ModelNotFoundError(RuntimeError):
    """The selected model source is absent from the local cache/path."""


class ModelAccessDeniedError(RuntimeError):
    """The selected model requires credentials or license acceptance."""


@dataclass(frozen=True)
class RuntimeDependencies:
    torch: Any
    pipeline_class: Any


@dataclass(frozen=True)
class PipelineKey:
    model_key: ModelKey
    source: str
    device_policy: DevicePolicy
    dtype_name: str = "bfloat16"


def import_runtime_dependencies() -> RuntimeDependencies:
    """Import heavyweight packages lazily inside the worker process."""
    try:
        import torch
        from diffusers import Flux2KleinPipeline
    except (ImportError, OSError) as exc:
        raise WorkerDependencyError(
            "The optional FLUX runtime is unavailable. Install requirements-flux.txt "
            "into the configured Artistic Edit Python environment."
        ) from exc
    return RuntimeDependencies(torch=torch, pipeline_class=Flux2KleinPipeline)


def configure_visible_devices(policy: DevicePolicy, cuda_device_index: int = 0) -> None:
    """Select physical CUDA devices before Torch is imported."""
    if policy == DevicePolicy.SINGLE_GPU_0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif policy == DevicePolicy.SINGLE_GPU_1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif policy in {
        DevicePolicy.MODEL_CPU_OFFLOAD,
        DevicePolicy.SEQUENTIAL_CPU_OFFLOAD,
    }:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(max(0, cuda_device_index))
    elif policy == DevicePolicy.CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Balanced multi-GPU intentionally preserves the inherited device list.


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class FluxModelManager:
    """Own and reuse at most one FLUX pipeline inside the worker."""

    def __init__(
        self,
        *,
        dependencies: RuntimeDependencies | None = None,
        local_files_only: bool = True,
        balanced_max_memory_gib: int = 21,
    ) -> None:
        self._dependencies = dependencies
        self.local_files_only = local_files_only
        self.balanced_max_memory_gib = balanced_max_memory_gib
        self.pipeline: Any | None = None
        self.pipeline_key: PipelineKey | None = None
        self.active_lora: LoraSpec | None = None

    @property
    def dependencies(self) -> RuntimeDependencies:
        if self._dependencies is None:
            self._dependencies = import_runtime_dependencies()
        return self._dependencies

    def ensure_loaded(
        self,
        model: FluxModelSpec,
        lora: LoraSpec | None,
        device_policy: DevicePolicy,
        cancel_event: threading.Event,
        status: Callable[..., None],
        runtime_ready: Callable[[], None] | None = None,
    ) -> Any:
        try:
            requested_key = PipelineKey(model.key, model.source, device_policy)
            if self.pipeline is None or self.pipeline_key != requested_key:
                if self.pipeline is not None:
                    self.unload()
                if cancel_event.is_set():
                    raise GenerationCancelled()
                status(
                    "loading",
                    message=f"Preparing {model.key.value} (BF16) runtime…",
                )
                self.pipeline = self._create_pipeline(model, device_policy, status=status)
                self.pipeline_key = requested_key
                self.active_lora = None
                if cancel_event.is_set():
                    self.unload()
                    raise GenerationCancelled()

            self._sync_lora(lora)
            if cancel_event.is_set():
                raise GenerationCancelled()
        finally:
            # On Windows, importing NumPy/PyTorch compiled extensions can stall
            # while another Python thread is blocked reading a QProcess pipe.
            # The server pauses that reader until runtime/model setup reaches
            # this point, then resumes it so generation cancellation stays live.
            if runtime_ready is not None:
                runtime_ready()
        return self.pipeline

    def _create_pipeline(
        self,
        model: FluxModelSpec,
        device_policy: DevicePolicy,
        *,
        status: Callable[..., None] | None = None,
    ) -> Any:
        if status is not None:
            status("loading", message="Importing PyTorch and Diffusers…")
        dependencies = self.dependencies
        torch = dependencies.torch
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "local_files_only": self.local_files_only,
        }
        if device_policy == DevicePolicy.BALANCED:
            memory = f"{self.balanced_max_memory_gib}GiB"
            device_count = int(torch.cuda.device_count())
            if device_count < 2:
                raise RuntimeError("Balanced multi-GPU requires at least two visible CUDA devices.")
            load_kwargs["device_map"] = "balanced"
            load_kwargs["max_memory"] = {
                index: memory for index in range(device_count)
            }

        try:
            if status is not None:
                status("loading", message=f"Loading {model.key.value} BF16 weights…")
            pipeline = dependencies.pipeline_class.from_pretrained(model.source, **load_kwargs)
        except OSError as exc:
            lowered = str(exc).lower()
            if "401" in lowered or "403" in lowered or "access denied" in lowered:
                raise ModelAccessDeniedError(
                    f"Access to FLUX model {model.source!r} was denied."
                ) from exc
            raise ModelNotFoundError(
                f"FLUX model {model.source!r} is unavailable in the local cache or path."
            ) from exc
        vae = getattr(pipeline, "vae", None)
        enable_tiling = getattr(vae, "enable_tiling", None)
        if callable(enable_tiling):
            enable_tiling()

        if status is not None:
            status("loading", message=f"Configuring {device_policy.value.replace('_', ' ')}…")
        if device_policy in {DevicePolicy.SINGLE_GPU_0, DevicePolicy.SINGLE_GPU_1}:
            pipeline.to("cuda:0")
        elif device_policy == DevicePolicy.MODEL_CPU_OFFLOAD:
            pipeline.enable_model_cpu_offload()
        elif device_policy == DevicePolicy.SEQUENTIAL_CPU_OFFLOAD:
            pipeline.enable_sequential_cpu_offload()
        elif device_policy == DevicePolicy.CPU:
            pipeline.to("cpu")
        return pipeline

    def _sync_lora(self, requested: LoraSpec | None) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot configure a LoRA before the base model is loaded.")
        current = self.active_lora
        if current == requested:
            return

        same_weights = (
            current is not None
            and requested is not None
            and current.path == requested.path
            and current.sha256 == requested.sha256
        )
        if same_weights:
            self.pipeline.set_adapters(
                [ADAPTER_NAME],
                adapter_weights=[requested.scale],
            )
            self.active_lora = requested
            return

        if current is not None:
            self.pipeline.unload_lora_weights()
            self.active_lora = None
        if requested is None:
            return

        path = Path(requested.path)
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file was not found: {path}")
        actual_sha256 = _sha256_file(path)
        if actual_sha256.lower() != requested.sha256.lower():
            raise ContractError("LoRA file changed after selection; refresh the LoRA list.")
        validate_lora_compatibility(path, requested.base_model)
        try:
            self.pipeline.load_lora_weights(str(path), adapter_name=ADAPTER_NAME)
            self.pipeline.set_adapters(
                [ADAPTER_NAME],
                adapter_weights=[requested.scale],
            )
        except Exception:
            try:
                self.pipeline.unload_lora_weights()
            except Exception:
                logger.exception("Failed to clean up after a LoRA load error")
            raise
        self.active_lora = requested

    def generate(
        self,
        request: ArtisticEditRequest,
        cancel_event: threading.Event,
        status: Callable[..., None],
        runtime_ready: Callable[[], None] | None = None,
    ) -> ArtisticEditResult:
        pipeline = self.ensure_loaded(
            request.model,
            request.lora,
            request.device_policy,
            cancel_event,
            status,
            runtime_ready,
        )
        input_path = Path(request.input_path)
        output_path = Path(request.output_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Artistic Edit input was not found: {input_path}")
        if not output_path.parent.is_dir():
            raise FileNotFoundError(
                f"Artistic Edit output directory was not found: {output_path.parent}"
            )

        with Image.open(input_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        if image.size != (request.width, request.height):
            raise ContractError(
                "Input PNG dimensions do not match the requested generation dimensions."
            )
        if cancel_event.is_set():
            raise GenerationCancelled()

        generator = self.dependencies.torch.Generator(device="cpu").manual_seed(request.seed)

        def on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
            if cancel_event.is_set():
                raise GenerationCancelled()
            status("progress", step=int(step_index) + 1, total=request.steps)
            if cancel_event.is_set():
                raise GenerationCancelled()
            return callback_kwargs

        started = time.monotonic()
        generated = pipeline(
            image=image,
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            generator=generator,
            callback_on_step_end=on_step_end,
        ).images[0]
        if cancel_event.is_set():
            raise GenerationCancelled()
        generated = generated.convert("RGB")
        if generated.size != (request.width, request.height):
            raise RuntimeError(
                "FLUX returned unexpected dimensions: "
                f"{generated.width}x{generated.height}."
            )

        temporary_output = output_path.with_name(
            f".{output_path.stem}.{uuid4().hex}.tmp"
        )
        try:
            generated.save(temporary_output, format="PNG")
            with Image.open(temporary_output) as verification:
                verification.load()
                if verification.size != (request.width, request.height):
                    raise RuntimeError("Temporary FLUX output failed dimension verification.")
            os.replace(temporary_output, output_path)
        finally:
            try:
                temporary_output.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary FLUX output: %s", temporary_output)

        return ArtisticEditResult(
            request_id=request.request_id,
            output_path=str(output_path),
            seed=request.seed,
            elapsed_seconds=time.monotonic() - started,
            width=request.width,
            height=request.height,
        )

    def unload(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        self.pipeline_key = None
        self.active_lora = None
        if pipeline is not None:
            remove_hooks = getattr(pipeline, "remove_all_hooks", None)
            if callable(remove_hooks):
                try:
                    remove_hooks()
                except Exception:
                    logger.exception("Failed to remove FLUX offload hooks")
        # Drop the last strong reference before collecting and flushing the CUDA
        # allocator so the next model load starts from a clean GPU state.
        del pipeline
        gc.collect()

        dependencies = self._dependencies
        if dependencies is None:
            return
        cuda = getattr(dependencies.torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                synchronize = getattr(cuda, "synchronize", None)
                if callable(synchronize):
                    synchronize()
                cuda.empty_cache()
        except Exception:
            logger.exception("Failed while releasing the CUDA allocator cache")


def classify_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, WorkerDependencyError):
        return "dependency_missing", str(exc)
    if isinstance(exc, LoraCompatibilityError):
        return "lora_incompatible", str(exc)
    if isinstance(exc, ModelAccessDeniedError):
        return "model_access_denied", str(exc)
    if isinstance(exc, ModelNotFoundError):
        return "model_not_found", str(exc)
    if isinstance(exc, ContractError):
        return "invalid_request", str(exc)
    if isinstance(exc, FileNotFoundError):
        return "invalid_request", str(exc)
    name = type(exc).__name__.lower()
    message = str(exc)
    lowered = message.lower()
    if "outofmemory" in name or "out of memory" in lowered or "cuda oom" in lowered:
        return "cuda_oom", "FLUX ran out of GPU memory. Unload the model or choose a lower-memory policy."
    if "401" in lowered or "403" in lowered or "access denied" in lowered:
        return "model_access_denied", message
    return "worker_error", message or type(exc).__name__


class FluxWorkerServer:
    """Queue commands from stdin while keeping cancellation responsive."""

    def __init__(
        self,
        temp_root: str | Path,
        startup_policy: DevicePolicy,
        *,
        manager: FluxModelManager | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.temp_root = Path(temp_root).resolve(strict=False)
        self.startup_policy = DevicePolicy(startup_policy)
        self.manager = manager or FluxModelManager()
        self.stdin = stdin or sys.stdin
        self.protocol_stdout = stdout or sys.stdout
        self._commands: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._cancelled_ids: set[str] = set()
        self._active_request_id: str | None = None
        self._shutdown_requested = False
        self._reader_resume = threading.Event()
        self._reader_resume.set()

    def emit(self, request_id: str, event: str, **fields: Any) -> None:
        message = {"id": request_id, "event": event, **fields}
        encoded = encode_json_line(message)
        with self._write_lock:
            self.protocol_stdout.write(encoded)
            self.protocol_stdout.flush()

    def _emit_error(self, request_id: str, exc: Exception) -> None:
        code, message = classify_error(exc)
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        self.emit(request_id, "error", code=code, message=message, details=details)

    def _reader(self) -> None:
        try:
            for line in self.stdin:
                try:
                    command = parse_protocol_line(line, temp_root=self.temp_root)
                except Exception as exc:
                    logger.warning("Rejected malformed worker command", exc_info=True)
                    request_id = str(uuid4())
                    try:
                        raw_message = json.loads(line)
                        if isinstance(raw_message, dict):
                            request_id = validate_request_id(raw_message.get("id"))
                    except Exception:
                        pass
                    self._emit_error(request_id, exc)
                    continue
                operation = command["op"]
                request_id = command["id"]
                if operation == "cancel":
                    with self._state_lock:
                        self._cancelled_ids.add(request_id)
                        if self._active_request_id == request_id:
                            self._cancel_event.set()
                    continue
                if operation == "shutdown":
                    with self._state_lock:
                        self._shutdown_requested = True
                        self._cancel_event.set()
                pause_for_runtime = operation in {"load", "generate"}
                if pause_for_runtime:
                    self._reader_resume.clear()
                self._commands.put(command)
                if pause_for_runtime:
                    self._reader_resume.wait()
        finally:
            self._commands.put(None)

    def _status(self, request_id: str) -> Callable[..., None]:
        return lambda event, **fields: self.emit(request_id, event, **fields)

    @staticmethod
    def _require_exact_fields(
        command: Mapping[str, Any],
        required: set[str],
        optional: set[str] | None = None,
    ) -> None:
        optional = optional or set()
        actual = set(command)
        missing = required - actual
        extra = actual - required - optional
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing fields: {', '.join(sorted(missing))}")
            if extra:
                problems.append(f"unexpected fields: {', '.join(sorted(extra))}")
            raise ContractError("; ".join(problems))

    def _parse_load(self, command: Mapping[str, Any]):
        self._require_exact_fields(
            command,
            {"id", "op", "model", "lora", "device_policy"},
        )
        model = FluxModelSpec.from_dict(command["model"])
        lora = None if command["lora"] is None else LoraSpec.from_dict(command["lora"])
        try:
            policy = DevicePolicy(command["device_policy"])
        except (TypeError, ValueError) as exc:
            raise ContractError("load.device_policy is invalid.") from exc
        if lora is not None and lora.base_model != model.key:
            raise ContractError("Selected LoRA does not match the selected FLUX model size.")
        return model, lora, policy

    def _parse_generate(self, command: Mapping[str, Any]) -> ArtisticEditRequest:
        payload = dict(command)
        payload.pop("op", None)
        payload["request_id"] = payload.pop("id")
        return ArtisticEditRequest.from_dict(payload, temp_root=self.temp_root)

    def _validate_policy(self, policy: DevicePolicy) -> None:
        if policy != self.startup_policy:
            raise ContractError(
                "Worker device policy differs from its startup policy; restart the worker."
            )

    def _call_backend(self, function: Callable[..., Any], *args, **kwargs):
        # Third-party print calls must never corrupt protocol stdout.
        with redirect_stdout(sys.stderr):
            return function(*args, **kwargs)

    def process_command(self, command: Mapping[str, Any]) -> bool:
        request_id = str(command.get("id", uuid4()))
        operation = command.get("op")
        try:
            if operation == "ping":
                self._require_exact_fields(command, {"id", "op"})
                self.emit(
                    request_id,
                    "ready",
                    capabilities={
                        "protocol_version": PROTOCOL_VERSION,
                        "model_keys": [key.value for key in ModelKey],
                        "device_policy": self.startup_policy.value,
                    },
                )
                return True

            if operation == "load":
                model, lora, policy = self._parse_load(command)
                self._validate_policy(policy)
                self._call_backend(
                    self.manager.ensure_loaded,
                    model,
                    lora,
                    policy,
                    self._cancel_event,
                    self._status(request_id),
                    self._reader_resume.set,
                )
                self.emit(
                    request_id,
                    "ready",
                    capabilities={
                        "loaded_model": model.key.value,
                    },
                )
                return True

            if operation == "generate":
                request = self._parse_generate(command)
                self._validate_policy(request.device_policy)
                result = self._call_backend(
                    self.manager.generate,
                    request,
                    self._cancel_event,
                    self._status(request_id),
                    self._reader_resume.set,
                )
                self.emit(request_id, "result", **{
                    key: value
                    for key, value in result.to_dict().items()
                    if key != "request_id"
                })
                return True

            if operation == "unload":
                self._require_exact_fields(command, {"id", "op"})
                self._call_backend(self.manager.unload)
                self.emit(request_id, "ready", capabilities={"loaded_model": None})
                return True

            if operation == "shutdown":
                self._require_exact_fields(command, {"id", "op"})
                self._call_backend(self.manager.unload)
                self.emit(request_id, "ready", capabilities={"shutdown": True})
                return False

            raise ContractError(f"Unknown worker operation: {operation!r}.")
        except GenerationCancelled:
            self.emit(request_id, "cancelled")
            return True
        except Exception as exc:
            logger.exception("FLUX worker operation failed: %s", operation)
            self._emit_error(request_id, exc)
            return operation != "shutdown"
        finally:
            if operation in {"load", "generate"}:
                self._reader_resume.set()

    def serve(self) -> int:
        reader = threading.Thread(target=self._reader, name="flux-command-reader", daemon=True)
        reader.start()
        keep_running = True
        while keep_running:
            command = self._commands.get()
            if command is None:
                break
            request_id = command["id"]
            operation = command["op"]
            if operation in {"load", "generate"}:
                with self._state_lock:
                    self._active_request_id = request_id
                    self._cancel_event.clear()
                    if request_id in self._cancelled_ids or self._shutdown_requested:
                        self._cancel_event.set()
            keep_running = self.process_command(command)
            with self._state_lock:
                if self._active_request_id == request_id:
                    self._active_request_id = None
                self._cancelled_ids.discard(request_id)
                self._cancel_event.clear()
        self._call_backend(self.manager.unload)
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Comic Translate FLUX worker")
    parser.add_argument("--temp-root", required=True, type=Path)
    parser.add_argument(
        "--device-policy",
        required=True,
        choices=[policy.value for policy in DevicePolicy],
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face model downloads instead of cache-only loading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_protocol_streams()
    args = parse_args(argv)
    temp_root = args.temp_root.resolve(strict=False)
    if not temp_root.is_dir():
        print(f"FLUX temporary directory does not exist: {temp_root}", file=sys.stderr)
        return 2
    policy = DevicePolicy(args.device_policy)
    configure_visible_devices(policy, args.cuda_device)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manager = FluxModelManager(local_files_only=not args.allow_downloads)
    return FluxWorkerServer(temp_root, policy, manager=manager).serve()


if __name__ == "__main__":
    raise SystemExit(main())
