"""Immutable, JSON-safe contracts shared by the GUI and FLUX worker.

The module deliberately depends only on the Python standard library. It can be
imported by the normal application even when the optional FLUX environment is
not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import UUID


MAX_PROMPT_LENGTH = 8192
MAX_MODEL_SOURCE_LENGTH = 2048
MAX_PROTOCOL_LINE_LENGTH = 1_000_000
MAX_IMAGE_DIMENSION = 16384
MIN_LORA_SCALE = 0.0
MAX_LORA_SCALE = 2.0

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ContractError(ValueError):
    """Raised when an artistic-edit message is malformed or unsafe."""


class ModelKey(StrEnum):
    KLEIN_4B = "klein-4b"
    KLEIN_9B = "klein-9b"


class SelectionMode(StrEnum):
    SELECTED_BOX = "selected_box"
    PAINTED_AREA = "painted_area"
    WHOLE_PAGE = "whole_page"


class DevicePolicy(StrEnum):
    SINGLE_GPU_0 = "single_gpu_0"
    SINGLE_GPU_1 = "single_gpu_1"
    BALANCED = "balanced"
    MODEL_CPU_OFFLOAD = "model_cpu_offload"
    SEQUENTIAL_CPU_OFFLOAD = "sequential_cpu_offload"
    CPU = "cpu"


WORKER_OPERATIONS = frozenset({"ping", "load", "generate", "cancel", "unload", "shutdown"})
WORKER_EVENTS = frozenset({"ready", "loading", "progress", "result", "cancelled", "error"})


def _enum_value(enum_type, value: Any, field: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ContractError(f"{field} must be one of: {choices}.") from exc


def _nonempty_string(
    value: Any,
    field: str,
    maximum: int,
    *,
    allow_newlines: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ContractError(f"{field} cannot be empty.")
    if len(normalized) > maximum:
        raise ContractError(f"{field} exceeds {maximum} characters.")
    permitted_controls = {"\n", "\r", "\t"} if allow_newlines else set()
    if any(
        ord(character) < 32 and character not in permitted_controls
        for character in normalized
    ):
        raise ContractError(f"{field} cannot contain control characters.")
    return normalized


def validate_request_id(value: Any) -> str:
    request_id = _nonempty_string(value, "request_id", 64)
    try:
        UUID(request_id)
    except ValueError as exc:
        raise ContractError("request_id must be a UUID.") from exc
    return request_id


def validate_lora_scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("LoRA scale must be a number.")
    scale = float(value)
    if not math.isfinite(scale) or not MIN_LORA_SCALE <= scale <= MAX_LORA_SCALE:
        raise ContractError(
            f"LoRA scale must be finite and between {MIN_LORA_SCALE} and {MAX_LORA_SCALE}."
        )
    return scale


def validate_local_path(
    value: Any,
    field: str,
    *,
    root: str | os.PathLike[str] | None = None,
    suffix: str | None = None,
) -> str:
    raw_path = _nonempty_string(value, field, 32768)
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        raise ContractError(f"{field} must be an absolute path.")
    candidate = candidate.resolve(strict=False)

    if suffix is not None and candidate.suffix.lower() != suffix.lower():
        raise ContractError(f"{field} must use the {suffix} extension.")

    if root is not None:
        root_path = Path(root).expanduser().resolve(strict=False)
        try:
            relative = candidate.relative_to(root_path)
        except ValueError as exc:
            raise ContractError(f"{field} escapes the permitted temporary directory.") from exc
        if not relative.parts:
            raise ContractError(f"{field} must identify a file below the temporary directory.")

    return str(candidate)


def _finite_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{field} must be finite.")
    if minimum is not None and number < minimum:
        raise ContractError(f"{field} must be at least {minimum}.")
    return number


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{field} must be an integer.")
    if not minimum <= value <= maximum:
        raise ContractError(f"{field} must be between {minimum} and {maximum}.")
    return value


def _require_keys(data: Mapping[str, Any], required: set[str], optional: set[str] | None = None) -> None:
    optional = optional or set()
    actual = set(data)
    missing = required - actual
    extra = actual - required - optional
    if missing:
        raise ContractError(f"Missing fields: {', '.join(sorted(missing))}.")
    if extra:
        raise ContractError(f"Unexpected fields: {', '.join(sorted(extra))}.")


def _load_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ContractError("Protocol message cannot be empty.")
    if len(text) > MAX_PROTOCOL_LINE_LENGTH:
        raise ContractError("Protocol message is too large.")

    def reject_constant(value: str):
        raise ContractError(f"Non-finite JSON number is not allowed: {value}.")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except ContractError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError("Protocol message is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise ContractError("Protocol message must be a JSON object.")
    return value


def encode_json_line(message: Mapping[str, Any]) -> str:
    """Encode one finite JSON object for the worker's stdout/stdin protocol."""
    if not isinstance(message, Mapping):
        raise ContractError("Protocol message must be a mapping.")
    try:
        return json.dumps(
            dict(message),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ContractError("Protocol message contains a non-JSON-safe value.") from exc


def parse_protocol_line(
    line: str,
    *,
    temp_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Decode and minimally validate a worker command or event."""
    message = _load_json_object(line)
    has_operation = "op" in message
    has_event = "event" in message
    if has_operation == has_event:
        raise ContractError("Protocol message must contain exactly one of op or event.")

    message["id"] = validate_request_id(message.get("id"))
    discriminator = "op" if has_operation else "event"
    allowed = WORKER_OPERATIONS if has_operation else WORKER_EVENTS
    value = message.get(discriminator)
    if value not in allowed:
        raise ContractError(f"Unknown worker {discriminator}: {value!r}.")

    for field in ("input_path", "output_path"):
        if field in message:
            message[field] = validate_local_path(
                message[field],
                field,
                root=temp_root,
                suffix=".png",
            )
    return message


def message_matches_request(message: Mapping[str, Any], active_request_id: str | None) -> bool:
    """Return false for late events belonging to an inactive request."""
    if active_request_id is None:
        return False
    return message.get("id") == active_request_id


@dataclass(frozen=True)
class FluxModelSpec:
    key: ModelKey | str
    source: str
    license_label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _enum_value(ModelKey, self.key, "model.key"))
        object.__setattr__(
            self,
            "source",
            _nonempty_string(self.source, "model.source", MAX_MODEL_SOURCE_LENGTH),
        )
        object.__setattr__(
            self,
            "license_label",
            _nonempty_string(self.license_label, "model.license_label", 256),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.value,
            "source": self.source,
            "license_label": self.license_label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FluxModelSpec":
        if not isinstance(data, Mapping):
            raise ContractError("model must be an object.")
        _require_keys(data, {"key", "source", "license_label"})
        return cls(**dict(data))


DEFAULT_MODEL_SPECS = {
    ModelKey.KLEIN_4B: FluxModelSpec(
        ModelKey.KLEIN_4B,
        "black-forest-labs/FLUX.2-klein-4B",
        "Apache 2.0",
    ),
    ModelKey.KLEIN_9B: FluxModelSpec(
        ModelKey.KLEIN_9B,
        "black-forest-labs/FLUX.2-klein-9B",
        "FLUX Non-Commercial License",
    ),
}


@dataclass(frozen=True)
class LoraSpec:
    path: str
    base_model: ModelKey | str
    scale: float
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            validate_local_path(self.path, "lora.path", suffix=".safetensors"),
        )
        object.__setattr__(
            self,
            "base_model",
            _enum_value(ModelKey, self.base_model, "lora.base_model"),
        )
        object.__setattr__(self, "scale", validate_lora_scale(self.scale))
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ContractError("lora.sha256 must be a 64-character hexadecimal SHA-256.")
        object.__setattr__(self, "sha256", self.sha256.lower())

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "base_model": self.base_model.value,
            "scale": self.scale,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoraSpec":
        if not isinstance(data, Mapping):
            raise ContractError("lora must be an object.")
        _require_keys(data, {"path", "base_model", "scale", "sha256"})
        return cls(**dict(data))


@dataclass(frozen=True)
class ArtisticEditRequest:
    request_id: str
    input_path: str
    output_path: str
    prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    model: FluxModelSpec
    lora: LoraSpec | None
    device_policy: DevicePolicy | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(
            self,
            "input_path",
            validate_local_path(self.input_path, "input_path", suffix=".png"),
        )
        object.__setattr__(
            self,
            "output_path",
            validate_local_path(self.output_path, "output_path", suffix=".png"),
        )
        object.__setattr__(
            self,
            "prompt",
            _nonempty_string(
                self.prompt,
                "prompt",
                MAX_PROMPT_LENGTH,
                allow_newlines=True,
            ),
        )
        object.__setattr__(
            self,
            "width",
            _integer(self.width, "width", minimum=16, maximum=MAX_IMAGE_DIMENSION),
        )
        object.__setattr__(
            self,
            "height",
            _integer(self.height, "height", minimum=16, maximum=MAX_IMAGE_DIMENSION),
        )
        if self.width % 16 or self.height % 16:
            raise ContractError("width and height must be multiples of 16.")
        object.__setattr__(self, "steps", _integer(self.steps, "steps", minimum=1, maximum=100))
        object.__setattr__(
            self,
            "guidance",
            _finite_number(self.guidance, "guidance", minimum=0.0),
        )
        object.__setattr__(
            self,
            "seed",
            _integer(self.seed, "seed", minimum=0, maximum=(2**63) - 1),
        )
        model = self.model if isinstance(self.model, FluxModelSpec) else FluxModelSpec.from_dict(self.model)
        object.__setattr__(self, "model", model)
        if self.lora is not None:
            lora = self.lora if isinstance(self.lora, LoraSpec) else LoraSpec.from_dict(self.lora)
            if lora.base_model != model.key:
                raise ContractError("Selected LoRA does not match the selected FLUX model size.")
            object.__setattr__(self, "lora", lora)
        object.__setattr__(
            self,
            "device_policy",
            _enum_value(DevicePolicy, self.device_policy, "device_policy"),
        )

    def validate_paths(self, temp_root: str | os.PathLike[str]) -> None:
        input_path = validate_local_path(
            self.input_path,
            "input_path",
            root=temp_root,
            suffix=".png",
        )
        output_path = validate_local_path(
            self.output_path,
            "output_path",
            root=temp_root,
            suffix=".png",
        )
        if input_path == output_path:
            raise ContractError("output_path must not overwrite input_path.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance": self.guidance,
            "seed": self.seed,
            "model": self.model.to_dict(),
            "lora": self.lora.to_dict() if self.lora is not None else None,
            "device_policy": self.device_policy.value,
        }

    def to_json(self) -> str:
        return encode_json_line(self.to_dict()).rstrip("\n")

    def to_command(self) -> dict[str, Any]:
        command = self.to_dict()
        command["id"] = command.pop("request_id")
        command["op"] = "generate"
        return command

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        temp_root: str | os.PathLike[str] | None = None,
    ) -> "ArtisticEditRequest":
        if not isinstance(data, Mapping):
            raise ContractError("Artistic edit request must be an object.")
        required = {
            "request_id", "input_path", "output_path", "prompt", "width", "height",
            "steps", "guidance", "seed", "model", "lora", "device_policy",
        }
        _require_keys(data, required)
        payload = dict(data)
        payload["model"] = FluxModelSpec.from_dict(payload["model"])
        if payload["lora"] is not None:
            payload["lora"] = LoraSpec.from_dict(payload["lora"])
        request = cls(**payload)
        if temp_root is not None:
            request.validate_paths(temp_root)
        return request

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        temp_root: str | os.PathLike[str] | None = None,
    ) -> "ArtisticEditRequest":
        return cls.from_dict(_load_json_object(text), temp_root=temp_root)


@dataclass(frozen=True)
class ArtisticEditResult:
    request_id: str
    output_path: str
    seed: int
    elapsed_seconds: float
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(
            self,
            "output_path",
            validate_local_path(self.output_path, "output_path", suffix=".png"),
        )
        object.__setattr__(
            self,
            "seed",
            _integer(self.seed, "seed", minimum=0, maximum=(2**63) - 1),
        )
        object.__setattr__(
            self,
            "elapsed_seconds",
            _finite_number(self.elapsed_seconds, "elapsed_seconds", minimum=0.0),
        )
        object.__setattr__(
            self,
            "width",
            _integer(self.width, "width", minimum=1, maximum=MAX_IMAGE_DIMENSION),
        )
        object.__setattr__(
            self,
            "height",
            _integer(self.height, "height", minimum=1, maximum=MAX_IMAGE_DIMENSION),
        )

    def validate_path(self, temp_root: str | os.PathLike[str]) -> None:
        validate_local_path(self.output_path, "output_path", root=temp_root, suffix=".png")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "output_path": self.output_path,
            "seed": self.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "width": self.width,
            "height": self.height,
        }

    def to_json(self) -> str:
        return encode_json_line(self.to_dict()).rstrip("\n")

    def to_event(self) -> dict[str, Any]:
        event = self.to_dict()
        event["id"] = event.pop("request_id")
        event["event"] = "result"
        return event

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        temp_root: str | os.PathLike[str] | None = None,
    ) -> "ArtisticEditResult":
        if not isinstance(data, Mapping):
            raise ContractError("Artistic edit result must be an object.")
        _require_keys(data, {"request_id", "output_path", "seed", "elapsed_seconds", "width", "height"})
        result = cls(**dict(data))
        if temp_root is not None:
            result.validate_path(temp_root)
        return result

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        temp_root: str | os.PathLike[str] | None = None,
    ) -> "ArtisticEditResult":
        return cls.from_dict(_load_json_object(text), temp_root=temp_root)
