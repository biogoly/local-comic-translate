"""Immutable, lightweight settings for the optional local ComfyUI backend."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ipaddress
import json
import math
from pathlib import PurePosixPath
import re
from urllib.parse import urlsplit, urlunsplit

SETTINGS_KEY = "artistic_edit/comfyui/settings"
CONFIGURED_KEY = "artistic_edit/comfyui/configured"
WORKFLOW_ID = "klein-distilled-v1"
QWEN_IMAGE_21_MODEL_KEY = "qwen-image-2.1"
QWEN_IMAGE_21_WORKFLOW_ID = "qwen-image-2.1-edit-v1"


def normalize_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except (AttributeError, ValueError) as exc:
        raise ValueError("Enter a valid local ComfyUI URL.") from exc
    if (parsed.scheme not in {"http", "https"} or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"} or not parsed.hostname):
        raise ValueError("ComfyUI requires a loopback HTTP URL without credentials or a path.")
    host = parsed.hostname
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Only loopback ComfyUI servers are supported.") from exc
    if not address.is_loopback or "%" in host:
        raise ValueError("Only loopback ComfyUI servers are supported.")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("ComfyUI port must be between 1 and 65535.")
    authority = f"[{address}]" if address.version == 6 else str(address)
    if port is not None:
        authority += f":{port}"
    return urlunsplit((parsed.scheme, authority, "", "", ""))


def validate_server_name(value: str, *, allow_subfolders: bool = True) -> str:
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise ValueError("ComfyUI resource name is empty or too long.")
    normalized = value.replace("\\", "/")
    if (normalized.startswith("/") or ":" in normalized
            or any(ord(char) < 32 for char in normalized)
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
            or (not allow_subfolders and "/" in normalized)):
        raise ValueError("ComfyUI resource name must be a relative catalog name.")
    if PurePosixPath(normalized).is_absolute():
        raise ValueError("ComfyUI resource name must be relative.")
    return normalized


@dataclass(frozen=True)
class ComfyConnectionConfig:
    mode: str = "connect"
    base_url: str = "http://127.0.0.1:8188"
    installation_root: str = ""
    python_executable: str | None = None
    gpu_index: int = 0
    memory_policy: str = "normal"

    def __post_init__(self):
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        if not isinstance(self.installation_root, str) or "\0" in self.installation_root:
            raise ValueError("ComfyUI installation must be a local directory path.")
        if self.python_executable is not None and (not isinstance(self.python_executable, str) or "\0" in self.python_executable):
            raise ValueError("ComfyUI Python must be a local executable path.")
        if self.mode not in {"connect", "managed"}:
            raise ValueError("Unknown ComfyUI connection mode.")
        if self.memory_policy not in {"normal", "low_cache"}:
            raise ValueError("Unknown ComfyUI memory policy.")
        if isinstance(self.gpu_index, bool) or not isinstance(self.gpu_index, int) or self.gpu_index < 0:
            raise ValueError("ComfyUI GPU index must be a nonnegative integer.")
        if self.mode == "managed" and not self.installation_root.strip():
            raise ValueError("Select an existing ComfyUI installation.")


@dataclass(frozen=True)
class ComfyModelProfile:
    profile_id: str = "klein-9b-native"
    model_key: str = "klein-9b"
    diffusion_model: str = "flux-2-klein-9b-fp8.safetensors"
    text_encoder: str = "qwen_3_8b_fp8mixed.safetensors"
    vae: str = "full_encoder_small_decoder.safetensors"
    loader: str = "native"
    workflow_id: str = WORKFLOW_ID

    def __post_init__(self):
        if self.model_key not in {"klein-4b", "klein-9b", QWEN_IMAGE_21_MODEL_KEY}:
            raise ValueError("Only distilled Klein 4B/9B and Qwen Image 2.1 profiles are supported.")
        expected_workflow = QWEN_IMAGE_21_WORKFLOW_ID if self.model_key == QWEN_IMAGE_21_MODEL_KEY else WORKFLOW_ID
        if self.loader not in {"native", "gguf"} or self.workflow_id != expected_workflow:
            raise ValueError("Unsupported ComfyUI workflow or loader.")
        if self.model_key == QWEN_IMAGE_21_MODEL_KEY and self.loader != "native":
            raise ValueError("Qwen Image 2.1 currently requires the native checkpoint loader.")
        if not isinstance(self.profile_id, str) or not self.profile_id or len(self.profile_id) > 128:
            raise ValueError("Invalid ComfyUI profile ID.")
        for name in ("diffusion_model", "text_encoder", "vae"):
            object.__setattr__(self, name, validate_server_name(getattr(self, name)))
        suffix = ".gguf" if self.loader == "gguf" else ".safetensors"
        if not self.diffusion_model.lower().endswith(suffix):
            raise ValueError("Checkpoint format does not match the selected loader.")

    @property
    def experimental(self) -> bool:
        return self.loader == "gguf" or self.model_key in {"klein-4b", QWEN_IMAGE_21_MODEL_KEY}

    @property
    def default_steps(self) -> int:
        return 25 if self.model_key == QWEN_IMAGE_21_MODEL_KEY else 4

    @property
    def default_cfg(self) -> float:
        return 1.0


# This selects the existing native files qualified in COMFYUI_RUNTIME_VALIDATION.
# It is not a dtype override or an instruction to quantize/download weights.
KLEIN_9B_INT8_PROFILE = ComfyModelProfile(
    diffusion_model="flux2Klein9BINT8_9b.safetensors",
    text_encoder="qwen_3_8b_fp8mixed.safetensors",
    vae="flux2-vae.safetensors",
)

QWEN_IMAGE_21_INT8_PROFILE = ComfyModelProfile(
    profile_id="qwen-image-2.1-native",
    model_key=QWEN_IMAGE_21_MODEL_KEY,
    diffusion_model="qwen_image_2.1_int8_convrot.safetensors",
    text_encoder="qwen3vl_8b_int8_convrot.safetensors",
    vae="qwen_image_2.1_vae_bf16.safetensors",
    workflow_id=QWEN_IMAGE_21_WORKFLOW_ID,
)


@dataclass(frozen=True)
class ComfyLoraSelection:
    name: str
    scale: float
    verified_model_key: str
    sha256: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "name", validate_server_name(self.name))
        if not self.name.lower().endswith(".safetensors"):
            raise ValueError("A ComfyUI LoRA must be a safetensors file.")
        if self.verified_model_key not in {"klein-4b", "klein-9b", QWEN_IMAGE_21_MODEL_KEY}:
            raise ValueError("Verify the LoRA's model family and size before selecting it.")
        if (isinstance(self.scale, bool) or not isinstance(self.scale, (float, int))
                or not math.isfinite(self.scale) or not 0 <= self.scale <= 2):
            raise ValueError("LoRA strength must be between 0 and 2.")
        if self.sha256 is not None and (not isinstance(self.sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", self.sha256)):
            raise ValueError("Invalid LoRA SHA-256 fingerprint.")


@dataclass(frozen=True)
class ComfySettings:
    connection: ComfyConnectionConfig = field(default_factory=ComfyConnectionConfig)
    profile: ComfyModelProfile = field(default_factory=ComfyModelProfile)
    lora: ComfyLoraSelection | None = None

    def __post_init__(self):
        if self.lora is not None:
            if self.lora.verified_model_key != self.profile.model_key:
                raise ValueError("The LoRA does not match the selected model family or size.")
            if self.profile.loader == "gguf":
                raise ValueError("GGUF LoRA loading has not been qualified for this backend.")


def load_comfy_settings(settings) -> ComfySettings:
    raw = settings.value(SETTINGS_KEY, "")
    if not raw:
        return ComfySettings()
    try:
        data = json.loads(str(raw))
        return ComfySettings(
            connection=ComfyConnectionConfig(**data["connection"]),
            profile=ComfyModelProfile(**data["profile"]),
            lora=ComfyLoraSelection(**data["lora"]) if data.get("lora") else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Saved ComfyUI settings are invalid; open Configure ComfyUI.") from exc


def save_comfy_settings(settings, config: ComfySettings) -> None:
    settings.setValue(SETTINGS_KEY, json.dumps(asdict(config), ensure_ascii=False, allow_nan=False))
    settings.setValue(CONFIGURED_KEY, True)
    settings.sync()
