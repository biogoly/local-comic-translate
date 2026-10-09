"""Raster execution contracts independent of the existing Diffusers wire format."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .comfy_config import ComfyConnectionConfig, ComfyLoraSelection, ComfyModelProfile, ComfySettings, QWEN_IMAGE_21_MODEL_KEY
from .contracts import ArtisticEditResult, MAX_IMAGE_DIMENSION, MAX_PROMPT_LENGTH, validate_local_path, validate_request_id


@dataclass(frozen=True)
class RasterEditSpec:
    request_id: str
    input_path: str
    output_path: str
    prompt: str
    width: int
    height: int
    seed: int
    steps: int = 4

    def __post_init__(self):
        validate_request_id(self.request_id)
        for name in ("input_path", "output_path"):
            object.__setattr__(self, name, validate_local_path(getattr(self, name), name, suffix=".png"))
        if self.input_path == self.output_path or Path(self.input_path).parent != Path(self.output_path).parent:
            raise ValueError("ComfyUI input and output must be distinct PNGs in the request directory.")
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > MAX_PROMPT_LENGTH:
            raise ValueError("Enter an edit prompt within the supported length.")
        for dimension in (self.width, self.height):
            if isinstance(dimension, bool) or not isinstance(dimension, int) or not 16 <= dimension <= MAX_IMAGE_DIMENSION or dimension % 16:
                raise ValueError("Prepared image dimensions must be multiples of 16.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("Seed must be a nonnegative 63-bit integer.")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 100:
            raise ValueError("Steps must be between 1 and 100.")


@dataclass(frozen=True)
class ComfyGenerationSpec:
    raster: RasterEditSpec
    connection: ComfyConnectionConfig
    profile: ComfyModelProfile
    lora: ComfyLoraSelection | None
    job_token: str
    client_id: str
    cfg: float = 1.0

    def __post_init__(self):
        validate_request_id(self.job_token)
        validate_request_id(self.client_id)
        ComfySettings(self.connection, self.profile, self.lora)
        if (isinstance(self.cfg, bool) or not isinstance(self.cfg, (float, int))
                or not math.isfinite(self.cfg) or not 0 <= self.cfg <= 20):
            raise ValueError("ComfyUI CFG must be a finite number between 0 and 20.")
        if self.profile.model_key != QWEN_IMAGE_21_MODEL_KEY and self.cfg != 1.0:
            raise ValueError("Distilled Klein requires CFG 1.0.")


@dataclass(frozen=True)
class ComfyCompletion:
    result: ArtisticEditResult
    metadata: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ComfyJobEvent:
    request_id: str
    job_token: str
    kind: str
    payload: Any = None

    def __post_init__(self):
        if isinstance(self.payload, dict):
            object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
