"""Header-only checks for the model-only LoRA layouts used by ComfyUI."""

from __future__ import annotations

from pathlib import Path
import re

from .comfy_config import QWEN_IMAGE_21_MODEL_KEY
from .lora_registry import (
    LoraCompatibilityError,
    read_safetensors_header,
    validate_lora_compatibility,
)


_BLOCK = re.compile(r"transformer_blocks\.(0|[1-9]\d*)\.(.+)")
_SUFFIX = re.compile(r"(.+)\.lora_([AB])\.weight$")
_BLOCK_SHAPES = {
    "attn.to_q": (4096, 4096),
    "attn.to_k": (4096, 4096),
    "attn.to_v": (4096, 4096),
    "attn.to_out.0": (4096, 4096),
    "img_mlp.gate_up": (24576, 4096),
    "img_mlp.gate_layer": (12288, 4096),
    "img_mlp.proj": (12288, 4096),
    "img_mlp.out": (4096, 12288),
}
_GLOBAL_SHAPES = {
    "img_in": (4096, 64),
    "modulation.1": (16384, 4096),
    "norm_out.linear": (4096, 4096),
    "proj_out": (64, 4096),
    "time_text_embed.timestep_embedder.linear_1": (4096, 256),
    "time_text_embed.timestep_embedder.linear_2": (4096, 4096),
    "txt_in.in_layer": (4096, 4096),
    "txt_in.out_layer": (4096, 4096),
}


def _qwen_target(name: str) -> tuple[str, tuple[int, int], bool]:
    generic = name.startswith("diffusion_model.")
    target = name
    for prefix in ("diffusion_model.", "transformer."):
        if target.startswith(prefix):
            target = target[len(prefix):]
            break
    match = _BLOCK.fullmatch(target)
    if match:
        index, module = int(match[1]), match[2]
        shape = _BLOCK_SHAPES.get(module)
        if index >= 32 or shape is None:
            raise LoraCompatibilityError(f"Unsupported Qwen Image 2.1 LoRA target: {name}.")
        # Comfy's generic map exposes fused weights; its Qwen map exposes
        # split gate/proj aliases instead of a transformer-prefixed gate_up.
        if ((module == "img_mlp.gate_up" and not generic)
                or (module in {"img_mlp.gate_layer", "img_mlp.proj"} and generic)):
            raise LoraCompatibilityError(f"Unsupported Qwen Image 2.1 LoRA mapping: {name}.")
        return target, shape, module.startswith("img_mlp.")
    if target in _GLOBAL_SHAPES:
        return target, _GLOBAL_SHAPES[target], False
    raise LoraCompatibilityError(f"Unsupported Qwen Image 2.1 LoRA target: {name}.")


def validate_comfy_lora_compatibility(path: str | Path, model_key: str) -> None:
    """Qualify known Qwen 2.1 model-only layouts; retain Klein's existing checks.

    Names and matrix dimensions must agree for every adapter pair. Metadata or
    a filename alone cannot establish compatibility. This deliberately rejects
    attention-only (ambiguous), text-encoder, DoRA, and unknown adapter formats.
    It does not load weights or establish the adapter's visual quality.
    """
    if model_key != QWEN_IMAGE_21_MODEL_KEY:
        validate_lora_compatibility(path, model_key)
        return

    header = read_safetensors_header(path)
    pairs: dict[str, dict[str, tuple[int, ...]]] = {}
    alphas: set[str] = set()
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict) or metadata.get("dtype") not in {"F16", "BF16", "F32", "F64"}:
            raise LoraCompatibilityError(f"Invalid Qwen LoRA tensor metadata: {name}.")
        shape = metadata.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        ):
            raise LoraCompatibilityError(f"Invalid Qwen LoRA tensor shape: {name}.")
        match = _SUFFIX.fullmatch(name)
        if match:
            if len(shape) != 2:
                raise LoraCompatibilityError(f"Qwen LoRA matrices must have two dimensions: {name}.")
            pairs.setdefault(match[1], {})[match[2]] = tuple(shape)
        elif name.endswith(".alpha") and shape in ([], [1]):
            alphas.add(name[:-len(".alpha")])
        else:
            raise LoraCompatibilityError(f"Unsupported Qwen LoRA tensor: {name}.")

    if not pairs:
        raise LoraCompatibilityError("LoRA contains no recognizable Qwen Image 2.1 adapter pairs.")
    if alphas - pairs.keys():
        raise LoraCompatibilityError("Qwen LoRA alpha has no matching adapter pair.")
    targets: set[str] = set()
    has_qwen_feedforward = False
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise LoraCompatibilityError(f"Qwen LoRA requires a complete A/B pair: {name}.")
        target, (output_width, input_width), feedforward = _qwen_target(name)
        if target in targets:
            raise LoraCompatibilityError(f"Qwen LoRA targets the same weight more than once: {target}.")
        targets.add(target)
        down, up = pair["A"], pair["B"]
        if down[1] != input_width or up[0] != output_width or down[0] != up[1]:
            raise LoraCompatibilityError(f"LoRA dimensions do not match Qwen Image 2.1: {name}.")
        has_qwen_feedforward |= feedforward

    for target in targets:
        if target.endswith(".img_mlp.gate_up"):
            prefix = target[:-len("gate_up")]
            if prefix + "gate_layer" in targets or prefix + "proj" in targets:
                raise LoraCompatibilityError("Qwen LoRA mixes fused and split patches for the same weight.")
    if not has_qwen_feedforward:
        raise LoraCompatibilityError(
            "Could not verify this LoRA as Qwen Image 2.1: attention-only adapters are ambiguous."
        )
