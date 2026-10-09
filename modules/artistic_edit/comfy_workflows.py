"""Reviewed API graphs for a single Klein or Qwen image edit; no graph imports."""
from __future__ import annotations

import hashlib
import json

from .backend_contracts import ComfyGenerationSpec
from .comfy_config import QWEN_IMAGE_21_MODEL_KEY, WORKFLOW_ID, validate_server_name

FINAL_OUTPUT_NODE = "17"
WORKFLOW_VERSION = 1
REQUIRED_INPUTS = {
    "UNETLoader": {"unet_name", "weight_dtype"},
    "CLIPLoader": {"clip_name", "type"},
    "VAELoader": {"vae_name"},
    "LoadImage": {"image"},
    "CLIPTextEncode": {"clip", "text"},
    "VAEEncode": {"pixels", "vae"},
    "ReferenceLatent": {"conditioning", "latent"},
    "ConditioningZeroOut": {"conditioning"},
    "CFGGuider": {"model", "positive", "negative", "cfg"},
    "RandomNoise": {"noise_seed"},
    "KSamplerSelect": {"sampler_name"},
    "Flux2Scheduler": {"steps", "width", "height"},
    "EmptyFlux2LatentImage": {"width", "height", "batch_size"},
    "SamplerCustomAdvanced": {"noise", "guider", "sampler", "sigmas", "latent_image"},
    "VAEDecode": {"samples", "vae"},
    "SaveImage": {"images", "filename_prefix"},
}
QWEN_REQUIRED_INPUTS = {
    "UNETLoader": {"unet_name", "weight_dtype"},
    "CLIPLoader": {"clip_name", "type"},
    "VAELoader": {"vae_name"},
    "LoadImage": {"image"},
    "ImagePadForOutpaint": {"image", "left", "top", "right", "bottom", "feathering"},
    "TextEncodeQwenImage21": {"clip", "prompt", "negative_prompt", "vae", "resolution", "images"},
    "QwenImage21Cache": {"model", "device", "dtype"},
    "KSampler": {"model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image", "denoise"},
    "VAEDecode": {"samples", "vae"},
    "ImageCrop": {"image", "width", "height", "x", "y"},
    "SplitImageWithAlpha": {"image"},
    "InvertMask": {"mask"},
    "ImageCompositeMasked": {"destination", "source", "x", "y", "resize_source", "mask"},
    "SaveImage": {"images", "filename_prefix"},
}
# Klein templates: C0 package 0.1.95. Qwen template: installed package 0.1.99.
SOURCE_TEMPLATE_HASHES = {
    "klein-4b": "e0388a8870495802314d58fa61616ddcdb7064dac5f85a8787c9e08180b8a560",
    "klein-9b": "4e4285c1a5cba3df9cf022073425ec9585c418a8cb0b7570f936110105763599",
    QWEN_IMAGE_21_MODEL_KEY: "c1e6551759de131f083a040da072ac6f032c3c8401ceca202412cd912897d0fc",
}


def combo_values(input_schema) -> tuple[str, ...]:
    """Comfy legacy nodes and V3 nodes expose different combo schema forms."""
    if not isinstance(input_schema, (list, tuple)) or not input_schema:
        return ()
    values = input_schema[0]
    if values == "COMBO" and len(input_schema) > 1 and isinstance(input_schema[1], dict):
        values = input_schema[1].get("options")
    if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) for value in values):
        return ()
    return tuple(values)


def validate_node_schemas(nodes, profile, lora=None) -> None:
    is_qwen = profile.model_key == QWEN_IMAGE_21_MODEL_KEY
    required = dict(QWEN_REQUIRED_INPUTS if is_qwen else REQUIRED_INPUTS)
    if profile.loader == "gguf":
        required.pop("UNETLoader")
        required["UnetLoaderGGUF"] = {"unet_name"}
    if lora is not None:
        required["LoraLoaderModelOnly"] = {"model", "lora_name", "strength_model"}
    for class_type, fields in required.items():
        schema = nodes.get(class_type)
        if not isinstance(schema, dict):
            raise ValueError(f"ComfyUI node is unavailable: {class_type}.")
        inputs = schema.get("input", {})
        names = set(inputs.get("required", {})) | set(inputs.get("optional", {}))
        if not fields <= names:
            raise ValueError(f"ComfyUI node schema is incompatible: {class_type}.")
    clip_types = combo_values(nodes["CLIPLoader"]["input"]["required"]["type"])
    clip_type = "qwen_image" if is_qwen else "flux2"
    if clip_type not in clip_types:
        raise ValueError(f"ComfyUI does not support the {'Qwen Image 2.1' if is_qwen else 'Klein'} text encoder.")
    sampler_node = "KSampler" if is_qwen else "KSamplerSelect"
    if "euler" not in combo_values(nodes[sampler_node]["input"]["required"]["sampler_name"]):
        raise ValueError("ComfyUI does not provide the Euler sampler.")
    if is_qwen:
        if "simple" not in combo_values(nodes["KSampler"]["input"]["required"]["scheduler"]):
            raise ValueError("ComfyUI does not provide the Simple scheduler for Qwen Image 2.1.")
        inputs = nodes["TextEncodeQwenImage21"]["input"]
        images = (inputs.get("required", {}) | inputs.get("optional", {}))["images"]
        template = images[1].get("template") if isinstance(images, (list, tuple)) and len(images) > 1 and isinstance(images[1], dict) else None
        if (not isinstance(images, (list, tuple)) or len(images) < 2
                or images[0] != "COMFY_AUTOGROW_V3" or not isinstance(template, dict)
                or not isinstance(template.get("names"), (list, tuple)) or "image_1" not in template["names"]):
            raise ValueError("ComfyUI node schema is incompatible: TextEncodeQwenImage21 images.image_1.")
        for field, expected in (("device", "auto"), ("dtype", "default")):
            if expected not in combo_values(nodes["QwenImage21Cache"]["input"]["required"][field]):
                raise ValueError(f"ComfyUI node schema is incompatible: QwenImage21Cache {field}.")


def build_workflow(spec: ComfyGenerationSpec, image_name: str) -> dict:
    raster, profile = spec.raster, spec.profile
    image_name = validate_server_name(image_name)

    def node(class_type, **inputs):
        return {"class_type": class_type, "inputs": inputs}

    if profile.model_key == QWEN_IMAGE_21_MODEL_KEY:
        # resolution=0 still rounds to 32; padding avoids resampling the app's crop.
        graph = {
            "1": node("UNETLoader", unet_name=profile.diffusion_model, weight_dtype="default"),
            "2": node("CLIPLoader", clip_name=profile.text_encoder, type="qwen_image", device="default"),
            "3": node("VAELoader", vae_name=profile.vae),
            "4": node("LoadImage", image=image_name),
            "5": node("ImagePadForOutpaint", image=["4", 0], left=0, top=0,
                      right=(-raster.width) % 32, bottom=(-raster.height) % 32, feathering=0),
            "6": node("TextEncodeQwenImage21", clip=["2", 0], vae=["3", 0], prompt=raster.prompt,
                      negative_prompt="", resolution=0, **{"images.image_1": ["5", 0]}),
            "7": node("QwenImage21Cache", model=["1", 0], device="auto", dtype="default"),
            "8": node("KSampler", model=["7", 0], positive=["6", 0], negative=["6", 1], latent_image=["6", 2],
                      seed=raster.seed, steps=raster.steps, cfg=spec.cfg, sampler_name="euler", scheduler="simple", denoise=1.0),
            "9": node("VAEDecode", samples=["8", 0], vae=["3", 0]),
            "10": node("ImageCrop", image=["9", 0], width=raster.width, height=raster.height, x=0, y=0),
            # Comfy's split mask is 1-alpha. Preserve input under generated transparency.
            "11": node("SplitImageWithAlpha", image=["10", 0]),
            "12": node("InvertMask", mask=["11", 1]),
            "13": node("ImageCompositeMasked", destination=["4", 0], source=["11", 0],
                       mask=["12", 0], x=0, y=0, resize_source=False),
            FINAL_OUTPUT_NODE: node("SaveImage", images=["13", 0], filename_prefix=f"local-comic-translate/{raster.request_id}"),
        }
        if spec.lora is not None:
            graph["18"] = node("LoraLoaderModelOnly", model=["1", 0], lora_name=spec.lora.name,
                               strength_model=spec.lora.scale)
            # The cache clones its model patcher; apply the adapter before that clone.
            graph["7"]["inputs"]["model"] = ["18", 0]
        return graph

    graph = {
        "1": node("UNETLoader", unet_name=profile.diffusion_model, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=profile.text_encoder, type="flux2", device="default"),
        "3": node("VAELoader", vae_name=profile.vae),
        "4": node("LoadImage", image=image_name),
        "5": node("CLIPTextEncode", clip=["2", 0], text=raster.prompt),
        "6": node("VAEEncode", pixels=["4", 0], vae=["3", 0]),
        "7": node("ReferenceLatent", conditioning=["5", 0], latent=["6", 0]),
        "8": node("ConditioningZeroOut", conditioning=["5", 0]),
        "9": node("ReferenceLatent", conditioning=["8", 0], latent=["6", 0]),
        "10": node("CFGGuider", model=["1", 0], positive=["7", 0], negative=["9", 0], cfg=1.0),
        "11": node("RandomNoise", noise_seed=raster.seed),
        "12": node("KSamplerSelect", sampler_name="euler"),
        "13": node("Flux2Scheduler", steps=raster.steps, width=raster.width, height=raster.height),
        "14": node("EmptyFlux2LatentImage", width=raster.width, height=raster.height, batch_size=1),
        "15": node("SamplerCustomAdvanced", noise=["11", 0], guider=["10", 0], sampler=["12", 0], sigmas=["13", 0], latent_image=["14", 0]),
        "16": node("VAEDecode", samples=["15", 0], vae=["3", 0]),
        FINAL_OUTPUT_NODE: node("SaveImage", images=["16", 0], filename_prefix=f"local-comic-translate/{raster.request_id}"),
    }
    if profile.loader == "gguf":
        graph["1"] = node("UnetLoaderGGUF", unet_name=profile.diffusion_model)
    if spec.lora is not None:
        graph["18"] = node("LoraLoaderModelOnly", model=["1", 0], lora_name=spec.lora.name, strength_model=spec.lora.scale)
        graph["10"]["inputs"]["model"] = ["18", 0]
    return graph


def workflow_fingerprint(graph: dict) -> str:
    return hashlib.sha256(json.dumps(graph, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
