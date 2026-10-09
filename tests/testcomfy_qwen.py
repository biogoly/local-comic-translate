from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

from PIL import Image
import numpy as np
import requests

from app.projects.patch_metadata import MAX_METADATA_KEYS, sanitize_patch_metadata
from modules.artistic_edit.comfy_client import ComfyCancelled, generate_comfy, probe_comfy
from modules.artistic_edit.comfy_config import (
    ComfyLoraSelection, ComfyModelProfile, ComfySettings, QWEN_IMAGE_21_INT8_PROFILE,
    QWEN_IMAGE_21_MODEL_KEY, QWEN_IMAGE_21_WORKFLOW_ID, load_comfy_settings, save_comfy_settings,
)
from modules.artistic_edit.comfy_workflows import (
    FINAL_OUTPUT_NODE, QWEN_REQUIRED_INPUTS, build_workflow, validate_node_schemas,
)
from tests.testcomfy_client import FakeServer
from tests.testcomfy_config import MemorySettings
from tests.testcomfy_workflows import make_spec


def qwen_nodes(profile=QWEN_IMAGE_21_INT8_PROFILE):
    nodes = {name: {"input": {"required": {field: ["ANY"] for field in fields}}}
             for name, fields in QWEN_REQUIRED_INPUTS.items()}
    nodes["UNETLoader"]["input"]["required"]["unet_name"] = [[profile.diffusion_model]]
    nodes["CLIPLoader"]["input"]["required"].update(clip_name=[[profile.text_encoder]], type=[["qwen_image"]])
    nodes["VAELoader"]["input"]["required"]["vae_name"] = [[profile.vae]]
    nodes["KSampler"]["input"]["required"].update(sampler_name=[["euler"]], scheduler=[["simple"]])
    nodes["QwenImage21Cache"]["input"]["required"].update(
        device=["COMBO", {"options": ["auto", "gpu", "cpu", "off"]}],
        dtype=["COMBO", {"options": ["default", "int8", "int4"]}],
    )
    encoder = nodes["TextEncodeQwenImage21"]["input"]
    encoder["optional"] = {"vae": encoder["required"].pop("vae")}
    encoder["required"]["images"] = ["COMFY_AUTOGROW_V3", {
        "template": {"input": {"required": {"image": ["IMAGE"]}}, "names": ["image_1", "image_2"], "min": 0},
    }]
    return nodes


class QwenConfigTests(unittest.TestCase):
    def test_native_profile_round_trip_keeps_family_and_files(self):
        profile = QWEN_IMAGE_21_INT8_PROFILE
        self.assertEqual(profile.model_key, QWEN_IMAGE_21_MODEL_KEY)
        self.assertEqual(profile.workflow_id, QWEN_IMAGE_21_WORKFLOW_ID)
        self.assertEqual(profile.text_encoder, "qwen3vl_8b_int8_convrot.safetensors")
        self.assertTrue(profile.experimental)
        self.assertEqual((profile.default_steps, profile.default_cfg), (25, 1.0))
        self.assertEqual((ComfyModelProfile().default_steps, ComfyModelProfile().default_cfg), (4, 1.0))
        settings = MemorySettings()
        config = ComfySettings(profile=profile)
        save_comfy_settings(settings, config)
        self.assertEqual(load_comfy_settings(settings), config)

    def test_cross_family_workflows_and_qwen_gguf_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "workflow"):
            replace(QWEN_IMAGE_21_INT8_PROFILE, workflow_id=ComfyModelProfile().workflow_id)
        with self.assertRaisesRegex(ValueError, "workflow"):
            replace(ComfyModelProfile(), workflow_id=QWEN_IMAGE_21_WORKFLOW_ID)
        with self.assertRaisesRegex(ValueError, "native checkpoint"):
            replace(QWEN_IMAGE_21_INT8_PROFILE, loader="gguf", diffusion_model="qwen.gguf")

    def test_existing_klein_lora_cannot_enter_qwen_settings(self):
        lora = ComfyLoraSelection("style.safetensors", 1.0, "klein-9b")
        with self.assertRaisesRegex(ValueError, "model family"):
            ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE, lora=lora)

    def test_qwen_lora_settings_round_trip_and_cannot_enter_klein_profile(self):
        lora = ComfyLoraSelection("QwenImage2.1/style.safetensors", 0.4, "qwen-image-2.1", "a" * 64)
        config = ComfySettings(profile=QWEN_IMAGE_21_INT8_PROFILE, lora=lora)
        settings = MemorySettings()
        save_comfy_settings(settings, config)
        self.assertEqual(load_comfy_settings(settings), config)
        with self.assertRaisesRegex(ValueError, "model family"):
            ComfySettings(lora=lora)


class QwenWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = make_spec(Path(self.temp.name))
        self.spec = replace(base, profile=QWEN_IMAGE_21_INT8_PROFILE,
                            raster=replace(base.raster, steps=25))

    def test_cfg_is_frozen_and_family_specific(self):
        for cfg in (False, True, "1", None, float("nan"), float("inf"), -0.01, 20.01):
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, "CFG"):
                replace(self.spec, cfg=cfg)
        for cfg in (0, 1, 2.5, 20):
            with self.subTest(cfg=cfg):
                self.assertEqual(replace(self.spec, cfg=cfg).cfg, cfg)
        with self.assertRaisesRegex(ValueError, "Klein requires CFG 1.0"):
            replace(self.spec, profile=ComfyModelProfile(), cfg=2.0)

    def test_qwen_lora_patches_model_before_cache_and_off_has_no_adapter(self):
        lora = ComfyLoraSelection("QwenImage2.1/style.safetensors", 0.4, "qwen-image-2.1")
        graph = build_workflow(replace(self.spec, lora=lora), "input.png")
        self.assertEqual(graph["18"], {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": lora.name, "strength_model": 0.4,
        }})
        self.assertEqual(graph["7"]["inputs"]["model"], ["18", 0])
        self.assertEqual(graph["8"]["inputs"]["model"], ["7", 0])
        self.assertEqual(graph["6"]["inputs"]["clip"], ["2", 0])
        disabled = build_workflow(self.spec, "input.png")
        self.assertNotIn("18", disabled)
        self.assertEqual(disabled["7"]["inputs"]["model"], ["1", 0])

    def test_qwen_lora_node_required_only_when_adapter_selected(self):
        lora = ComfyLoraSelection("style.safetensors", 1.0, "qwen-image-2.1")
        nodes = qwen_nodes()
        validate_node_schemas(nodes, self.spec.profile)
        with self.assertRaisesRegex(ValueError, "LoraLoaderModelOnly"):
            validate_node_schemas(nodes, self.spec.profile, lora)
        nodes["LoraLoaderModelOnly"] = {"input": {"required": {
            "model": ["MODEL"], "lora_name": [[lora.name]], "strength_model": ["FLOAT"],
        }}}
        validate_node_schemas(nodes, self.spec.profile, lora)

    def test_prepared_geometry_is_padded_without_resizing_and_restored(self):
        for width, height in ((16, 16), (272, 400), (1024, 784), (2048, 2048)):
            with self.subTest(width=width, height=height):
                spec = replace(self.spec, raster=replace(self.spec.raster, width=width, height=height))
                graph = build_workflow(spec, "input.png")
                by_type = {entry["class_type"]: entry["inputs"] for entry in graph.values()}
                pad = by_type["ImagePadForOutpaint"]
                crop = by_type["ImageCrop"]
                self.assertEqual((width + pad["right"]) % 32, 0)
                self.assertEqual((height + pad["bottom"]) % 32, 0)
                self.assertLess(pad["right"], 32)
                self.assertLess(pad["bottom"], 32)
                self.assertEqual((pad["left"], pad["top"], pad["feathering"]), (0, 0, 0))
                self.assertEqual((crop["width"], crop["height"], crop["x"], crop["y"]), (width, height, 0, 0))
                self.assertEqual(by_type["TextEncodeQwenImage21"]["resolution"], 0)
                self.assertFalse(any("Scale" in name or "Resize" in name for name in by_type))

    def test_single_reference_owns_canvas_and_sampler_conditioning(self):
        graph = build_workflow(replace(self.spec, cfg=2.5), "uploads/input (1).png")
        encode = graph["6"]["inputs"]
        sampler = graph["8"]["inputs"]
        self.assertEqual(graph["2"]["inputs"]["type"], "qwen_image")
        self.assertEqual(graph["4"]["inputs"]["image"], "uploads/input (1).png")
        self.assertEqual(encode["prompt"], self.spec.raster.prompt)
        self.assertEqual(encode["negative_prompt"], "")
        self.assertEqual({key for key in encode if key.startswith("images.")}, {"images.image_1"})
        self.assertEqual((sampler["positive"], sampler["negative"], sampler["latent_image"]),
                         (["6", 0], ["6", 1], ["6", 2]))
        self.assertEqual((sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]),
                         (25, 2.5, "euler", "simple", 1.0))
        self.assertEqual(sampler["seed"], self.spec.raster.seed)
        self.assertEqual(graph[FINAL_OUTPUT_NODE]["inputs"]["images"], ["13", 0])
        self.assertEqual(graph["11"]["class_type"], "SplitImageWithAlpha")
        self.assertEqual(graph["11"]["inputs"]["image"], ["10", 0])
        self.assertEqual(graph["12"], {"class_type": "InvertMask", "inputs": {"mask": ["11", 1]}})
        self.assertEqual(graph["13"], {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": ["4", 0], "source": ["11", 0], "mask": ["12", 0],
            "x": 0, "y": 0, "resize_source": False,
        }})
        self.assertFalse(any(entry["class_type"] == "ReferenceLatent" for entry in graph.values()))

    def test_qwen_preflight_requires_its_nodes_without_klein_nodes(self):
        nodes = qwen_nodes()
        validate_node_schemas(nodes, self.spec.profile)
        for node in ("TextEncodeQwenImage21", "QwenImage21Cache", "ImagePadForOutpaint", "ImageCrop",
                     "SplitImageWithAlpha", "InvertMask", "ImageCompositeMasked"):
            with self.subTest(node=node), self.assertRaisesRegex(ValueError, node):
                validate_node_schemas({key: value for key, value in nodes.items() if key != node}, self.spec.profile)

    def test_output_graph_preserves_transparent_input_and_accepts_opaque_rgb(self):
        spec = replace(self.spec, raster=replace(self.spec.raster, width=16, height=16))
        graph = build_workflow(spec, "input.png")
        original = np.full((16, 16, 3), (0.2, 0.4, 0.6), dtype=np.float32)
        generated_rgb = np.full((32, 32, 3), (0.8, 0.1, 0.7), dtype=np.float32)

        # Small output-only evaluator follows the inspected core node contracts;
        # graph references decide which image and mask reach SaveImage.
        def saved_pixels(decoded):
            cache = {"4": (original,), "9": (decoded,)}

            def output(link):
                node_id, slot = link
                if node_id not in cache:
                    entry = graph[node_id]
                    inputs = entry["inputs"]
                    kind = entry["class_type"]
                    if kind == "ImageCrop":
                        image = output(inputs["image"])
                        x, y = inputs["x"], inputs["y"]
                        result = (image[y:y + inputs["height"], x:x + inputs["width"]],)
                    elif kind == "SplitImageWithAlpha":
                        image = output(inputs["image"])
                        alpha = image[:, :, 3] if image.shape[2] == 4 else np.ones(image.shape[:2])
                        result = (image[:, :, :3], 1.0 - alpha)
                    elif kind == "InvertMask":
                        result = (1.0 - output(inputs["mask"]),)
                    elif kind == "ImageCompositeMasked":
                        self.assertEqual((inputs["x"], inputs["y"], inputs["resize_source"]), (0, 0, False))
                        alpha = output(inputs["mask"])[:, :, None]
                        result = (output(inputs["source"]) * alpha + output(inputs["destination"]) * (1.0 - alpha),)
                    else:
                        self.fail(f"Unexpected output operation: {kind}")
                    cache[node_id] = result
                return cache[node_id][slot]

            return output(graph[FINAL_OUTPUT_NODE]["inputs"]["images"])

        transparent = np.dstack((generated_rgb, np.zeros((32, 32), dtype=np.float32)))
        np.testing.assert_array_equal(saved_pixels(transparent), original)
        np.testing.assert_array_equal(saved_pixels(generated_rgb), generated_rgb[:16, :16])
        translucent = np.dstack((generated_rgb, np.full((32, 32), 0.5, dtype=np.float32)))
        np.testing.assert_allclose(saved_pixels(translucent), 0.5 * original + 0.5 * generated_rgb[:16, :16])

    def test_preflight_rejects_incompatible_encoder_autogrow_and_options(self):
        cases = (("CLIPLoader", "type", [["flux2"]], "text encoder"),
                 ("KSampler", "sampler_name", [["heun"]], "Euler"),
                 ("KSampler", "scheduler", [["normal"]], "Simple"),
                 ("QwenImage21Cache", "device", [["gpu"]], "QwenImage21Cache"),
                 ("TextEncodeQwenImage21", "images", ["COMFY_AUTOGROW_V3", {"template": {"names": ["wrong"]}}], "images.image_1"))
        for node, field, value, message in cases:
            with self.subTest(node=node, field=field):
                nodes = qwen_nodes()
                nodes[node]["input"]["required"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_node_schemas(nodes, self.spec.profile)

    def test_client_submits_once_and_records_actual_qwen_parameters(self):
        spec = replace(self.spec, cfg=2.5)
        Image.new("RGB", (spec.raster.width, spec.raster.height), "white").save(spec.raster.input_path)
        http = FakeServer(spec)
        http.nodes = qwen_nodes()
        result = generate_comfy(spec, threading.Event(), http=http, poll_interval=0)
        posts = [kwargs["json"]["prompt"] for method, path, kwargs in http.calls if path == "/prompt"]
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["8"]["inputs"]["cfg"], 2.5)
        self.assertEqual(result.metadata["workflow_id"], QWEN_IMAGE_21_WORKFLOW_ID)
        self.assertEqual((result.metadata["guidance"], result.metadata["steps"], result.metadata["scheduler"]), (2.5, 25, "simple"))
        self.assertEqual((result.metadata["width"], result.metadata["height"]), (272, 400))
        self.assertEqual((result.metadata["sampling_width"], result.metadata["sampling_height"]), (288, 416))
        self.assertEqual(result.metadata["reference_resolution"], 0)
        self.assertEqual(result.metadata["output_alpha"], "composited_over_input")
        # These app-owned fields are merged with execution provenance when Apply creates a patch.
        metadata = dict(result.metadata, prompt=spec.raster.prompt, cuda_device_index=None,
                        selection_mode="rectangle", source_crop_sha256="a" * 64, working_scale=1.0)
        self.assertLessEqual(len(metadata), MAX_METADATA_KEYS)
        self.assertEqual(sanitize_patch_metadata(metadata), metadata)

    def test_client_resolves_qwen_adapter_catalog_and_persists_provenance(self):
        lora = ComfyLoraSelection("QwenImage2.1/style.safetensors", 0.4, "qwen-image-2.1", "a" * 64)
        spec = replace(self.spec, lora=lora)
        Image.new("RGB", (spec.raster.width, spec.raster.height), "white").save(spec.raster.input_path)
        http = FakeServer(spec)
        http.nodes = qwen_nodes()
        registered = "QwenImage2.1\\style.safetensors"
        http.nodes["LoraLoaderModelOnly"] = {"input": {"required": {
            "model": ["MODEL"], "lora_name": [[registered]], "strength_model": ["FLOAT"],
        }}}
        result = generate_comfy(spec, threading.Event(), http=http, poll_interval=0)
        posted = [kwargs["json"]["prompt"] for method, path, kwargs in http.calls if path == "/prompt"]
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0]["18"]["inputs"]["lora_name"], registered)
        self.assertEqual(posted[0]["18"]["inputs"]["strength_model"], 0.4)
        self.assertEqual((result.metadata["lora_name"], result.metadata["lora_scale"], result.metadata["lora_sha256"]),
                         (lora.name, lora.scale, lora.sha256))
        with Image.open(result.result.output_path) as output:
            self.assertEqual((output.mode, output.size), ("RGB", (272, 400)))

    def test_missing_qwen_node_fails_before_any_upload(self):
        http = FakeServer(self.spec)
        http.nodes = qwen_nodes()
        del http.nodes["TextEncodeQwenImage21"]
        with self.assertRaisesRegex(ValueError, "TextEncodeQwenImage21"):
            generate_comfy(self.spec, threading.Event(), http=http, poll_interval=0)
        self.assertFalse(any(method == "POST" for method, path, kwargs in http.calls))

    def test_only_explicitly_qualified_runtime_versions_enable_targeted_cancel(self):
        for version, expected in (("0.37.0", True), ("0.38.0", True), ("0.36.0", False),
                                  ("0.38.1", False), ("0.39.0", False), ("unknown", False)):
            with self.subTest(version=version):
                http = FakeServer(self.spec)
                http.version = version
                info = probe_comfy(self.spec.connection, http=http)
                self.assertIs(info.targeted_cancel, expected)

    def test_038_cancel_targets_only_our_qwen_job(self):
        Image.new("RGB", (self.spec.raster.width, self.spec.raster.height), "white").save(self.spec.raster.input_path)
        http = FakeServer(self.spec)
        http.nodes = qwen_nodes()
        http.version = "0.38.0"
        http.running = True
        cancelled = threading.Event()
        http.on_prompt = cancelled.set
        with self.assertRaises(ComfyCancelled) as caught:
            generate_comfy(self.spec, cancelled, http=http, poll_interval=0)
        self.assertFalse(caught.exception.server_may_continue)
        cancel_posts = [path for method, path, kwargs in http.calls if method == "POST" and path.endswith("/cancel")]
        self.assertEqual(cancel_posts, [f"/api/jobs/{http.id}/cancel"])
        self.assertNotIn(http.foreign_id, cancel_posts[0])
        self.assertFalse(any(path in {"/interrupt", "/queue", "/free"} for method, path, kwargs in http.calls if method == "POST"))
        self.assertFalse(Path(self.spec.raster.output_path).exists())

    def test_038_accepted_ambiguous_submission_recovers_same_prompt_id(self):
        Image.new("RGB", (self.spec.raster.width, self.spec.raster.height), "white").save(self.spec.raster.input_path)
        http = FakeServer(self.spec)
        http.nodes = qwen_nodes()
        http.version = "0.38.0"
        http.post_error = requests.Timeout()
        http.recover_submission = True
        submitted = []
        generate_comfy(self.spec, threading.Event(), http=http, poll_interval=0, on_submitted=submitted.append)
        self.assertEqual(submitted, [self.spec.raster.request_id])
        self.assertEqual(sum(path == "/prompt" for method, path, kwargs in http.calls), 1)


if __name__ == "__main__":
    unittest.main()
