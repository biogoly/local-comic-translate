from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from modules.artistic_edit.backend_contracts import ComfyGenerationSpec, RasterEditSpec
from modules.artistic_edit.comfy_config import ComfyConnectionConfig, ComfyLoraSelection, ComfyModelProfile
from modules.artistic_edit.comfy_workflows import REQUIRED_INPUTS, FINAL_OUTPUT_NODE, build_workflow, validate_node_schemas


def make_spec(root):
    return ComfyGenerationSpec(
        RasterEditSpec(str(uuid4()), str(root / "input.png"), str(root / "output.png"),
                       'Replace "text" with 日本語.\nKeep the artwork.', 272, 400, 2**63 - 1, 4),
        ComfyConnectionConfig(), ComfyModelProfile(), None, str(uuid4()), str(uuid4()),
    )


def fake_nodes(profile=None):
    profile = profile or ComfyModelProfile()
    nodes = {name: {"input": {"required": {field: ["ANY"] for field in fields}}}
             for name, fields in REQUIRED_INPUTS.items()}
    nodes["UNETLoader"]["input"]["required"]["unet_name"] = [[profile.diffusion_model]]
    nodes["CLIPLoader"]["input"]["required"].update(clip_name=[[profile.text_encoder]], type=[["flux2"]])
    nodes["VAELoader"]["input"]["required"]["vae_name"] = [[profile.vae]]
    nodes["KSamplerSelect"]["input"]["required"]["sampler_name"] = [["euler"]]
    nodes["LoraLoaderModelOnly"] = {"input": {"required": {"model": ["MODEL"], "lora_name": [["style.safetensors"]], "strength_model": ["FLOAT"]}}}
    return nodes


class ComfyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.spec = make_spec(Path(self.temp.name))

    def test_odd_multiple_of_16_dimensions_and_prompt_preserved(self):
        graph = build_workflow(self.spec, "uploads/input (1).png")
        self.assertEqual(graph["13"]["inputs"]["width"], 272)
        self.assertEqual(graph["14"]["inputs"]["height"], 400)
        self.assertEqual(graph["5"]["inputs"]["text"], self.spec.raster.prompt)
        self.assertEqual(graph["11"]["inputs"]["noise_seed"], 2**63 - 1)
        self.assertEqual(graph[FINAL_OUTPUT_NODE]["class_type"], "SaveImage")
        self.assertFalse(any(node["class_type"] == "ImageScaleToTotalPixels" for node in graph.values()))

    def test_reference_conditioning_and_sampler_match_distilled_template(self):
        graph = build_workflow(self.spec, "input.png")
        self.assertEqual(graph["10"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["7"]["inputs"]["latent"], ["6", 0])
        self.assertEqual(graph["9"]["inputs"]["latent"], ["6", 0])
        self.assertEqual(graph["16"]["inputs"]["samples"], ["15", 0])
        self.assertEqual(graph["12"]["inputs"]["sampler_name"], "euler")

    def test_lora_changes_and_removal_have_no_shared_graph_state(self):
        selected = replace(self.spec, lora=ComfyLoraSelection("style.safetensors", 0.7, "klein-9b"))
        graph = build_workflow(selected, "input.png")
        self.assertEqual(graph["10"]["inputs"]["model"], ["18", 0])
        self.assertEqual(graph["18"]["inputs"]["strength_model"], 0.7)
        base = build_workflow(self.spec, "input.png")
        self.assertNotIn("18", base)
        self.assertEqual(base["10"]["inputs"]["model"], ["1", 0])

    def test_node_and_schema_missing_fail_preflight(self):
        nodes = fake_nodes()
        validate_node_schemas(nodes, self.spec.profile)
        nodes["ReferenceLatent"]["input"]["required"].pop("latent")
        with self.assertRaisesRegex(ValueError, "ReferenceLatent"):
            validate_node_schemas(nodes, self.spec.profile)

    def test_v3_combo_schema_from_installed_037_runtime(self):
        nodes = fake_nodes()
        nodes["KSamplerSelect"]["input"]["required"]["sampler_name"] = ["COMBO", {"multiselect": False, "options": ["euler", "heun"]}]
        validate_node_schemas(nodes, self.spec.profile)
        nodes["KSamplerSelect"]["input"]["required"]["sampler_name"][1]["options"] = ["heun"]
        with self.assertRaisesRegex(ValueError, "Euler"):
            validate_node_schemas(nodes, self.spec.profile)

    def test_gguf_requires_specific_loader_schema(self):
        profile = replace(self.spec.profile, loader="gguf", diffusion_model="klein-Q8_0.gguf")
        with self.assertRaisesRegex(ValueError, "UnetLoaderGGUF"):
            validate_node_schemas(fake_nodes(), profile)
        graph = build_workflow(replace(self.spec, profile=profile), "input.png")
        self.assertEqual(graph["1"], {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "klein-Q8_0.gguf"}})

    def test_invalid_geometry_and_output_path_fail_before_transfer(self):
        with self.assertRaises(ValueError):
            replace(self.spec.raster, width=273)
        with self.assertRaises(ValueError):
            replace(self.spec.raster, output_path=self.spec.raster.input_path)
        with self.assertRaises(ValueError):
            replace(self.spec.raster, output_path=str(Path(self.temp.name).parent / "output.png"))


if __name__ == "__main__":
    unittest.main()
