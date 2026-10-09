import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules.artistic_edit.comfy_config import QWEN_IMAGE_21_MODEL_KEY
from modules.artistic_edit.comfy_lora import validate_comfy_lora_compatibility
from modules.artistic_edit.lora_registry import LoraCompatibilityError


def adapter(target, input_width, output_width, rank=8):
    return {
        f"{target}.lora_A.weight": {"shape": [rank, input_width], "dtype": "BF16", "data_offsets": [0, 0]},
        f"{target}.lora_B.weight": {"shape": [output_width, rank], "dtype": "BF16", "data_offsets": [0, 0]},
    }


def fused_adapter():
    return adapter("diffusion_model.transformer_blocks.0.img_mlp.gate_up", 4096, 24576)


class ComfyQwenLoraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "adapter.safetensors"

    def validate(self, header):
        payload = json.dumps(header).encode("utf-8")
        self.path.write_bytes(len(payload).to_bytes(8, "little") + payload)
        validate_comfy_lora_compatibility(self.path, QWEN_IMAGE_21_MODEL_KEY)

    def test_generic_fused_layout_and_optional_alpha(self):
        header = fused_adapter()
        header.update(adapter("diffusion_model.transformer_blocks.31.attn.to_q", 4096, 4096, rank=32))
        header["diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha"] = {
            "shape": [], "dtype": "F32", "data_offsets": [0, 0],
        }
        self.validate(header)

    def test_split_layout_with_globals_and_variable_ranks(self):
        header = {}
        for module in ("gate_layer", "proj"):
            header.update(adapter(f"transformer.transformer_blocks.31.img_mlp.{module}", 4096, 12288, 256))
        header.update(adapter("transformer.transformer_blocks.0.img_mlp.out", 12288, 4096, 64))
        header.update(adapter("transformer.modulation.1", 4096, 16384, 256))
        header.update(adapter("transformer.time_text_embed.timestep_embedder.linear_1", 256, 4096, 256))
        self.validate(header)

    def test_unprefixed_split_layout(self):
        self.validate(adapter("transformer_blocks.0.img_mlp.proj", 4096, 12288))

    def test_klein_validation_is_delegated_unchanged(self):
        with patch("modules.artistic_edit.comfy_lora.validate_lora_compatibility") as validate:
            validate_comfy_lora_compatibility(self.path, "klein-9b")
        validate.assert_called_once_with(self.path, "klein-9b")

    def test_rejects_old_qwen_dimensions(self):
        header = fused_adapter() | adapter("transformer.transformer_blocks.0.attn.to_q", 3072, 3072)
        with self.assertRaisesRegex(LoraCompatibilityError, "dimensions"):
            self.validate(header)

    def test_rejects_old_qwen_and_klein_targets(self):
        for target in (
            "transformer.transformer_blocks.0.img_mlp.net.0.proj",
            "transformer.transformer_blocks.0.attn.add_q_proj",
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            "text_encoder.layers.0.self_attn.q_proj",
            "transformer.transformer_blocks.32.attn.to_q",
            "transformer.transformer_blocks.00.attn.to_q",
        ):
            with self.subTest(target=target), self.assertRaisesRegex(LoraCompatibilityError, "Unsupported"):
                self.validate(fused_adapter() | adapter(target, 4096, 4096))

    def test_rejects_attention_only_despite_metadata_claim(self):
        header = adapter("transformer.transformer_blocks.0.attn.to_q", 4096, 4096)
        header["__metadata__"] = {"ss_base_model_version": "qwen_image_2"}
        with self.assertRaisesRegex(LoraCompatibilityError, "ambiguous"):
            self.validate(header)

    def test_rejects_unsupported_fused_and_split_prefixes(self):
        for target, output_width in (
            ("transformer.transformer_blocks.0.img_mlp.gate_up", 24576),
            ("transformer_blocks.0.img_mlp.gate_up", 24576),
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_layer", 12288),
            ("diffusion_model.transformer_blocks.0.img_mlp.proj", 12288),
        ):
            with self.subTest(target=target), self.assertRaisesRegex(LoraCompatibilityError, "mapping"):
                self.validate(adapter(target, 4096, output_width))

    def test_rejects_missing_partner(self):
        for suffix in ("A", "B"):
            header = fused_adapter()
            header.pop(f"diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_{suffix}.weight")
            with self.subTest(suffix=suffix), self.assertRaisesRegex(LoraCompatibilityError, "complete A/B"):
                self.validate(header)

    def test_rejects_wrong_rank_or_target_dimensions(self):
        for shape in ([24576, 7], [12288, 8]):
            header = fused_adapter()
            header["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight"]["shape"] = shape
            with self.subTest(shape=shape), self.assertRaisesRegex(LoraCompatibilityError, "dimensions"):
                self.validate(header)

    def test_rejects_duplicate_and_overlapping_targets(self):
        duplicate = fused_adapter()
        duplicate.update(adapter("diffusion_model.transformer_blocks.0.attn.to_q", 4096, 4096))
        duplicate.update(adapter("transformer.transformer_blocks.0.attn.to_q", 4096, 4096))
        with self.assertRaisesRegex(LoraCompatibilityError, "more than once"):
            self.validate(duplicate)
        overlap = fused_adapter() | adapter("transformer.transformer_blocks.0.img_mlp.proj", 4096, 12288)
        with self.assertRaisesRegex(LoraCompatibilityError, "fused and split"):
            self.validate(overlap)

    def test_rejects_unsupported_tensor_payloads(self):
        for name, metadata in (
            ("unexpected", {"shape": [1], "dtype": "BF16"}),
            ("orphan.alpha", {"shape": [], "dtype": "F32"}),
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.dora_scale", {"shape": [24576], "dtype": "F32"}),
        ):
            with self.subTest(name=name), self.assertRaises(LoraCompatibilityError):
                self.validate(fused_adapter() | {name: metadata})

    def test_rejects_malformed_shapes_and_nonfloating_types(self):
        key = "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight"
        for metadata in (
            {"shape": [True, 4096], "dtype": "BF16"},
            {"shape": [0, 4096], "dtype": "BF16"},
            {"shape": [8, 4096, 1], "dtype": "BF16"},
            {"shape": [8, 4096], "dtype": "I8"},
            None,
        ):
            with self.subTest(metadata=metadata), self.assertRaises(LoraCompatibilityError):
                self.validate(fused_adapter() | {key: metadata})

    def test_rejects_empty_or_metadata_only_file(self):
        for header in ({}, {"__metadata__": {"base_model": "Qwen-Image-2.1"}}):
            with self.subTest(header=header), self.assertRaisesRegex(LoraCompatibilityError, "no recognizable"):
                self.validate(header)

    def test_header_only_reader_retains_truncation_and_size_guards(self):
        for content in (b"bad", (17 * 1024 * 1024).to_bytes(8, "little")):
            self.path.write_bytes(content)
            with self.subTest(content=content), self.assertRaises(LoraCompatibilityError):
                validate_comfy_lora_compatibility(self.path, QWEN_IMAGE_21_MODEL_KEY)


if __name__ == "__main__":
    unittest.main()
