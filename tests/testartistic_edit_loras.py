import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules.artistic_edit.contracts import ContractError, ModelKey
from modules.artistic_edit.lora_registry import (
    LoraCompatibilityError,
    LoraRegistry,
    validate_lora_compatibility,
)


class ArtisticEditLoraRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.four_b = self.root / "4b"
        self.nine_b = self.root / "9b"
        self.four_b.mkdir()
        self.nine_b.mkdir()
        self.registry = LoraRegistry(self.root)

    def write_weight(self, folder, name, content=b"weights"):
        path = folder / name
        path.write_bytes(content)
        return path

    def write_header_weight(self, folder, name, width, *, double_block=0, single_block=None):
        if single_block is None:
            tensor_name = (
                f"transformer.transformer_blocks.{double_block}.attn.to_q.lora_A.weight"
            )
        else:
            tensor_name = (
                "transformer.single_transformer_blocks."
                f"{single_block}.attn.to_qkv_mlp_proj.lora_A.weight"
            )
        header = json.dumps(
            {
                tensor_name: {
                    "dtype": "F32",
                    "shape": [8, width],
                    "data_offsets": [0, 0],
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return self.write_weight(
            folder,
            name,
            len(header).to_bytes(8, "little") + header,
        )

    def test_scan_is_model_specific_and_only_includes_safetensors(self):
        four = self.write_weight(self.four_b, "four.safetensors")
        nine = self.write_weight(self.nine_b, "nine.safetensors")
        self.write_weight(self.four_b, "ignored.pt")
        (self.four_b / "nested").mkdir()
        self.write_weight(self.four_b / "nested", "nested.safetensors")

        four_entries = self.registry.scan(ModelKey.KLEIN_4B)
        nine_entries = self.registry.scan(ModelKey.KLEIN_9B)
        self.assertEqual([entry.path for entry in four_entries], [str(four.resolve())])
        self.assertEqual([entry.path for entry in nine_entries], [str(nine.resolve())])
        self.assertTrue(all(entry.sha256 is None for entry in four_entries + nine_entries))

    def test_valid_sidecar_populates_display_metadata(self):
        self.write_weight(self.four_b, "ink.safetensors")
        (self.four_b / "ink.json").write_text(
            json.dumps(
                {
                    "display_name": "Ink Lettering",
                    "base_model": "klein-4b",
                    "trigger_words": ["inked title", "hand lettering"],
                    "default_scale": 0.65,
                    "description": "Keeps rough ink edges.",
                }
            ),
            encoding="utf-8",
        )
        entry = self.registry.scan(ModelKey.KLEIN_4B)[0]
        self.assertEqual(entry.display_name, "Ink Lettering")
        self.assertEqual(entry.trigger_words, ("inked title", "hand lettering"))
        self.assertEqual(entry.default_scale, 0.65)
        self.assertIsNone(entry.sidecar_error)

    def test_bad_or_mismatched_sidecar_falls_back_without_cross_population(self):
        self.write_weight(self.four_b, "wrong.safetensors")
        self.write_weight(self.four_b, "broken.safetensors")
        (self.four_b / "wrong.json").write_text(
            '{"display_name":"Wrong","base_model":"klein-9b"}',
            encoding="utf-8",
        )
        (self.four_b / "broken.json").write_text("{not-json", encoding="utf-8")
        entries = {entry.filename: entry for entry in self.registry.scan(ModelKey.KLEIN_4B)}
        for filename in ("wrong.safetensors", "broken.safetensors"):
            with self.subTest(filename=filename):
                entry = entries[filename]
                self.assertEqual(entry.display_name, Path(filename).stem)
                self.assertEqual(entry.base_model, ModelKey.KLEIN_4B)
                self.assertIsNotNone(entry.sidecar_error)
        self.assertEqual(self.registry.scan(ModelKey.KLEIN_9B), [])

    def test_checksum_is_lazy_cached_and_invalidated_by_file_metadata(self):
        path = self.write_weight(self.four_b, "hash.safetensors", b"first")
        entry = self.registry.scan(ModelKey.KLEIN_4B)[0]
        expected_first = hashlib.sha256(b"first").hexdigest()
        with patch.object(self.registry, "_hash_file", wraps=self.registry._hash_file) as hasher:
            self.assertEqual(self.registry.checksum(entry), expected_first)
            self.assertEqual(self.registry.checksum(entry), expected_first)
            self.assertEqual(hasher.call_count, 1)

            path.write_bytes(b"second-content")
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            expected_second = hashlib.sha256(b"second-content").hexdigest()
            self.assertEqual(self.registry.checksum(entry), expected_second)
            self.assertEqual(hasher.call_count, 2)

    def test_to_lora_spec_hashes_on_selection_and_validates_strength(self):
        path = self.write_weight(self.nine_b, "style.safetensors", b"nine")
        entry = self.registry.scan(ModelKey.KLEIN_9B)[0]
        spec = self.registry.to_lora_spec(entry, scale=1.25)
        self.assertEqual(spec.path, str(path.resolve()))
        self.assertEqual(spec.base_model, ModelKey.KLEIN_9B)
        self.assertEqual(spec.scale, 1.25)
        with self.assertRaisesRegex(ContractError, "between 0.0 and 2.0"):
            self.registry.to_lora_spec(entry, scale=3.0)

    def test_header_preflight_distinguishes_4b_and_9b_widths(self):
        four = self.write_header_weight(self.four_b, "four-header.safetensors", 3072)
        nine = self.write_header_weight(self.nine_b, "nine-header.safetensors", 4096)
        validate_lora_compatibility(four, ModelKey.KLEIN_4B)
        validate_lora_compatibility(nine, ModelKey.KLEIN_9B)
        with self.assertRaisesRegex(LoraCompatibilityError, "9B LoRA"):
            validate_lora_compatibility(nine, ModelKey.KLEIN_4B)
        with self.assertRaisesRegex(LoraCompatibilityError, "4B/FLUX.1"):
            validate_lora_compatibility(four, ModelKey.KLEIN_9B)

    def test_header_preflight_rejects_flux1_block_count_and_malformed_file(self):
        flux1 = self.write_header_weight(
            self.four_b,
            "flux1.safetensors",
            3072,
            double_block=18,
        )
        broken = self.write_weight(self.four_b, "broken-header.safetensors", b"bad")
        with self.assertRaisesRegex(LoraCompatibilityError, "likely FLUX.1"):
            validate_lora_compatibility(flux1, ModelKey.KLEIN_4B)
        with self.assertRaisesRegex(LoraCompatibilityError, "truncated"):
            validate_lora_compatibility(broken, ModelKey.KLEIN_4B)


if __name__ == "__main__":
    unittest.main()
