import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from modules.artistic_edit.contracts import (
    ArtisticEditRequest,
    ArtisticEditResult,
    ContractError,
    DevicePolicy,
    FluxModelSpec,
    LoraSpec,
    ModelKey,
    encode_json_line,
    message_matches_request,
    parse_protocol_line,
)


class ArtisticEditContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp_root = Path(self.temporary.name)
        self.request_id = str(uuid4())
        self.input_path = self.temp_root / "input.png"
        self.output_path = self.temp_root / "output.png"
        self.lora_path = self.temp_root / "lettering.safetensors"
        self.model = FluxModelSpec(ModelKey.KLEIN_4B, "local/model", "Apache 2.0")
        self.lora = LoraSpec(
            str(self.lora_path),
            ModelKey.KLEIN_4B,
            0.75,
            "a" * 64,
        )

    def request(self, **overrides):
        values = {
            "request_id": self.request_id,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "prompt": "Replace the title\nwhile preserving its painted style.",
            "width": 1024,
            "height": 768,
            "steps": 20,
            "guidance": 3.5,
            "seed": 42,
            "model": self.model,
            "lora": self.lora,
            "device_policy": DevicePolicy.MODEL_CPU_OFFLOAD,
        }
        values.update(overrides)
        return ArtisticEditRequest(**values)

    def test_request_round_trip_preserves_multiline_prompt(self):
        request = self.request()
        restored = ArtisticEditRequest.from_json(
            request.to_json(),
            temp_root=self.temp_root,
        )
        self.assertEqual(restored, request)
        self.assertIn("\n", restored.prompt)
        command = request.to_command()
        self.assertEqual(command["id"], self.request_id)
        self.assertEqual(command["op"], "generate")
        self.assertNotIn("request_id", command)

    def test_result_round_trip_and_event(self):
        result = ArtisticEditResult(
            self.request_id,
            str(self.output_path),
            42,
            1.25,
            1024,
            768,
        )
        restored = ArtisticEditResult.from_json(result.to_json(), temp_root=self.temp_root)
        self.assertEqual(restored, result)
        self.assertEqual(result.to_event()["event"], "result")

    def test_request_rejects_model_lora_mismatch(self):
        mismatched = LoraSpec(
            str(self.lora_path),
            ModelKey.KLEIN_9B,
            1.0,
            "b" * 64,
        )
        with self.assertRaisesRegex(ContractError, "does not match"):
            self.request(lora=mismatched)

    def test_request_rejects_invalid_dimensions_and_numbers(self):
        with self.assertRaisesRegex(ContractError, "multiples of 16"):
            self.request(width=1001)
        with self.assertRaisesRegex(ContractError, "finite"):
            self.request(guidance=float("nan"))
        with self.assertRaisesRegex(ContractError, "between 0.0 and 2.0"):
            LoraSpec(str(self.lora_path), ModelKey.KLEIN_4B, 2.1, "c" * 64)

    def test_paths_are_confined_to_the_request_temp_directory(self):
        request = self.request(output_path=str(self.temp_root.parent / "escaped.png"))
        with self.assertRaisesRegex(ContractError, "escapes"):
            request.validate_paths(self.temp_root)
        with self.assertRaisesRegex(ContractError, "absolute"):
            ArtisticEditResult(self.request_id, "relative.png", 1, 0.1, 16, 16)
        same_path = self.request(output_path=str(self.input_path))
        with self.assertRaisesRegex(ContractError, "must not overwrite"):
            same_path.validate_paths(self.temp_root)

    def test_protocol_parser_rejects_unknown_malformed_and_nonfinite_messages(self):
        valid = encode_json_line({"id": self.request_id, "op": "ping"})
        self.assertEqual(parse_protocol_line(valid)["op"], "ping")
        with self.assertRaisesRegex(ContractError, "Unknown"):
            parse_protocol_line(encode_json_line({"id": self.request_id, "op": "explode"}))
        with self.assertRaisesRegex(ContractError, "valid JSON"):
            parse_protocol_line("not-json")
        nonfinite = json.dumps({"id": self.request_id, "event": "progress"})[:-1]
        nonfinite += ',"fraction":NaN}'
        with self.assertRaisesRegex(ContractError, "Non-finite"):
            parse_protocol_line(nonfinite)

    def test_protocol_path_validation_and_late_result_filter(self):
        event = encode_json_line(
            {
                "id": self.request_id,
                "event": "result",
                "output_path": str(self.output_path),
            }
        )
        parsed = parse_protocol_line(event, temp_root=self.temp_root)
        self.assertTrue(message_matches_request(parsed, self.request_id))
        self.assertFalse(message_matches_request(parsed, str(uuid4())))
        self.assertFalse(message_matches_request(parsed, None))


if __name__ == "__main__":
    unittest.main()
