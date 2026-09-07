from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules.translation.llm.llama_server import (
    LlamaServerConfig,
    MANAGED_MODEL_ALIAS,
    LlamaServerRuntime,
    resolve_llama_server,
)
from modules.translation.llm.local import LocalLLMTranslation
from modules.translation.llm.local_response import (
    TranslationResponseError,
    build_chat_payload,
    chat_completions_url,
    extract_message_content,
    parse_translation_object,
    request_valid_translation,
)


class TranslationResponseTests(unittest.TestCase):
    def test_accepts_wrapped_json_with_exact_block_ids(self):
        content = '<think>done</think>\n```json\n{"block_0":"Hello","block_1":"Bye"}\n```'

        result = parse_translation_object(content, ["block_0", "block_1"])

        self.assertEqual(result, {"block_0": "Hello", "block_1": "Bye"})

    def test_rejects_missing_or_extra_ids(self):
        with self.assertRaisesRegex(TranslationResponseError, "missing keys"):
            parse_translation_object('{"block_0":"Hello"}', ["block_0", "block_1"])

        with self.assertRaisesRegex(TranslationResponseError, "unexpected keys"):
            parse_translation_object(
                '{"block_0":"Hello","block_1":"Bye","block_2":"No"}',
                ["block_0", "block_1"],
            )

    def test_rejects_non_string_translation(self):
        with self.assertRaisesRegex(TranslationResponseError, "non-string values"):
            parse_translation_object('{"block_0":42}', ["block_0"])

    def test_rejects_blank_or_punctuation_only_required_translation(self):
        with self.assertRaisesRegex(TranslationResponseError, "without translated text"):
            parse_translation_object(
                '{"block_0":"Hello","block_1":" ... "}',
                ["block_0", "block_1"],
                required_content_keys=["block_0", "block_1"],
            )

    def test_extracts_string_and_content_parts(self):
        self.assertEqual(
            extract_message_content({"choices": [{"message": {"content": "text"}}]}),
            "text",
        )
        self.assertEqual(
            extract_message_content({
                "choices": [{"message": {"content": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ]}}],
            }),
            "onetwo",
        )

    def test_retries_incomplete_output_once(self):
        responses = iter([
            '{"block_0":"Hello"}',
            '{"block_0":"Hello","block_1":"Bye"}',
        ])
        prompts = []

        def perform(prompt):
            prompts.append(prompt)
            return next(responses)

        result = request_valid_translation(
            perform,
            "translate",
            ["block_0", "block_1"],
            max_attempts=2,
        )

        self.assertEqual(result["block_1"], "Bye")
        self.assertEqual(len(prompts), 2)
        self.assertIn("previous response was invalid", prompts[1])

    def test_retries_blank_translation(self):
        responses = iter([
            '{"block_0":""}',
            '{"block_0":"Translated"}',
        ])

        result = request_valid_translation(
            lambda _prompt: next(responses),
            "translate",
            ["block_0"],
            max_attempts=2,
            required_content_keys=["block_0"],
        )

        self.assertEqual(result["block_0"], "Translated")


class OpenAICompatibilityTests(unittest.TestCase):
    def test_multimodal_payload_is_image_first_and_uses_max_tokens(self):
        payload = build_chat_payload(
            model="gemma4:e4b",
            system_prompt="system",
            user_prompt="translate",
            image_data_url="data:image/jpeg;base64,abc",
            temperature=0.2,
            top_p=0.9,
            top_k=64,
            max_tokens=4096,
        )

        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "image_url")
        self.assertEqual(user_content[1], {"type": "text", "text": "translate"})
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["top_k"], 64)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_endpoint_normalization(self):
        self.assertEqual(
            chat_completions_url("http://127.0.0.1:11434/v1/"),
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        full = "http://127.0.0.1:8080/v1/chat/completions"
        self.assertEqual(chat_completions_url(full), full)

    def test_local_translator_reads_top_k_setting(self):
        settings = MagicMock()
        settings.get_llm_settings.return_value = {
            "local_top_k": 64,
            "image_input_enabled": True,
        }
        translator = LocalLLMTranslation()

        translator.initialize(settings, "Japanese", "English")

        self.assertEqual(translator.top_k, 64)


class LlamaServerRuntimeTests(unittest.TestCase):
    def test_command_contains_model_alias_and_multimodal_projector(self):
        config = LlamaServerConfig(
            executable_path="llama-server",
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
            context_size=16384,
            gpu_layers=99,
        )

        command = LlamaServerRuntime.build_command(config, 54321)

        self.assertEqual(command[0], "llama-server")
        self.assertIn(MANAGED_MODEL_ALIAS, command)
        self.assertIn("--jinja", command)
        self.assertEqual(command[command.index("--port") + 1], "54321")
        self.assertEqual(command[command.index("--mmproj") + 1], "mmproj.gguf")

    def test_explicit_executable_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "llama-server"
            executable.touch()

            self.assertEqual(resolve_llama_server(str(executable)), executable.resolve())

    def test_managed_runtime_reuses_process_and_stops_it(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "llama-server"
            model = Path(directory) / "model.gguf"
            executable.touch()
            model.touch()
            config = LlamaServerConfig(
                executable_path=str(executable),
                model_path=str(model),
            )
            process = MagicMock()
            process.poll.return_value = None
            process.stdout = []
            process.wait.return_value = 0
            runtime = LlamaServerRuntime()

            with (
                patch(
                    "modules.translation.llm.llama_server.subprocess.Popen",
                    return_value=process,
                ) as popen,
                patch.object(runtime, "_wait_until_ready"),
            ):
                first_url = runtime.ensure_running(config)
                second_url = runtime.ensure_running(config)
                runtime.stop()

            self.assertEqual(first_url, second_url)
            self.assertTrue(first_url.startswith("http://127.0.0.1:"))
            popen.assert_called_once()
            process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
