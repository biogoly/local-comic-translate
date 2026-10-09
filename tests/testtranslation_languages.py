"""Language routing and Portuguese dialect contracts; no live provider calls."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from modules.translation.deepl import DeepLTranslation
from modules.translation.llm.gpt import GPTTranslation
from modules.translation.llm.local import LocalLLMTranslation
from modules.translation.microsoft import MicrosoftTranslation
from modules.utils.language_utils import get_language_code
from modules.utils.textblock import TextBlock


def settings_fixture():
    settings = MagicMock()
    settings.ui.tr.side_effect = lambda text: text
    settings.get_llm_settings.return_value = {"image_input_enabled": False}
    settings.get_credentials.return_value = {
        "api_key": "test-not-a-real-key",
        "api_key_translator": "test-not-a-real-key",
        "region_translator": "test-region",
    }
    return settings


class TranslationLanguageTests(unittest.TestCase):
    def test_canonical_codes_remain_stable(self):
        for name, code in (
            ("Portuguese", "pt"),
            ("Brazilian Portuguese", "pt-br"),
            ("Vietnamese", "vi"),
            ("Auto", None),
        ):
            with self.subTest(name=name):
                self.assertEqual(get_language_code(name), code)

    def test_local_and_cloud_requests_use_explicit_dialects(self):
        for translator_class in (LocalLLMTranslation, GPTTranslation):
            for source, target, source_name, target_name in (
                ("Vietnamese", "Portuguese", "Vietnamese", "European Portuguese (Portugal)"),
                ("Portuguese", "Brazilian Portuguese", "European Portuguese (Portugal)", "Brazilian Portuguese (Brazil)"),
                ("Brazilian Portuguese", "Vietnamese", "Brazilian Portuguese (Brazil)", "Vietnamese"),
            ):
                with self.subTest(backend=translator_class.__name__, source=source, target=target):
                    translator = translator_class()
                    if translator_class is GPTTranslation:
                        translator.initialize(settings_fixture(), source, target, "GPT-4.1-mini")
                    else:
                        translator.initialize(settings_fixture(), source, target)
                    block = TextBlock(text="Xin chào! Olá! Cảm ơn.")
                    translated = "Cảm ơn! Ação, coração e bênção."
                    response = '{"block_0":"' + translated + '"}'
                    with patch.object(translator, "_perform_translation", return_value=response) as perform:
                        self.assertIs(translator.translate([block], None, "Comic context")[0], block)
                    user_prompt, system_prompt, _image = perform.call_args.args
                    self.assertIn(f"translates {source_name} to {target_name}.", system_prompt)
                    self.assertIn(f"a {target_name} speaker", system_prompt)
                    self.assertIn(f"already in {target_name}", system_prompt)
                    self.assertIn("DO NOT translate the keys", system_prompt)
                    self.assertIn("Comic context", user_prompt)
                    self.assertIn(block.text, user_prompt)
                    self.assertEqual(block.translation, translated)
                    self.assertEqual(translator.source_lang, source)
                    self.assertEqual(translator.target_lang, target)

    def test_other_llm_languages_keep_existing_prompt(self):
        translator = LocalLLMTranslation()
        for source, target in (("Japanese", "English"), ("Auto", "French")):
            prompt = translator.get_system_prompt(source, target)
            self.assertTrue(prompt.startswith(f"You are an expert translator who translates {source} to {target}."))
            self.assertNotIn("Portuguese", prompt)

    def test_deepl_request_codes_and_unicode(self):
        for source, target, expected_source, expected_target in (
            ("Portuguese", "Vietnamese", "PT", "VI"),
            ("Brazilian Portuguese", "Portuguese", "PT", "PT-PT"),
            ("Vietnamese", "Brazilian Portuguese", "VI", "PT-BR"),
            ("Auto", "Vietnamese", None, "VI"),
            ("Simplified Chinese", "English", "ZH", "EN-US"),
            ("Traditional Chinese", "French", "ZH", "FR"),
            ("English", "Simplified Chinese", "EN", "ZH-HANS"),
            ("English", "Traditional Chinese", "EN", "ZH-HANT"),
        ):
            with self.subTest(source=source, target=target):
                sdk = MagicMock()
                sdk.translate_text.return_value = SimpleNamespace(text="Chào bạn! Olá!")
                module = SimpleNamespace(Translator=MagicMock(return_value=sdk))
                translator = DeepLTranslation()
                with patch.dict(sys.modules, {"deepl": module}):
                    translator.initialize(settings_fixture(), source, target)
                block = TextBlock(text="Cảm ơn! Ação.")
                translator.translate([block])
                sdk.translate_text.assert_called_once_with(
                    translator.preprocess_text(block.text, expected_source),
                    source_lang=expected_source,
                    target_lang=expected_target,
                )
                self.assertEqual(block.translation, "Chào bạn! Olá!")

    def test_microsoft_honors_explicit_source_and_target(self):
        for source, target, expected_source, expected_target in (
            ("Portuguese", "Vietnamese", "pt-pt", "vi"),
            ("Brazilian Portuguese", "Portuguese", "pt", "pt-pt"),
            ("Vietnamese", "Brazilian Portuguese", "vi", "pt"),
            ("Auto", "Vietnamese", None, "vi"),
            ("Simplified Chinese", "English", "zh-Hans", "en"),
            ("Traditional Chinese", "French", "zh-Hant", "fr"),
            ("English", "Simplified Chinese", "en", "zh-Hans"),
            ("Japanese", "Traditional Chinese", "ja", "zh-Hant"),
        ):
            with self.subTest(source=source, target=target):
                translator = MicrosoftTranslation()
                translator.initialize(settings_fixture(), source, target)
                block = TextBlock(text="Cảm ơn! Ação.")
                response = MagicMock()
                response.json.return_value = [{"translations": [{"text": "Chào bạn! Olá!"}]}]
                with patch("modules.translation.microsoft.requests.post", return_value=response) as post:
                    translator.translate([block])
                params = post.call_args.kwargs["params"]
                self.assertEqual(params["to"], expected_target)
                if expected_source is None:
                    self.assertNotIn("from", params)
                else:
                    self.assertEqual(params["from"], expected_source)
                self.assertEqual(post.call_args.kwargs["json"], [
                    {"text": translator.preprocess_text(block.text, expected_source)}
                ])
                self.assertEqual(block.translation, "Chào bạn! Olá!")


if __name__ == "__main__":
    unittest.main()
