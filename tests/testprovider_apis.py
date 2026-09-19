import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from PySide6 import QtCore, QtWidgets
from app.ui.settings.settings_page import SettingsPage
from modules.translation.factory import TranslationFactory
from modules.translation.processor import Translator
from modules.translation.providers import PROVIDERS
from modules.utils.pipeline_config import validate_ocr, validate_translator


class ProviderAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = QtCore.QSettings(str(Path(self.temp.name) / 'settings.ini'), QtCore.QSettings.IniFormat)
        self.patch_settings = patch('app.ui.settings.settings_page.app_settings', return_value=self.store)
        self.patch_settings.start()
        self.page = SettingsPage()
        self.page.load_settings()
        TranslationFactory._engines.clear()

    def tearDown(self):
        self.page.shutdown()
        self.page.deleteLater()
        self.patch_settings.stop()
        self.store.clear()
        self.temp.cleanup()
        TranslationFactory._engines.clear()

    def test_default_is_local_and_no_account_or_custom_controls(self):
        self.assertEqual(self.page.get_tool_selection('translator'), 'Local LLM')
        self.assertFalse(hasattr(self.page, 'auth_client'))
        self.assertFalse(hasattr(self.page.ui, 'account_page'))
        self.assertEqual(self.page.ui.translator_combo.findText('Custom'), -1)
        self.assertEqual(self.page.ui.stacked_widget.count(), len(self.page.ui.nav_cards))

    def test_direct_engines_use_user_keys_and_model_and_invalidate_cache(self):
        for provider, (service, models) in PROVIDERS.items():
            with self.subTest(provider=provider):
                self.page.ui.credential_widgets[service + '_api_key'].setText('test-key')
                engine = TranslationFactory.create_engine(self.page, 'Spanish', 'English', provider)
                self.assertEqual(engine.api_key, 'test-key')
                self.assertEqual(getattr(engine, 'model_api_name', engine.model), models[0])
                self.page.ui.credential_widgets[service + '_model'].setCurrentText('another-model')
                other = TranslationFactory.create_engine(self.page, 'Spanish', 'English', provider)
                self.assertIsNot(engine, other)
                self.assertEqual(getattr(other, 'model_api_name', other.model), 'another-model')

    def test_keys_are_opt_in_and_old_nested_copies_are_erased(self):
        key = self.page.ui.credential_widgets['Open AI GPT_api_key']
        key.setText('not-for-disk')
        self.store.setValue('credentials/Custom/api_key', 'old-key')
        self.store.setValue('credentials/Custom_api_key', 'old-key')
        self.page.ui.llms_page.local_api_key.setText('local-secret')
        self.page.save_settings()
        self.assertEqual(self.page.get_credentials('OpenAI')['api_key'], 'not-for-disk')
        self.assertFalse(self.store.contains('credentials/Open AI GPT_api_key'))
        self.assertFalse(self.store.contains('credentials/Custom/api_key'))
        self.assertFalse(self.store.contains('credentials/Custom_api_key'))
        self.assertEqual(self.store.value('llm/local_api_key'), '')
        self.page.ui.save_keys_checkbox.setChecked(True)
        self.page.save_settings()
        self.assertEqual(self.store.value('credentials/Open AI GPT_api_key'), 'not-for-disk')
        self.page.load_settings()
        self.assertEqual(key.text(), 'not-for-disk')
        self.page.ui.save_keys_checkbox.setChecked(False)
        self.page.save_settings()
        self.assertFalse(self.store.contains('credentials/Open AI GPT_api_key'))

    def test_old_named_cloud_model_migrates_without_resetting_choice(self):
        self.store.setValue('tools/translator', 'GPT-4.1')
        self.page.load_settings()
        self.assertEqual(self.page.get_tool_selection('translator'), 'OpenAI')
        self.assertEqual(self.page.get_credentials('OpenAI')['model'], 'gpt-4.1')

    def test_old_custom_endpoint_is_preserved_as_external_connection(self):
        self.store.setValue('tools/translator', 'Custom')
        self.store.setValue('credentials/Custom_api_url', 'http://localhost:8888/v1')
        self.store.setValue('credentials/Custom_model', 'test-model')
        self.store.setValue('credentials/Custom_api_key', 'old-secret')
        self.store.setValue('credentials/save_keys', True)
        self.page.load_settings()
        self.assertEqual(self.page.get_tool_selection('translator'), 'Local LLM')
        self.assertEqual(self.page.get_llm_settings()['local_endpoint'], 'http://localhost:8888/v1')
        self.assertEqual(self.page.get_llm_settings()['local_api_key'], 'old-secret')

    def test_validation_requires_key_not_upstream_login(self):
        main = SimpleNamespace(settings_page=self.page)
        self.page.ui.translator_combo.setCurrentText('Google Gemini')
        with patch('modules.utils.pipeline_config.Messages.show_error_with_copy') as error:
            self.assertFalse(validate_translator(main, 'English'))
            error.assert_called_once()
        self.page.ui.credential_widgets['Google Gemini_api_key'].setText('gemini-key')
        self.assertTrue(validate_translator(main, 'English'))
        self.page.ui.ocr_combo.setCurrentText('Gemini-2.5-Flash-Lite')
        self.assertTrue(validate_ocr(main))
        self.page.ui.ocr_combo.setCurrentText('Default')
        self.assertTrue(validate_ocr(main))

    def test_provider_labels_map_in_all_locales(self):
        root = Path(__file__).resolve().parents[1]
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            translator = QtCore.QTranslator()
            translator.load(str(root / f'resources/translations/compiled/ct_{locale}.qm'))
            self.app.installTranslator(translator)
            try:
                for provider in PROVIDERS:
                    value = Translator._get_translator_key(SimpleNamespace(settings=self.page), self.page.ui.tr(provider))
                    self.assertEqual(value, provider)
            finally:
                self.app.removeTranslator(translator)

    def test_gemini_request_uses_header_key_and_supported_thinking_config(self):
        self.page.ui.credential_widgets['Google Gemini_api_key'].setText('private-key')
        self.page.ui.credential_widgets['Google Gemini_model'].setCurrentText('gemini-2.5-pro')
        engine = TranslationFactory.create_engine(self.page, 'Spanish', 'English', 'Google Gemini')
        response = Mock(status_code=200)
        response.json.return_value = {'candidates': [{'content': {'parts': [{'text': '{}'}]}}]}
        with patch('modules.translation.llm.gemini.requests.post', return_value=response) as post:
            engine._perform_translation('hello', 'translate', None)
        self.assertNotIn('private-key', post.call_args.args[0])
        self.assertEqual(post.call_args.kwargs['headers']['x-goog-api-key'], 'private-key')
        self.assertIn('thinkingBudget', post.call_args.kwargs['json']['generationConfig']['thinkingConfig'])

    def test_deepseek_uses_compatible_payload_and_direct_endpoint(self):
        import json
        self.page.ui.credential_widgets['Deepseek_api_key'].setText('direct-key')
        engine = TranslationFactory.create_engine(self.page, 'Spanish', 'English', 'Deepseek')
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': '{}'}}]}
        with patch('modules.translation.llm.gpt.requests.post', return_value=response) as post:
            engine._perform_translation('Hola', 'translate', None)
        self.assertEqual(post.call_args.args[0], 'https://api.deepseek.com/v1/chat/completions')
        payload = json.loads(post.call_args.kwargs['data'])
        self.assertIn('max_tokens', payload)
        self.assertNotIn('max_completion_tokens', payload)
        self.assertEqual(payload['messages'][0]['content'], 'translate')

    def test_unknown_provider_does_not_fall_back_to_paid_service(self):
        with self.assertRaises(ValueError):
            TranslationFactory._get_engine_class('Custom')


if __name__ == '__main__':
    unittest.main()
