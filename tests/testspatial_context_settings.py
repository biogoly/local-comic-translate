"""Opt-in spatial context controls and isolated settings persistence."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtWidgets

from app.ui.settings.settings_page import SettingsPage
from app.ui.settings.llms_page import LlmsPage
from modules.translation.providers import translator_supports_images


class SpatialContextSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='spatial-settings-')
        self.addCleanup(self.temp.cleanup)
        self.store = QtCore.QSettings(str(Path(self.temp.name) / 'test.ini'), QtCore.QSettings.IniFormat)
        self.addCleanup(self.store.sync)
        settings_patch = patch('app.ui.settings.settings_page.app_settings', return_value=self.store)
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        gpu_patch = patch('app.ui.settings.settings_page.is_gpu_available', return_value=False)
        gpu_patch.start()
        self.addCleanup(gpu_patch.stop)

    def page(self):
        page = SettingsPage()
        self.addCleanup(page.deleteLater)
        self.addCleanup(page.shutdown)
        return page

    def test_opt_in_default_and_image_toggle_preserves_preference(self):
        page = self.page()
        coordinate = page.ui.spatial_context_checkbox
        self.assertIs(coordinate, page.ui.llms_page.spatial_context_checkbox)
        self.assertFalse(page.get_llm_settings()['spatial_context_enabled'])
        self.assertFalse(coordinate.isEnabled())
        page.ui.image_checkbox.setChecked(True)
        self.assertTrue(coordinate.isEnabled())
        coordinate.setChecked(True)
        page.ui.image_checkbox.setChecked(False)
        self.assertFalse(coordinate.isEnabled())
        self.assertTrue(coordinate.isChecked())
        page.ui.image_checkbox.setChecked(True)
        self.assertTrue(coordinate.isEnabled())
        self.assertTrue(coordinate.isChecked())

    def test_provider_and_model_changes_gate_control_without_clearing_it(self):
        page = self.page()
        page.ui.image_checkbox.setChecked(True)
        page.ui.spatial_context_checkbox.setChecked(True)
        for provider in ('Local LLM', 'OpenAI', 'Google Gemini', 'Anthropic Claude'):
            page.ui.translator_combo.setCurrentText(page.ui.tr(provider))
            self.assertTrue(page.ui.spatial_context_checkbox.isEnabled(), provider)
        page.ui.translator_combo.setCurrentText(page.ui.tr('Deepseek'))
        model = page.ui.credential_widgets['Deepseek_model']
        model.setCurrentText('deepseek-flash')
        self.assertTrue(page.ui.spatial_context_checkbox.isEnabled())
        model.setCurrentText('deepseek-v4-pro')
        self.assertFalse(page.ui.spatial_context_checkbox.isEnabled())
        self.assertTrue(page.ui.spatial_context_checkbox.isChecked())
        model.setCurrentText('deepseek-flash')
        self.assertTrue(page.ui.spatial_context_checkbox.isEnabled())
        for provider in ('DeepL', 'Yandex', 'Microsoft Translator', 'unknown provider'):
            page.ui.translator_combo.addItem(provider)
            page.ui.translator_combo.setCurrentText(provider)
            self.assertFalse(page.ui.spatial_context_checkbox.isEnabled(), provider)
            self.assertTrue(page.ui.spatial_context_checkbox.isChecked())

    def test_save_reload_retains_disabled_preference_and_model_capability(self):
        page = self.page()
        page.ui.image_checkbox.setChecked(True)
        page.ui.spatial_context_checkbox.setChecked(True)
        page.ui.translator_combo.setCurrentText(page.ui.tr('Deepseek'))
        page.ui.credential_widgets['Deepseek_model'].setCurrentText('deepseek-v4-pro')
        page.save_settings()
        self.assertTrue(self.store.value('llm/spatial_context_enabled', type=bool))
        restored = self.page()
        restored.load_settings()
        self.assertTrue(restored.get_llm_settings()['spatial_context_enabled'])
        self.assertFalse(restored.ui.spatial_context_checkbox.isEnabled())
        restored.ui.translator_combo.setCurrentText(restored.ui.tr('Local LLM'))
        self.assertTrue(restored.ui.spatial_context_checkbox.isEnabled())
        restored.ui.image_checkbox.setChecked(False)
        restored.save_settings()
        page.load_settings()
        self.assertTrue(page.ui.spatial_context_checkbox.isChecked())
        self.assertFalse(page.ui.spatial_context_checkbox.isEnabled())

    def test_legacy_settings_restore_default_off(self):
        self.store.setValue('llm/image_input_enabled', True)
        page = self.page()
        page.ui.spatial_context_checkbox.setChecked(True)
        page.load_settings()
        self.assertFalse(page.get_llm_settings()['spatial_context_enabled'])
        self.assertTrue(page.ui.spatial_context_checkbox.isEnabled())

    def test_localized_provider_labels_use_adapter_capability(self):
        catalogs = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'
        translator = QtCore.QTranslator()
        self.assertTrue(translator.load(str(catalogs / 'ct_ja.qm')))
        self.app.installTranslator(translator)
        self.addCleanup(self.app.removeTranslator, translator)
        page = self.page()
        page.ui.image_checkbox.setChecked(True)
        page.ui.translator_combo.setCurrentText(page.ui.tr('Anthropic Claude'))
        self.assertTrue(page.ui.spatial_context_checkbox.isEnabled())
        page.ui.translator_combo.setCurrentText(page.ui.tr('Deepseek'))
        page.ui.credential_widgets['Deepseek_model'].setCurrentText('deepseek-v4-pro')
        self.assertFalse(page.ui.spatial_context_checkbox.isEnabled())

    def test_shared_capabilities_preserve_legacy_provider_aliases(self):
        for alias in ('GPT-4.1', 'Gemini-2.5-Pro', 'Claude-4.6-Sonnet', 'Local LLM'):
            self.assertTrue(translator_supports_images(alias))
        self.assertFalse(translator_supports_images('Deepseek', 'deepseek-v4-pro'))
        self.assertTrue(translator_supports_images('Deepseek', 'deepseek-flash'))
        self.assertFalse(translator_supports_images('DeepL'))
        self.assertFalse(translator_supports_images('unknown'))

    def test_new_controls_use_all_nine_compiled_locales(self):
        catalogs = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'
        source = 'Provide text-box coordinates (experimental)'
        help_source = ('Send text-box positions with the page image. Requires image input '
                       'and a vision-capable model; improvement is not guaranteed.')
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            with self.subTest(locale=locale):
                translator = QtCore.QTranslator()
                self.assertTrue(translator.load(str(catalogs / f'ct_{locale}.qm')))
                self.app.installTranslator(translator)
                page = None
                try:
                    page = LlmsPage()
                    expected = translator.translate('LlmsPage', source)
                    expected_help = translator.translate('LlmsPage', help_source)
                    self.assertTrue(expected)
                    self.assertTrue(expected_help)
                    self.assertNotEqual(expected, source)
                    self.assertNotEqual(expected_help, help_source)
                    self.assertEqual(page.spatial_context_checkbox.text(), expected)
                    self.assertEqual(page.spatial_context_help.text(), expected_help)
                    self.assertEqual(page.spatial_context_checkbox.toolTip(), expected_help)
                finally:
                    if page is not None:
                        page.deleteLater()
                    self.app.removeTranslator(translator)


if __name__ == '__main__':
    unittest.main()
