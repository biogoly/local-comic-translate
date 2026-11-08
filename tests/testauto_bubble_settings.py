"""Optional bubble fitting controls, isolated persistence, and shipped translations."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtWidgets

from app.controllers.projects import ProjectController
from app.ui.settings.settings_page import SettingsPage
from app.ui.settings.text_rendering_page import TextRenderingPage
from modules.rendering.render import TextRenderingSettings


def legacy_render_settings():
    return TextRenderingSettings(
        alignment_id=1, font_family='Arial', min_font_size=9, max_font_size=40,
        color='#000000', upper_case=False, outline=False, outline_color='#ffffff',
        outline_width='1.0', bold=False, italic=False, underline=False,
        line_spacing='1.0', direction=QtCore.Qt.LayoutDirection.LeftToRight,
    )


class AutoBubbleSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='bubble-settings-')
        self.addCleanup(self.temp.cleanup)
        self.settings = QtCore.QSettings(str(Path(self.temp.name) / 'test.ini'), QtCore.QSettings.Format.IniFormat)
        self.addCleanup(self.settings.sync)
        self.addCleanup(patch.stopall)
        patch('app.ui.settings.settings_page.app_settings', return_value=self.settings).start()
        patch('app.controllers.projects.app_settings', return_value=self.settings).start()
        patch('app.ui.settings.settings_page.is_gpu_available', return_value=False).start()

    def make_settings_page(self):
        page = SettingsPage()
        self.addCleanup(page.deleteLater)
        return page

    def test_disabled_default_padding_range_and_toggle(self):
        page = self.make_settings_page()
        self.assertIs(page.ui.auto_bubble_fit_checkbox, page.ui.text_rendering_page.auto_bubble_fit_checkbox)
        self.assertIs(page.ui.auto_bubble_padding_spinbox, page.ui.text_rendering_page.auto_bubble_padding_spinbox)
        self.assertEqual(page.get_bubble_render_settings(), {'auto_bubble_fit': False, 'bubble_padding': 8.0})
        padding = page.ui.auto_bubble_padding_spinbox
        self.assertEqual((padding.minimum(), padding.maximum()), (0, 200))
        self.assertFalse(padding.isEnabled())
        page.ui.auto_bubble_fit_checkbox.setChecked(True)
        self.assertTrue(padding.isEnabled())
        padding.setValue(13)
        page.ui.auto_bubble_fit_checkbox.setChecked(False)
        self.assertFalse(padding.isEnabled())
        self.assertEqual(padding.value(), 13)

    def test_settings_save_reload_and_disable_preserve_other_render_preferences(self):
        self.settings.setValue('text_rendering/min_font_size', 19)
        page = self.make_settings_page()
        page.ui.auto_bubble_fit_checkbox.setChecked(True)
        page.ui.auto_bubble_padding_spinbox.setValue(14)
        page.save_settings()
        self.assertTrue(self.settings.value('text_rendering/auto_bubble_fit', type=bool))
        self.assertEqual(self.settings.value('text_rendering/bubble_padding', type=float), 14.0)
        self.assertEqual(self.settings.value('text_rendering/min_font_size', type=int), 19)
        restored = self.make_settings_page()
        restored.load_settings()
        self.assertEqual(restored.get_bubble_render_settings(), {'auto_bubble_fit': True, 'bubble_padding': 14.0})
        self.assertTrue(restored.ui.auto_bubble_padding_spinbox.isEnabled())
        restored.ui.auto_bubble_fit_checkbox.setChecked(False)
        restored.save_settings()
        page.load_settings()
        self.assertFalse(page.ui.auto_bubble_fit_checkbox.isChecked())
        self.assertFalse(page.ui.auto_bubble_padding_spinbox.isEnabled())

    def test_legacy_settings_missing_keys_restore_disabled_defaults(self):
        page = self.make_settings_page()
        page.ui.auto_bubble_fit_checkbox.setChecked(True)
        page.ui.auto_bubble_padding_spinbox.setValue(50)
        page.load_settings()
        self.assertEqual(page.get_bubble_render_settings(), {'auto_bubble_fit': False, 'bubble_padding': 8.0})
        self.assertFalse(page.ui.auto_bubble_padding_spinbox.isEnabled())
        old_constructor = legacy_render_settings()
        self.assertFalse(old_constructor.auto_bubble_fit)
        self.assertEqual(old_constructor.bubble_padding, 8.0)

    def test_main_page_settings_save_load_and_missing_key_defaults(self):
        page = self.make_settings_page()
        main = MagicMock()
        main.settings_page = page
        main.lang_mapping = {'Auto': 'Auto', 'English': 'English'}
        main.reverse_lang_mapping = main.lang_mapping
        main.s_combo.currentText.return_value = 'Auto'
        main.t_combo.currentText.return_value = 'English'
        main.image_viewer = SimpleNamespace(brush_size=10, eraser_size=20)
        main.saveGeometry.return_value = QtCore.QByteArray()
        main.saveState.return_value = QtCore.QByteArray()
        options = legacy_render_settings()
        options.auto_bubble_fit, options.bubble_padding = True, 22.0
        main.render_settings.return_value = options
        controller = ProjectController.__new__(ProjectController)
        controller.main = main
        controller.save_main_page_settings()
        controller.load_main_page_settings()
        self.assertEqual(page.get_bubble_render_settings(), {'auto_bubble_fit': True, 'bubble_padding': 22.0})
        self.assertTrue(page.ui.auto_bubble_padding_spinbox.isEnabled())
        self.settings.remove('text_rendering/auto_bubble_fit')
        self.settings.remove('text_rendering/bubble_padding')
        controller.load_main_page_settings()
        self.assertEqual(page.get_bubble_render_settings(), {'auto_bubble_fit': False, 'bubble_padding': 8.0})

    def test_controls_and_tooltip_in_all_nine_interface_languages(self):
        catalogs = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'
        tooltip = ('Fit newly rendered text to speech-bubble interiors, with Box fallback. '
                   'Existing lettering is preserved; other regions continue through processing.')
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            with self.subTest(locale=locale):
                translator = QtCore.QTranslator()
                self.assertTrue(translator.load(str(catalogs / f'ct_{locale}.qm')))
                self.app.installTranslator(translator)
                page = None
                try:
                    page = TextRenderingPage()
                    for source, actual in (
                        ('Fit to bubble when rendering', page.auto_bubble_fit_checkbox.text()),
                        ('Bubble padding:', page.auto_bubble_padding_spinbox.accessibleName()),
                        (tooltip, page.auto_bubble_fit_checkbox.toolTip()),
                    ):
                        expected = translator.translate('TextRenderingPage', source)
                        self.assertTrue(expected)
                        self.assertNotEqual(expected, source)
                        self.assertEqual(actual, expected)
                    for source in (
                        'Page already contains rendered text; use manual Render to add missing regions.',
                        'Existing Rendered Text',
                    ):
                        expected = translator.translate('Messages', source)
                        self.assertTrue(expected)
                        self.assertNotEqual(expected, source)
                        self.assertEqual(QtCore.QCoreApplication.translate('Messages', source), expected)
                finally:
                    if page is not None:
                        page.deleteLater()
                    self.app.removeTranslator(translator)


if __name__ == '__main__':
    unittest.main()
