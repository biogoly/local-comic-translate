import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtWidgets

from app.controllers.artistic_edit import ArtisticEditController, build_translation_prompt
from app.ui.artistic_edit_panel import ArtisticEditPanel
from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog
from app.ui.main_window.builders.workspace import WorkspaceMixin
from app.ui.settings.settings_ui import SettingsPageUI
from app.ui.settings.settings_page import SettingsPage


class ComicTranslate(WorkspaceMixin, QtWidgets.QMainWindow):
    """Exercise the actual Qt context used by the application's workspace mixin."""


class UiTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.catalog_dir = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'

    def test_feature_widgets_and_saved_values_in_every_translated_interface(self):
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            with self.subTest(locale=locale):
                translator = QtCore.QTranslator()
                self.assertTrue(translator.load(str(self.catalog_dir / f'ct_{locale}.qm')))
                self.app.installTranslator(translator)
                panel = settings = preview = workspace = None
                try:
                    panel = ArtisticEditPanel()
                    expected = translator.translate('ArtisticEditPanel', 'Generate Preview')
                    self.assertTrue(expected)
                    self.assertNotEqual(expected, 'Generate Preview')
                    self.assertEqual(panel.generate_button.text(), expected)
                    self.assertEqual(panel.area_combo.currentData(), 'selected_box')
                    self.assertEqual(panel.template_key(), 'lettering')
                    self.assertEqual(panel.options().device_policy.value, 'model_cpu_offload')

                    settings = SettingsPageUI()
                    for name in ('Midnight', 'Parchment', 'Lavender', 'Mint'):
                        label = translator.translate('SettingsPageUI', name)
                        self.assertNotEqual(label, name)
                        settings.theme_combo.setCurrentText(label)
                        self.assertEqual(settings.value_mappings[settings.theme_combo.currentText()], name)
                        self.assertEqual(settings.reverse_mappings[name], label)
                    local_label = translator.translate('SettingsPageUI', 'Local LLM')
                    self.assertEqual(settings.value_mappings[local_label], 'Local LLM')
                    self.assertEqual(settings.llms_page.local_settings_widget.title(), local_label)
                    SettingsPage._sync_extra_context_limit(SimpleNamespace(ui=settings), local_label)
                    context = 'Long translation context. ' * 200
                    settings.llms_page.extra_context.setPlainText(context)
                    self.assertEqual(settings.llms_page.extra_context.toPlainText(), context)

                    workspace = ComicTranslate()
                    for source in ('Sort Pages', 'Artistic Edit (FLUX.2 Klein)', 'Fit Page to Window (Ctrl+0)'):
                        self.assertEqual(workspace.tr(source), translator.translate('WorkspaceMixin', source))
                        self.assertNotEqual(workspace.tr(source), source)

                    page = np.zeros((20, 20, 3), dtype=np.uint8)
                    preview = ArtisticEditPreviewDialog(page, page, '', apply_available=True)
                    self.assertEqual(preview.windowTitle(), translator.translate('ArtisticEditPreviewDialog', 'Artistic Edit Preview'))
                    self.assertEqual(preview.apply_button.text(), translator.translate('ArtisticEditPreviewDialog', 'Apply'))
                    with self.assertRaises(ValueError) as error:
                        build_translation_prompt('Hello', '', 'lettering')
                    self.assertEqual(str(error.exception), translator.translate('ArtisticEditController', 'The selected box has no translation yet.'))

                    controller = SimpleNamespace(panel=panel, tr=lambda text: translator.translate('ArtisticEditController', text))
                    ArtisticEditController._on_loading(controller, 'Loading klein-4b BF16 weights…')
                    self.assertIn('klein-4b', panel.status_label.text())
                    self.assertNotEqual(panel.status_label.text(), 'Loading klein-4b BF16 weights…')
                    ArtisticEditController._on_loading(controller, 'Configuring model cpu offload…')
                    self.assertIn(panel.device_combo.currentText(), panel.status_label.text())
                finally:
                    for widget in (panel, settings, preview, workspace):
                        if widget is not None:
                            widget.deleteLater()
                    self.app.removeTranslator(translator)


if __name__ == '__main__':
    unittest.main()
