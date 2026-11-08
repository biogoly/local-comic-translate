"""Exercise the context editor with the actual shipped Qt translations."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtWidgets

from app.ui.dialogs.translation_context import ContextEntryDialog, SummarySuggestionDialog, TranslationContextDialog
from modules.translation.context import empty_translation_profile


class TranslationContextLocaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_editor_entry_validation_and_status_in_all_nine_languages(self):
        catalogs = Path(__file__).resolve().parents[1] / 'resources/translations/compiled'
        for locale in ('de', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'tr', 'zh-CN'):
            with self.subTest(locale=locale):
                translator = QtCore.QTranslator()
                self.assertTrue(translator.load(str(catalogs / f'ct_{locale}.qm')))
                self.app.installTranslator(translator)
                dialog = entry = suggestion = None
                try:
                    translated = lambda source: translator.translate('TranslationContextDialog', source)
                    dialog = TranslationContextDialog(empty_translation_profile())
                    self.assertEqual(dialog.windowTitle(), translated('Translation context'))
                    self.assertNotEqual(dialog.windowTitle(), 'Translation context')
                    self.assertEqual(dialog.history_checkbox.text(), translated('Include approved translations from earlier pages'))
                    self.assertEqual(dialog.suggest_summary_button.text(), translated('Suggest story summary…'))
                    request = SimpleNamespace(provider='Fixture', destination='fixture.example', local=False,
                                              examples=(SimpleNamespace(source='Bonjour', target='Olá'),), summary='')
                    suggestion = SummarySuggestionDialog(request, 'French', 'Portuguese')
                    suggestion_tr = lambda source: translator.translate('SummarySuggestionDialog', source)
                    self.assertEqual(suggestion.generate_button.text(), suggestion_tr('Generate'))
                    self.assertEqual(suggestion.use_button.text(), suggestion_tr('Use suggestion'))
                    self.assertEqual(suggestion.inputs_toggle.text(), suggestion_tr(
                        'Approved translations ({count}) · {source} → {target}').format(count=1, source='French', target='Portuguese'))
                    disclosure = suggestion_tr('Generate sends the displayed material to {destination} using {provider}.')
                    self.assertTrue(any(label.text() == disclosure.format(destination='fixture.example', provider='Fixture')
                                        for label in suggestion.findChildren(QtWidgets.QLabel)))
                    self.assertEqual([dialog.tabs.tabText(i) for i in range(3)],
                                     [translated(s) for s in ('Glossary', 'Characters', 'Notes')])
                    for button, source in ((dialog.apply_button, 'Apply'), (dialog.undo_button, 'Undo context'),
                                           (dialog.redo_button, 'Redo context'), (dialog.preview_button, 'Build preview'),
                                           (dialog.add_selected_button, 'Add selected source/translation')):
                        self.assertTrue(translated(source))
                        self.assertEqual(button.text(), translated(source))
                    dialog.notes_edit.setPlainText('Đặng — ありがとう — lembrança')
                    self.assertEqual(dialog.profile()['notes'], 'Đặng — ありがとう — lembrança')
                    self.assertEqual(dialog.status_label.text(), translated(
                        'Unapplied changes. Apply to use them for translation.'))
                    dialog.set_supported(False)
                    self.assertEqual(dialog.support_label.text(), translated(
                        'This translator does not support structured LLM context. Project entries remain editable for use with an LLM.'))
                    self.assertFalse(dialog.preview_button.isEnabled())
                    dialog.set_selection({'page_id': 'page', 'region_id': 'region', 'text': 'Alice'})
                    self.assertEqual(dialog.selection_label.text(), translated('Selected region: {text}').format(text='Alice'))
                    entry = ContextEntryDialog('glossary')
                    self.assertEqual(entry.windowTitle(), translator.translate('ContextEntryDialog', 'Glossary entry'))
                    entry.accept()
                    self.assertEqual(entry.error.text(), translator.translate('ContextEntryDialog',
                                     'Enter a source term and preferred translation.'))
                    self.assertEqual(entry.match.itemText(1), translator.translate('ContextEntryDialog', 'Whole word'))
                    for context in ('WorkspaceMixin', 'ComicTranslate', 'ComicTranslateUI'):
                        self.assertTrue(translator.translate(context, 'Translation context'))
                finally:
                    for widget in (suggestion, entry, dialog):
                        if widget is not None:
                            widget.close()
                            widget.deleteLater()
                    self.app.removeTranslator(translator)


if __name__ == '__main__':
    unittest.main()
