"""Vietnamese/Portuguese selectors retain canonical project identifiers."""
import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.controllers.image import ImageStateController
from app.controllers.text import TextController
from app.ui.dayu_widgets.combo_box import MComboBox
from app.ui.main_window.constants import supported_source_languages, supported_target_languages
from app.ui.main_window.window import ComicTranslateUI


class ComicTranslate(ComicTranslateUI):
    """Exercise inherited Qt translation context without loading user settings."""

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self._init_language_mappings()
        self.s_combo = MComboBox(self)
        self.s_combo.addItems([self.reverse_lang_mapping[lang] for lang in supported_source_languages])
        self.t_combo = MComboBox(self)
        self.t_combo.addItems([self.reverse_lang_mapping[lang] for lang in supported_target_languages])
        self.manual_radio = QtWidgets.QRadioButton(self)
        self.manual_radio.setChecked(True)
        self.t_text_edit = QtWidgets.QTextEdit(self)
        self.image_files = ["one", "two"]
        self.image_states = {"one": {}, "two": {}}
        self.curr_img_idx = 0
        self.mark_project_dirty = lambda: None


class SourceLanguageChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.catalog_dir = Path(__file__).resolve().parents[1] / "resources/translations/compiled"

    def window(self):
        main = ComicTranslate()
        self.addCleanup(main.deleteLater)
        return main

    def restore(self, main, state, **kwargs):
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        return controller.restore_page_languages(state, **kwargs)

    def test_both_selectors_include_new_sources_and_explicit_portuguese_dialects(self):
        main = self.window()
        for canonical, label in (("Vietnamese", "Vietnamese"),
                                 ("Portuguese", "Portuguese (Portugal)"),
                                 ("Brazilian Portuguese", "Portuguese (Brazil)")):
            with self.subTest(language=canonical):
                self.assertIn(canonical, supported_source_languages)
                self.assertEqual(main.lang_mapping[label], canonical)
                self.assertEqual(main.reverse_lang_mapping[canonical], label)
                for combo in (main.s_combo, main.t_combo):
                    self.assertGreaterEqual(combo.findText(label), 0)
                    self.assertEqual(sum(combo.itemText(i) == label for i in range(combo.count())), 1)
        self.assertEqual(main.lang_mapping["Portuguese"], "Portuguese")
        self.assertEqual(main.lang_mapping["Brazilian Portuguese"], "Brazilian Portuguese")
        self.assertEqual(main.s_combo.findText("Portuguese"), -1)
        self.assertEqual(main.t_combo.findText("Brazilian Portuguese"), -1)

    def test_legacy_project_identifiers_restore_to_new_display_labels(self):
        main = self.window()
        for language in ("Vietnamese", "Portuguese", "Brazilian Portuguese"):
            state = {"source_lang": language, "target_lang": language}
            self.assertEqual(self.restore(main, state, seed_sticky_languages=True), (language, language))
            self.assertEqual(main.s_combo.currentText(), main.reverse_lang_mapping[language])
            self.assertEqual(main.t_combo.currentText(), main.reverse_lang_mapping[language])
            self.assertEqual(state, {"source_lang": language, "target_lang": language})

    def test_manual_language_pair_remains_sticky_and_saves_canonical_names(self):
        main = self.window()
        for source, target in (("Vietnamese", "Portuguese"),
                               ("Portuguese", "Brazilian Portuguese"),
                               ("Brazilian Portuguese", "Vietnamese"),
                               ("Auto", "Portuguese")):
            main.s_combo.setCurrentText(main.reverse_lang_mapping[source])
            main.t_combo.setCurrentText(main.reverse_lang_mapping[target])
            controller = TextController.__new__(TextController)
            controller.main = main
            controller.save_src_trg()
            for state in main.image_states.values():
                self.assertEqual(state, {"source_lang": source, "target_lang": target})
            stale_page = {"source_lang": "Auto", "target_lang": "English"}
            self.assertEqual(self.restore(main, stale_page), (source, target))
            self.assertEqual(main.s_combo.currentText(), main.reverse_lang_mapping[source])
            self.assertEqual(main.t_combo.currentText(), main.reverse_lang_mapping[target])

    def test_automatic_pages_restore_their_individual_languages(self):
        main = self.window()
        main.manual_radio.setChecked(False)
        for source, target in (("Portuguese", "Vietnamese"), ("Brazilian Portuguese", "English")):
            self.assertEqual(self.restore(main, {"source_lang": source, "target_lang": target}), (source, target))
            self.assertEqual(main.s_combo.currentText(), main.reverse_lang_mapping[source])
            self.assertEqual(main.t_combo.currentText(), main.reverse_lang_mapping[target])

    def test_localized_labels_and_legacy_aliases_roundtrip_in_every_catalog(self):
        for locale in ("de", "es", "fr", "it", "ja", "ko", "ru", "tr", "zh-CN"):
            with self.subTest(locale=locale):
                translator = QtCore.QTranslator()
                self.assertTrue(translator.load(str(self.catalog_dir / f"ct_{locale}.qm")))
                self.app.installTranslator(translator)
                try:
                    main = self.window()
                    for canonical, source in (("Portuguese", "Portuguese (Portugal)"),
                                              ("Brazilian Portuguese", "Portuguese (Brazil)")):
                        label = main.reverse_lang_mapping[canonical]
                        self.assertNotEqual(label, source)
                        for context in ("ComicTranslate", "ComicTranslateUI"):
                            self.assertEqual(translator.translate(context, source), label)
                        old_label = main.tr(canonical)
                        self.assertEqual(main.lang_mapping[old_label], canonical)
                        state = {"source_lang": old_label, "target_lang": old_label}
                        self.assertEqual(self.restore(main, state, seed_sticky_languages=True), (canonical, canonical))
                        self.assertEqual(main.s_combo.currentText(), label)
                        self.assertEqual(main.t_combo.currentText(), label)
                finally:
                    self.app.removeTranslator(translator)


if __name__ == "__main__":
    unittest.main()
