import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.controllers.image import ImageStateController
from app.controllers.text import TextController


class _Combo:
    def __init__(self, text):
        self._text = text
        self._signals_blocked = False

    def currentText(self):
        return self._text

    def setCurrentText(self, text):
        self._text = text

    def blockSignals(self, blocked):
        previous = self._signals_blocked
        self._signals_blocked = blocked
        return previous


class _Radio:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _TextOption:
    def setTextDirection(self, direction):
        self.direction = direction


class _Document:
    def __init__(self):
        self.option = _TextOption()

    def defaultTextOption(self):
        return self.option

    def setDefaultTextOption(self, option):
        self.option = option


class _TextEdit:
    def __init__(self):
        self._document = _Document()

    def document(self):
        return self._document


def _main(*, manual=True, source="Spanish", target="English"):
    return SimpleNamespace(
        s_combo=_Combo(source),
        t_combo=_Combo(target),
        manual_radio=_Radio(manual),
        lang_mapping={
            "Auto": "Auto",
            "Spanish": "Spanish",
            "Japanese": "Japanese",
            "English": "English",
            "French": "French",
        },
        reverse_lang_mapping={
            "Auto": "Auto",
            "Spanish": "Spanish",
            "Japanese": "Japanese",
            "English": "English",
            "French": "French",
        },
    )


class PageLanguagePersistenceTests(unittest.TestCase):
    def test_manual_page_restore_keeps_current_source_language(self):
        main = _main(manual=True, source="Spanish")
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        destination_state = {"source_lang": "Auto", "target_lang": "French"}

        source, target = controller.restore_page_languages(destination_state)

        self.assertEqual(source, "Spanish")
        self.assertEqual(target, "English")
        self.assertEqual(main.s_combo.currentText(), "Spanish")
        self.assertEqual(main.t_combo.currentText(), "English")
        self.assertEqual(destination_state["source_lang"], "Spanish")
        self.assertEqual(destination_state["target_lang"], "English")

    def test_manual_target_language_sticks_across_pages_and_explicit_changes(self):
        main = _main(source="English", target="French")
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        for target in ("French", "Japanese", "English"):
            main.t_combo.setCurrentText(target)
            for state in ({"source_lang": "Auto", "target_lang": "English"}, {},
                          {"source_lang": "Spanish", "target_lang": "French"}):
                self.assertEqual(controller.restore_page_languages(state), ("English", target))
                self.assertEqual(main.t_combo.currentText(), target)
                self.assertEqual(state["target_lang"], target)
                self.assertFalse(main.t_combo._signals_blocked)

    def test_manual_auto_selection_is_sticky(self):
        main = _main(manual=True, source="Auto", target="French")
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        destination_state = {"source_lang": "Japanese", "target_lang": "English"}

        controller.restore_page_languages(destination_state)

        self.assertEqual(main.s_combo.currentText(), "Auto")
        self.assertEqual(destination_state["source_lang"], "Auto")
        self.assertEqual(main.t_combo.currentText(), "French")

    def test_automatic_mode_restores_each_pages_source_language(self):
        main = _main(manual=False, source="Spanish", target="French")
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        destination_state = {"source_lang": "Japanese", "target_lang": "English"}

        controller.restore_page_languages(destination_state)

        self.assertEqual(main.s_combo.currentText(), "Japanese")
        self.assertEqual(destination_state["source_lang"], "Japanese")
        self.assertEqual(main.t_combo.currentText(), "English")

    def test_project_load_seeds_both_manual_languages_from_initial_page(self):
        main = _main(manual=True, source="Spanish")
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        initial_state = {"source_lang": "Japanese", "target_lang": "French"}

        controller.restore_page_languages(initial_state, seed_sticky_languages=True)

        self.assertEqual(main.s_combo.currentText(), "Japanese")
        self.assertEqual(main.t_combo.currentText(), "French")
        self.assertEqual(controller.restore_page_languages(
            {"source_lang": "Auto", "target_lang": "English"}
        ), ("Japanese", "French"))

    def test_manual_language_pair_change_propagates_to_all_pages(self):
        main = _main(manual=True, source="English", target="French")
        main.image_files = ["page-1", "page-2"]
        main.image_states = {
            "page-1": {"source_lang": "Auto", "target_lang": "English"},
            "page-2": {"source_lang": "Japanese", "target_lang": "English"},
        }
        main.curr_img_idx = 0
        main.t_text_edit = _TextEdit()
        main.mark_project_dirty = lambda: None
        controller = TextController.__new__(TextController)
        controller.main = main

        controller.save_src_trg()

        self.assertEqual(main.image_states["page-1"]["source_lang"], "English")
        self.assertEqual(main.image_states["page-2"]["source_lang"], "English")
        self.assertEqual(main.image_states["page-1"]["target_lang"], "French")
        self.assertEqual(main.image_states["page-2"]["target_lang"], "French")

    def test_automatic_source_change_remains_page_specific(self):
        main = _main(manual=False, source="Spanish", target="English")
        main.image_files = ["page-1", "page-2"]
        main.image_states = {
            "page-1": {"source_lang": "Auto", "target_lang": "English"},
            "page-2": {"source_lang": "Japanese", "target_lang": "French"},
        }
        main.curr_img_idx = 0
        main.t_text_edit = _TextEdit()
        main.mark_project_dirty = lambda: None
        controller = TextController.__new__(TextController)
        controller.main = main

        controller.save_src_trg()

        self.assertEqual(main.image_states["page-1"]["source_lang"], "Spanish")
        self.assertEqual(main.image_states["page-2"]["source_lang"], "Japanese")


if __name__ == "__main__":
    unittest.main()
