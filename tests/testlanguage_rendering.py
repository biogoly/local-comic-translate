"""Real Qt manual text, HTML persistence, and PNG export for Latin diacritics.

Set LCT_LANGUAGE_RENDER_OUTPUT to retain the PNG fixtures and coverage report.
These are supplied text fixtures, not OCR or live translation quality tests.
"""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import imkit as imk
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.image import ImageStateController
from app.projects.parsers import ProjectDecoder, ProjectEncoder
from modules.utils.textblock import TextBlock
from testmanual_text_box import _ManualTextMain


SAMPLES = {
    "Vietnamese": "Tiếng Việt: Xin chào! Cảm ơn.\nĐường phố yên tĩnh, những kỷ niệm đẹp.\nĂ Â Đ Ê Ô Ơ Ư — ă â đ ê ô ơ ư",
    "Portuguese": "Português (Portugal): Olá! Obrigado.\nA ação, o coração e a bênção.\nÀ Á Â Ã Ç É Ê Í Ó Ô Õ Ú Ü — à á â ã ç é ê í ó ô õ ú ü",
    "Brazilian Portuguese": "Português (Brasil): Olá! Obrigado.\nVocê vê o ônibus? Não, está na estação.\nÀ Á Â Ã Ç É Ê Í Ó Ô Õ Ú Ü — à á â ã ç é ê í ó ô õ ú ü",
}
# Vietnamese precomposed tone/vowel combinations, Portuguese diacritics, and
# decomposed combining marks. Coverage is checked on the actual selected font.
COVERAGE_TEXT = (
    "".join(SAMPLES.values())
    + "".join(chr(code) for code in range(0x1EA0, 0x1EFA))
    + unicodedata.normalize("NFD", "".join(SAMPLES.values()))
)
GLYPH_CHARACTERS = sorted(set(COVERAGE_TEXT) - {" ", "\n"})


def find_supported_font():
    installed = set(QtGui.QFontDatabase.families())
    candidates = ["Arial", "Segoe UI", "DejaVu Sans", "Noto Sans", "Liberation Sans"]
    candidates.extend(sorted(installed - set(candidates)))
    for family in candidates:
        font = QtGui.QFont(family, 22)
        if not QtGui.QFontInfo(font).family():
            continue
        raw = QtGui.QRawFont.fromFont(font)
        if raw.isValid() and all(raw.supportsCharacter(ord(char)) for char in GLYPH_CHARACTERS):
            return font, raw
    return None, None


class LanguageRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.application_font_id = -1
        # Qt's Windows offscreen plugin may have no system font database. Load
        # one existing OS font into this test process only; never install fonts
        # or change the user's application preferences.
        if os.name == "nt" and not QtGui.QFontDatabase.families():
            font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
            if font_path.is_file():
                cls.application_font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        cls.font, cls.raw_font = find_supported_font()
        if cls.font is None:
            if cls.application_font_id >= 0:
                QtGui.QFontDatabase.removeApplicationFont(cls.application_font_id)
            raise unittest.SkipTest("No installed font covers the Vietnamese/Portuguese test characters")

    @classmethod
    def tearDownClass(cls):
        cls.raw_font = None
        if cls.application_font_id >= 0:
            QtGui.QFontDatabase.removeApplicationFont(cls.application_font_id)

    def setUp(self):
        self.main = self.new_main()

    def new_main(self):
        main = _ManualTextMain()
        main.image_viewer.display_image_array(np.full((360, 1100, 3), 255, np.uint8), fit=False)
        main.font_dropdown.clear()
        main.font_dropdown.addItem(self.font.family())
        main.font_size_dropdown.clear()
        main.font_size_dropdown.addItem("22")
        for language in SAMPLES:
            main.t_combo.addItem(language)
            main.lang_mapping[language] = language
        return main

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def create_item(self, language, text):
        self.main.t_combo.setCurrentText(language)
        block = TextBlock(
            text_bbox=np.array([25, 25, 1075, 330]),
            translation=text,
            target_lang=language,
            is_manual=True,
        )
        self.main.blk_list.append(block)
        item = self.main.text_ctrl.create_manual_text_item(block, text)
        self.app.processEvents()
        self.assertIsNotNone(item)
        self.assertEqual(item.toPlainText(), text)
        self.assertFalse(item.vertical)
        self.main.image_viewer.deselect_all()
        return item

    def test_font_covers_vietnamese_portuguese_and_combining_marks(self):
        self.assertTrue(all(self.raw_font.supportsCharacter(ord(char)) for char in GLYPH_CHARACTERS))
        for sample in SAMPLES.values():
            self.assertNotIn(0, self.raw_font.glyphIndexesForString(sample.replace("\n", " ")))

    def test_manual_text_html_and_canvas_state_preserve_unicode(self):
        # Separate windows avoid overlapping manual items and exercise the real
        # canvas serializer/deserializer, without reading application preferences.
        for language, sample in SAMPLES.items():
            for text in (sample, unicodedata.normalize("NFD", sample)):
                with self.subTest(language=language, normalization="NFC" if text == sample else "NFD"):
                    item = self.create_item(language, text)
                    html = item.document().toHtml()
                    document = QtGui.QTextDocument()
                    document.setHtml(html)
                    self.assertEqual(document.toPlainText(), text)
                    state = json.loads(
                        json.dumps(self.main.image_viewer.save_state(), ensure_ascii=False, default=ProjectEncoder().encode),
                        object_hook=ProjectDecoder().decode,
                    )
                    restored = self.new_main()
                    try:
                        restored.image_viewer.load_state(state)
                        self.app.processEvents()
                        self.assertEqual(restored.image_viewer.text_items[0].toPlainText(), text)
                        before = self.main.image_viewer.get_image_array(paint_all=True)
                        after = restored.image_viewer.get_image_array(paint_all=True)
                        np.testing.assert_array_equal(before, after)
                    finally:
                        restored.deleteLater()
                    self.main.deleteLater()
                    self.app.processEvents()
                    self.main = self.new_main()

    def test_real_manual_text_scene_exports_png_with_visible_accents(self):
        retained = os.environ.get("LCT_LANGUAGE_RENDER_OUTPUT")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(retained or temporary)
            output_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "kind": "real Qt manual text and PNG export; supplied text, no inference",
                "requested_font": self.font.family(),
                "resolved_font": QtGui.QFontInfo(self.font).family(),
                "glyph_characters_checked": len(GLYPH_CHARACTERS),
                "missing_characters": [],
                "samples": [],
            }
            for language, text in SAMPLES.items():
                with self.subTest(language=language):
                    item = self.create_item(language, text)
                    expected = self.main.image_viewer.get_image_array(paint_all=True)
                    self.assertEqual(expected.shape, (360, 1100, 3))
                    self.assertGreater(int(np.count_nonzero(np.any(expected < 128, axis=2))), 500)
                    path = output_dir / (language.lower().replace(" ", "-") + ".png")
                    # The production export method only needs its main reference.
                    ImageStateController.save_current_image(SimpleNamespace(main=self.main), str(path))
                    self.assertTrue(path.is_file())
                    decoded = imk.read_image(str(path))
                    np.testing.assert_array_equal(decoded, expected)

                    # The same production text item rendered without diacritics
                    # must differ, proving the accented text was not stripped.
                    unaccented = "".join(
                        char for char in unicodedata.normalize("NFD", text)
                        if not unicodedata.combining(char)
                    )
                    item.set_text(unaccented, item.textWidth())
                    self.app.processEvents()
                    plain = self.main.image_viewer.get_image_array(paint_all=True)
                    changed = int(np.count_nonzero(np.any(decoded != plain, axis=2)))
                    self.assertGreater(changed, 100)
                    report["samples"].append({
                        "language": language,
                        "file": path.name,
                        "width": decoded.shape[1],
                        "height": decoded.shape[0],
                        "dark_pixels": int(np.count_nonzero(np.any(decoded < 128, axis=2))),
                        "pixels_different_from_unaccented": changed,
                        "png_lossless_round_trip": True,
                    })
                    self.main.deleteLater()
                    self.app.processEvents()
                    self.main = self.new_main()
            if retained:
                (output_dir / "qt-rendering-report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )


if __name__ == "__main__":
    unittest.main()
