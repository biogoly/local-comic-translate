"""Actual Qt layout contract tests; no OCR, LLM or image-generation calls."""

import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import QApplication

from modules.rendering.layout import build_layout_candidates, measure_layout
from modules.rendering.render import pyside_word_wrap, WordWrapResult
from app.ui.canvas.text_item import TextBlockItem


class TextLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if os.name == "nt":
            path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
            if path.is_file():
                QFontDatabase.addApplicationFont(str(path))

    def document(self, text, size=20):
        doc = QTextDocument()
        doc.setDefaultFont(QFont("Arial", size))
        doc.setPlainText(text)
        doc.setTextWidth(240)
        return doc

    def candidates(self, doc, **changes):
        options = dict(width=240, height=180, font_size=20,
                       minimum_font_size=10, maximum_font_size=32)
        options.update(changes)
        return build_layout_candidates(doc, **options)

    @staticmethod
    def restored(candidate):
        doc = QTextDocument()
        doc.setDefaultFont(candidate.default_font)
        doc.setHtml(candidate.html)
        doc.setTextWidth(candidate.width)
        return doc

    def test_current_preserves_wrap_width_and_source_document_is_unchanged(self):
        doc = self.document("Hello there. A longer sentence to wrap naturally.")
        doc.setTextWidth(310)
        before = (doc.toHtml(), doc.textWidth(), doc.revision(), doc.defaultFont())
        result = self.candidates(doc, width=190)
        self.assertAlmostEqual(result[0].width, 310)
        self.assertEqual(before, (doc.toHtml(), doc.textWidth(), doc.revision(), doc.defaultFont()))

    def test_measure_only_reports_same_current_without_search_or_mutation(self):
        doc = self.document("A somewhat longer sentence for the current layout.")
        before = (doc.toHtml(), doc.textWidth(), doc.revision())
        measured = measure_layout(doc, width=240, height=80, font_size=20)
        searched = self.candidates(doc, height=80)[0]
        self.assertEqual(measured.fits, searched.fits)
        self.assertEqual(measured.overflow_reasons, searched.overflow_reasons)
        self.assertEqual(measured.html, searched.html)
        self.assertEqual(before, (doc.toHtml(), doc.textWidth(), doc.revision()))

    def test_readable_fit_preserves_text_and_newlines(self):
        text = "Bonjour ! Je suis ravi de vous revoir.\nMerci beaucoup."
        doc = self.document(text)
        for candidate in self.candidates(doc, height=260):
            self.assertEqual(self.restored(candidate).toPlainText(), text)
        self.assertTrue(any(item.fits for item in self.candidates(doc, height=260)[1:]))

    def test_minimum_size_fallback_is_not_a_success(self):
        doc = self.document("A whole paragraph that cannot fit into a very small box.")
        alternatives = self.candidates(doc, width=35, height=12, minimum_font_size=18)[1:]
        self.assertTrue(all(not item.fits for item in alternatives))
        self.assertTrue(all("height" in item.overflow_reasons for item in alternatives))

    def test_forced_break_inside_long_word_is_reported(self):
        doc = self.document("Supercalifragilisticexpialidocious")
        candidates = self.candidates(doc, width=40, height=1000,
                                     minimum_font_size=18, maximum_font_size=20)
        self.assertTrue(all(not item.fits for item in candidates[1:]))
        self.assertTrue(all("unbreakable_word" in item.overflow_reasons for item in candidates[1:]))

    def test_selected_styles_colors_and_utf16_offsets_survive_size_change(self):
        text = "😀 Cảm ơn, ação — merci beaucoup!"
        doc = self.document(text)
        start = len("😀 Cảm ơn, ".encode("utf-16-le")) // 2
        end = start + len("ação")
        cursor = QTextCursor(doc)
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        style = QTextCharFormat()
        style.setFontWeight(QFont.Weight.Bold)
        style.setFontItalic(True)
        style.setFontUnderline(True)
        style.setForeground(Qt.GlobalColor.red)
        style.setFontPointSize(24)
        cursor.mergeCharFormat(style)
        candidate = self.candidates(doc, width=400, height=220)[-1]
        self.assertTrue(candidate.fits)
        restored = self.restored(candidate)
        self.assertEqual(restored.toPlainText(), text)
        check = QTextCursor(restored)
        check.setPosition(start)
        check.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        fmt = check.charFormat()
        self.assertEqual(check.selectedText(), "ação")
        self.assertEqual(fmt.fontWeight(), QFont.Weight.Bold)
        self.assertTrue(fmt.fontItalic())
        self.assertTrue(fmt.fontUnderline())
        self.assertEqual(fmt.foreground().color(), Qt.GlobalColor.red)
        self.assertAlmostEqual(fmt.fontPointSize(), 24 * candidate.font_size / 20, places=2)

    def test_manual_break_reflow_is_explicit_and_preserves_utf16_positions(self):
        doc = self.document("😀 Hello\nWorld\nLast")
        original = doc.toPlainText()
        keep = self.candidates(doc, keep_manual_breaks=True)[-1]
        reflow = self.candidates(doc, keep_manual_breaks=False)[-1]
        self.assertEqual(self.restored(keep).toPlainText(), original)
        self.assertEqual(self.restored(reflow).toPlainText(), "😀 Hello World Last")
        self.assertEqual(self.restored(reflow).characterCount(), doc.characterCount())
        self.assertEqual(doc.toPlainText(), original)

    def test_outline_space_is_measured(self):
        doc = self.document("Hi")
        doc.setTextWidth(60)
        plain = self.candidates(doc, width=60, height=50, outline_width=0)[0]
        outlined = self.candidates(doc, width=60, height=50, outline_width=12)[0]
        self.assertTrue(plain.fits)
        self.assertFalse(outlined.fits)
        self.assertIn("width", outlined.overflow_reasons)
        self.assertGreater(outlined.content_width, plain.content_width)

    def test_japanese_no_space_and_rtl_are_not_rewritten(self):
        for text in ("ありがとう。今日はいい天気ですね。", "مرحبا بكم في هذا المكان الجميل"):
            with self.subTest(text=text):
                doc = self.document(text)
                if text.startswith("مرحبا"):
                    option = doc.defaultTextOption()
                    option.setTextDirection(Qt.LayoutDirection.RightToLeft)
                    doc.setDefaultTextOption(option)
                candidates = self.candidates(doc, width=380, height=300)
                for candidate in candidates:
                    self.assertEqual(self.restored(candidate).toPlainText(), text)
                self.assertTrue(any(item.fits for item in candidates[1:]))

    def test_vertical_clone_preserves_source_layout_and_detects_overflow(self):
        item = TextBlockItem(font_family="Arial", font_size=20, outline_color=None)
        item.set_text("日本語の縦書きです。", 150)
        item.set_vertical(True)
        item.layout.set_max_size(150, 180)
        before = (item.toHtml(), item.layout.max_width, item.layout.max_height)
        good = self.candidates(item.document(), width=150, height=180, vertical=True)
        tiny = self.candidates(item.document(), width=12, height=15, vertical=True,
                              minimum_font_size=18, maximum_font_size=20)
        self.assertEqual(before, (item.toHtml(), item.layout.max_width, item.layout.max_height))
        self.assertTrue(any(result.fits for result in good[1:]))
        self.assertTrue(all(not result.fits for result in tiny[1:]))
        for result in good:
            self.assertEqual(self.restored(result).toPlainText(), item.toPlainText())

    def test_below_preferred_warning_does_not_invent_overflow(self):
        doc = self.document("Hi", size=8)
        result = self.candidates(doc, font_size=8, minimum_font_size=12)[0]
        self.assertTrue(result.fits)
        self.assertTrue(result.below_minimum)

    def test_readability_includes_small_inline_words_and_inherited_font(self):
        doc = self.document("A small word", size=20)
        cursor = QTextCursor(doc)
        cursor.setPosition(2)
        cursor.setPosition(7, QTextCursor.MoveMode.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setFontPointSize(8)
        cursor.mergeCharFormat(fmt)
        result = measure_layout(doc, width=300, height=100, font_size=20, minimum_font_size=10)
        self.assertTrue(result.fits)
        self.assertTrue(result.below_minimum)
        inherited = self.document("Inherited small text", size=8)
        inherited_result = measure_layout(inherited, width=300, height=100,
                                           font_size=20, minimum_font_size=10)
        self.assertTrue(inherited_result.below_minimum)

    def test_tables_are_not_silently_flattened(self):
        doc = self.document("")
        doc.setHtml("<table><tr><td>Keep me</td></tr></table>")
        result = self.candidates(doc)
        self.assertEqual(len(result), 1)
        self.assertIn("unsupported_content", result[0].overflow_reasons)
        self.assertIn("Keep me", self.restored(result[0]).toPlainText())

    def test_invalid_constraints_fail_before_mutation(self):
        doc = self.document("No mutation")
        original = doc.toHtml()
        for options in ({"width": 0}, {"height": float("nan")},
                        {"outline_width": -1}, {"maximum_font_size": 5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.candidates(doc, **options)
        self.assertEqual(doc.toHtml(), original)

    def test_plain_wrap_opt_in_diagnostics_preserve_legacy_tuple_api(self):
        options = dict(text="A paragraph which simply cannot fit into this tiny rectangle.",
                       font_input="Arial", roi_width=15, roi_height=10, line_spacing=1.2,
                       outline_width=0, bold=False, italic=False, underline=False,
                       alignment=Qt.AlignmentFlag.AlignCenter,
                       direction=Qt.LayoutDirection.LeftToRight, init_font_size=20,
                       min_font_size=18)
        original = pyside_word_wrap(**options)
        metrics = pyside_word_wrap(**options, return_metrics=True)
        diagnostic = pyside_word_wrap(**options, return_diagnostics=True)
        self.assertEqual(len(original), 2)
        self.assertEqual(len(metrics), 4)
        self.assertIsInstance(diagnostic, WordWrapResult)
        self.assertEqual(original, (diagnostic.text, diagnostic.font_size))
        self.assertEqual(metrics[2:], (diagnostic.width, diagnostic.height))
        self.assertFalse(diagnostic.fits)
        self.assertIn("height", diagnostic.overflow_reasons)


if __name__ == "__main__":
    unittest.main()
