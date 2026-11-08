"""Synthetic Qt shaped-text tests for typography-only bubble masks."""

import os
from pathlib import Path
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPainter, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import QApplication

from app.ui.canvas.text.vertical_layout import VerticalTextDocumentLayout
from modules.rendering.bubble_layout import build_bubble_candidates, measure_bubble_layout


class BubbleLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if os.name == 'nt':
            path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'arial.ttf'
            if path.is_file():
                QFontDatabase.addApplicationFont(str(path))

    def document(self, text='Bonjour ! Je suis ravi de vous revoir.', size=20):
        document = QTextDocument()
        document.setDefaultFont(QFont('Arial', size))
        document.setPlainText(text)
        document.setTextWidth(160)
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        paragraph = cursor.blockFormat()
        paragraph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cursor.mergeBlockFormat(paragraph)
        return document

    @staticmethod
    def ellipse(height=180, width=280):
        y, x = np.ogrid[:height, :width]
        return ((x-width/2)/(width/2-8)) ** 2 + ((y-height/2)/(height/2-8)) ** 2 <= 1

    def build(self, document, mask, **changes):
        options = dict(mask=mask, font_size=20, minimum_font_size=10, maximum_font_size=32)
        options.update(changes)
        return build_bubble_candidates(document, **options)

    @staticmethod
    def restore(result):
        document = QTextDocument()
        document.setDefaultFont(result.default_font)
        document.setHtml(result.html)
        document.setTextWidth(result.width)
        return document

    def assert_raster_inside(self, result, mask, outline=0):
        """Independent pixel check of accepted output, not the integral algorithm."""
        document = self.restore(result)
        height, width = mask.shape
        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.translate(result.offset_x, result.offset_y)
        if outline:
            for dx in (-outline, 0, outline):
                for dy in (-outline, 0, outline):
                    painter.save()
                    painter.translate(dx, dy)
                    document.drawContents(painter)
                    painter.restore()
        document.drawContents(painter)
        painter.end()
        pixels = np.array(image.constBits()).reshape(height, image.bytesPerLine())[:, :width*4].reshape(height, width, 4)
        ink = pixels[:, :, 3] > 0
        self.assertTrue(np.any(ink), 'Expected visible rendered text')
        self.assertFalse(np.any(ink & ~mask), 'Accepted lettering painted outside the bubble mask')

    def test_round_bubble_preserves_text_and_final_raster_stays_inside(self):
        document = self.document()
        mask = self.ellipse()
        results = self.build(document, mask)
        self.assertEqual([result.key for result in results], ['current', 'balanced', 'larger'])
        for result in results[1:]:
            self.assertTrue(result.fits, result.overflow_reasons)
            self.assertEqual(self.restore(result).toPlainText(), document.toPlainText())
            self.assert_raster_inside(result, mask)

    def test_rectangular_caption_uses_corners_and_padding_from_supplied_mask(self):
        mask = np.zeros((160, 300), dtype=bool)
        mask[12:148, 12:288] = True
        result = self.build(self.document('A rectangular caption uses its available width.'), mask)[-1]
        self.assertTrue(result.fits)
        self.assert_raster_inside(result, mask)
        self.assertGreater(result.width, 150)

    def test_outlines_are_included_in_final_mask_containment(self):
        mask = self.ellipse(210, 310)
        result = self.build(self.document('À bientôt, mon ami !'), mask, outline_width=3)[-1]
        self.assertTrue(result.fits)
        self.assert_raster_inside(result, mask, outline=3)

    def test_hole_inside_line_is_rejected_even_when_all_four_corners_are_valid(self):
        document = self.document('AAAAAAAAAAAA')
        document.setTextWidth(250)
        mask = np.ones((90, 300), dtype=bool)
        mask[15:25, 110:120] = False
        result = measure_bubble_layout(document, mask=mask, current_position=(0, 0), font_size=20)
        self.assertFalse(result.fits)
        self.assertIn('boundary', result.overflow_reasons)

    def test_current_keeps_wrap_size_and_position_without_modifying_source(self):
        document = self.document('Tiếng Việt — ação.\nCảm ơn!')
        before = (document.toHtml(), document.textWidth(), document.revision())
        mask = self.ellipse(260, 330)
        saved_mask = mask.copy()
        results = self.build(document, mask, current_position=(27, 41))
        current = results[0]
        self.assertEqual((current.offset_x, current.offset_y), (27, 41))
        self.assertEqual(current.width, document.textWidth())
        self.assertEqual(current.font_size, 20)
        self.assertEqual(before, (document.toHtml(), document.textWidth(), document.revision()))
        np.testing.assert_array_equal(mask, saved_mask)

    def test_tailed_bubble_does_not_put_a_last_line_in_the_narrow_tail(self):
        mask = np.zeros((260, 300), dtype=bool)
        mask[:180] = self.ellipse(180, 300)
        mask[150:250, 140:160] = True
        result = self.build(self.document('A speech balloon should keep all of its words in the body.'), mask)[-1]
        self.assertTrue(result.fits)
        self.assertLess(result.offset_y + result.height, 185)
        self.assert_raster_inside(result, mask)

    def test_irregular_concave_bubble_allows_only_fully_contained_lines(self):
        mask = np.zeros((230, 330), dtype=bool)
        mask[10:220, 15:315] = True
        mask[10:75, 220:315] = False
        mask[160:220, 15:95] = False
        result = self.build(self.document('An irregular balloon still needs a clear and readable translation.'), mask)[-1]
        self.assertTrue(result.fits)
        self.assert_raster_inside(result, mask)

    def test_impossible_and_empty_masks_never_report_a_minimum_fit(self):
        document = self.document('An impossible fitting area.')
        for mask in (np.ones((5, 5), dtype=bool), np.zeros((80, 90), dtype=bool)):
            with self.subTest(shape=mask.shape):
                results = self.build(document, mask, minimum_font_size=18)
                self.assertTrue(all(not result.fits for result in results))
                self.assertTrue(all('boundary' in result.overflow_reasons for result in results))

    def test_rich_spans_and_manual_breaks_survive_candidates(self):
        document = self.document('😀 Bonjour mon ami!\nAção et Cảm ơn.')
        start = len('😀 Bonjour '.encode('utf-16-le')) // 2
        cursor = QTextCursor(document)
        cursor.setPosition(start)
        cursor.setPosition(start + 3, QTextCursor.MoveMode.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setFontWeight(QFont.Weight.Bold)
        fmt.setFontItalic(True)
        fmt.setFontUnderline(True)
        cursor.mergeCharFormat(fmt)
        result = self.build(document, self.ellipse(280, 380))[-1]
        self.assertTrue(result.fits)
        restored = self.restore(result)
        self.assertEqual(restored.toPlainText(), document.toPlainText())
        check = QTextCursor(restored)
        check.setPosition(start + 1)
        self.assertEqual(check.charFormat().fontWeight(), QFont.Weight.Bold)
        self.assertTrue(check.charFormat().fontItalic())
        self.assertTrue(check.charFormat().fontUnderline())
        joined = self.build(document, self.ellipse(280, 380), keep_manual_breaks=False)[-1]
        self.assertEqual(self.restore(joined).toPlainText(), document.toPlainText().replace('\n', ' '))
        self.assertEqual(self.restore(joined).characterCount(), document.characterCount())

    def test_long_unbreakable_word_remains_a_warning(self):
        results = self.build(self.document('Supercalifragilisticexpialidocious'), self.ellipse(240, 90),
                             minimum_font_size=18, maximum_font_size=20)
        self.assertTrue(all(not result.fits for result in results[1:]))
        self.assertIn('unbreakable_word', results[-1].overflow_reasons)

    def test_vertical_and_invalid_geometry_are_rejected_explicitly(self):
        document = self.document()
        for values in ({'mask': np.ones(10)}, {'mask': np.array([[float('nan')]])},
                       {'mask': self.ellipse(), 'current_position': (float('nan'), 0)}):
            with self.subTest(values=list(values)), self.assertRaises(ValueError):
                measure_bubble_layout(document, font_size=20, **values)
        document.setDocumentLayout(VerticalTextDocumentLayout(document, line_spacing=1.2))
        with self.assertRaises(ValueError):
            self.build(document, self.ellipse())


if __name__ == '__main__':
    unittest.main()
