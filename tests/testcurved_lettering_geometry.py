"""Real Qt shaping, arc rendering, bounds, and source-document preservation."""
import copy
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.curved_layout import CurvedTextDocumentLayout


class CurvedLetteringGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def item(self, text='ARTISTIC LETTERING', angle=75, width=400):
        item = TextBlockItem(text, font_family='Arial', font_size=24)
        item.set_font('Arial', 24)
        item.set_alignment(QtCore.Qt.AlignCenter)
        item.setTextWidth(width)
        item.set_lettering_transform({'arc_angle': angle})
        return item

    def pixels(self, item, outline=False):
        image = QtGui.QImage(700, 400, QtGui.QImage.Format_RGBA8888)
        image.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(image)
        painter.translate(20, 20)
        if outline:
            item.paint(painter, QtWidgets.QStyleOptionGraphicsItem())
        else:
            item.document().drawContents(painter)
        painter.end()
        return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(400, 700, 4).copy()

    def test_signed_arc_and_shared_caret_hit_geometry(self):
        for angle in (-120, -55, 55, 120):
            item = self.item(angle=angle)
            carets = item.layout.lines[0].carets
            start, end = carets[0][1], carets[-1][1]
            middle = carets[len(carets) // 2][1]
            edge_y = (start.center().y() + end.center().y()) / 2
            self.assertGreater((edge_y - middle.center().y()) * angle, 0)
            for position, caret in carets:
                self.assertEqual(item.layout.hitTest(caret.center(), QtCore.Qt.FuzzyHit), position)

    def test_near_zero_curve_does_not_add_document_margins(self):
        item = self.item(angle=0)
        before = item.boundingRect()
        item.set_lettering_transform({'arc_angle': 0.00001})
        after = item.boundingRect()
        self.assertAlmostEqual(after.width(), before.width(), places=4)
        self.assertAlmostEqual(after.height(), before.height(), places=4)

    def test_each_wrapped_line_keeps_its_source_indices(self):
        item = self.item('A long translated title\nWith another line and several words', width=190)
        self.assertGreater(len(item.layout.lines), 2)
        seen = []
        for record in item.layout.lines:
            seen.extend(position for position, _ in record.carets)
            midpoint = record.carets[len(record.carets) // 2]
            self.assertEqual(item.layout.hitTest(midpoint[1].center(), QtCore.Qt.FuzzyHit), midpoint[0])
        self.assertEqual(max(seen), item.document().characterCount() - 1)
        self.assertEqual(item.toPlainText(), 'A long translated title\nWith another line and several words')

    def test_graphemes_utf16_and_rtl_preserve_shaped_glyphs(self):
        text = 'Việt e\u0301 中文 العربية 😀'
        item = self.item(text, width=600)
        boundaries = set(CurvedTextDocumentLayout._boundaries(text))
        carets = {p for record in item.layout.lines for p, _ in record.carets}
        self.assertEqual(carets, boundaries)
        # Combining mark and surrogate pair never gain an internal mouse caret.
        self.assertNotIn(6, carets)
        self.assertNotIn(len(text.encode('utf-16-le')) // 2 - 1, carets)
        for record in item.layout.lines:
            expected = sum(len(run.glyphIndexes()) for run in record.qt_line.glyphRuns())
            actual = sum(len(run.glyphIndexes()) for placement in record.placements for run in placement.runs)
            self.assertEqual(actual, expected)
        self.assertEqual(item.toPlainText(), text)

    def test_rich_word_formats_and_colors_stay_in_source_and_paint(self):
        item = self.item('A colored word')
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(2)
        cursor.setPosition(9, QtGui.QTextCursor.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor('#cc1122'))
        fmt.setFontWeight(QtGui.QFont.Bold)
        fmt.setFontItalic(True)
        fmt.setFontUnderline(True)
        cursor.mergeCharFormat(fmt)
        source = item.toHtml()
        item.set_lettering_transform({'arc_angle': -85})
        self.assertEqual(item.toHtml(), source)
        colored = [p for record in item.layout.lines for p in record.placements if 2 <= p.start < 9]
        self.assertTrue(colored)
        self.assertTrue(all(p.format.fontWeight() == QtGui.QFont.Bold and p.format.fontItalic()
                            and p.format.fontUnderline() for p in colored))
        pixels = self.pixels(item)
        self.assertTrue(((pixels[..., 0] > 150) & (pixels[..., 1] < 80) & (pixels[..., 2] < 100)).any())

    def test_bidi_boundary_cluster_does_not_span_the_rtl_word(self):
        item = self.item('Hello العربية end', width=500)
        placements = {p.start: p for p in item.layout.lines[0].placements}
        # The first and last Arabic letters touch ambiguous bidi cursor
        # boundaries. Their painted cluster cells must remain letter-sized.
        for position in (6, 12):
            placement = placements[position]
            advances = [run.rawFont().advancesForGlyphIndexes(run.glyphIndexes())[0].x()
                        for run in placement.runs]
            self.assertLessEqual(placement.cell.width(), sum(advances) + 0.01)

    def test_outline_clone_has_identical_curve_positions(self):
        item = self.item('Curved outline')
        item.set_outline(QtGui.QColor('#00aa00'), 2)
        clone = item._clone_outline_document()
        self.assertIsInstance(clone.documentLayout(), CurvedTextDocumentLayout)
        self.assertEqual(clone.textWidth(), item.logical_text_width())
        for position in (0, 3, item.document().characterCount() - 1):
            self.assertEqual(clone.documentLayout().caret_line(position), item.layout.caret_line(position))
        pixels = self.pixels(item, outline=True)
        self.assertTrue(((pixels[..., 1] > 100) & (pixels[..., 0] < 100) & (pixels[..., 2] < 100)).any())

    def test_all_shaped_ink_and_carets_fit_reported_bounds(self):
        for angle in (-120, 120):
            item = self.item('flying Áj\nItalic edge', angle)
            item.set_italic(True)
            bounds = item.boundingRect().adjusted(-0.001, -0.001, 0.001, 0.001)
            for record in item.layout.lines:
                for _, caret in record.carets:
                    self.assertTrue(bounds.contains(caret.p1()))
                    self.assertTrue(bounds.contains(caret.p2()))
                for placement in record.placements:
                    for run in placement.runs:
                        ink = run.rawFont().boundingRect(run.glyphIndexes()[0]).translated(run.positions()[0])
                        if not ink.isEmpty():
                            self.assertTrue(bounds.contains(placement.transform.mapRect(ink)))

    def test_empty_retype_keeps_curve_font_and_finite_geometry(self):
        item = self.item('Words')
        cursor = QtGui.QTextCursor(item.document())
        cursor.select(QtGui.QTextCursor.Document)
        cursor.removeSelectedText()
        self.assertTrue(item.boundingRect().isValid())
        item.setTextCursor(cursor)
        item.set_plain_text('Retyped')
        self.assertEqual(item.toPlainText(), 'Retyped')
        self.assertEqual(item.lettering_transform['arc_angle'], 75)
        self.assertTrue(item.layout.lines)

    def test_reset_restores_native_layout_and_exact_rich_text(self):
        item = self.item('Reset this')
        source, width = item.toHtml(), item.textWidth()
        item.set_lettering_transform({})
        self.assertNotIsInstance(item.document().documentLayout(), CurvedTextDocumentLayout)
        self.assertIsNone(item.layout)
        self.assertEqual(item.toHtml(), source)
        self.assertEqual(item.textWidth(), width)

    def test_vertical_rejects_curve_without_partial_change(self):
        item = self.item('水平')
        before = item.toHtml()
        with self.assertRaises(ValueError):
            item.set_vertical(True)
        self.assertFalse(item.vertical)
        self.assertEqual(item.toHtml(), before)
        item.set_lettering_transform({})
        item.set_vertical(True)
        with self.assertRaises(ValueError):
            item.set_lettering_transform({'arc_angle': 40})
        self.assertEqual(item.lettering_transform, {})

    def test_copy_owns_its_layout_and_logical_width(self):
        item = self.item('Independent layout')
        clone = copy.copy(item)
        self.assertIsNot(clone.layout, item.layout)
        self.assertIs(clone.layout.document(), clone.document())
        self.assertEqual(clone.logical_text_width(), item.logical_text_width())
        clone.set_lettering_transform({'arc_angle': -50})
        clone.set_plain_text('Changed')
        self.assertEqual(item.toPlainText(), 'Independent layout')
        self.assertEqual(item.layout.arc_angle, 75)


if __name__ == '__main__':
    unittest.main()
