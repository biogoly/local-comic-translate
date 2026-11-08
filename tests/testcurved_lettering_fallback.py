"""Native Qt fallback and shaping regressions using original fixture fonts."""
import os
import unittest
from collections import Counter
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text.curved_layout import CurvedTextDocumentLayout


class CurvedLetteringFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        base = Path(__file__).parent / 'fixtures' / 'curved_lettering'
        cls.font_ids = [QtGui.QFontDatabase.addApplicationFont(str(base / filename))
                        for filename in ('CurveTestPrimary.ttf', 'CurveTestFallback.ttf')]
        if any(font_id < 0 for font_id in cls.font_ids):
            raise RuntimeError('Could not load curved lettering fixture fonts')

    @classmethod
    def tearDownClass(cls):
        for font_id in cls.font_ids:
            QtGui.QFontDatabase.removeApplicationFont(font_id)

    def document(self, text, *, width=400, angle=75, html=False):
        doc = QtGui.QTextDocument()
        font = QtGui.QFont()
        font.setFamilies(['Curve Test Primary', 'Curve Test Fallback'])
        font.setPointSize(14)
        doc.setDefaultFont(font)
        doc.setTextWidth(width)
        if html:
            doc.setHtml(text)
        else:
            doc.setPlainText(text)
        cursor = QtGui.QTextCursor(doc)
        cursor.select(QtGui.QTextCursor.Document)
        block_format = QtGui.QTextBlockFormat()
        block_format.setAlignment(QtCore.Qt.AlignCenter)
        cursor.mergeBlockFormat(block_format)
        layout = CurvedTextDocumentLayout(doc, angle)
        doc.setDocumentLayout(layout)
        layout.update_layout()
        return doc, layout

    @staticmethod
    def glyphs(runs):
        return Counter((run.rawFont(), glyph, position.x(), position.y())
                       for run in runs
                       for glyph, position in zip(run.glyphIndexes(), run.positions()))

    def assert_native_glyphs_preserved(self, layout):
        for record in layout.lines:
            line = record.qt_line
            native = self.glyphs(line.glyphRuns(line.textStart(), line.textLength()))
            curved = self.glyphs(run for placement in record.placements for run in placement.runs)
            self.assertEqual(curved, native)

    def test_accent_fallback_does_not_freeze_rest_of_multiline_quote(self):
        quote = ('MA CHÉRIE! PUT AWAY THOSE\n\nHORRIBLE BOOKS AND COME\n\n'
                 'SHOW MY GUESTS THAT\n\nMULTIPLICATION TRICK!')
        for html in (False, True):
            for width in (400, 190):
                with self.subTest(html=html, width=width):
                    text = '<p>' + quote.replace('\n', '<br>') + '</p>' if html else quote
                    doc, layout = self.document(text, width=width, html=html)
                    self.assert_native_glyphs_preserved(layout)
                    placements = [placement for line in layout.lines for placement in line.placements]
                    for position, character in enumerate(doc.toPlainText()):
                        if not character.isspace():
                            matching = [p for p in placements if p.start <= position < p.end]
                            self.assertEqual(len(matching), 1, (position, character))
                            self.assertEqual(len(matching[0].runs), 1, (position, character))
                    first = next(p for p in placements if p.start == 0)
                    accent = next(p for p in placements if p.start == 5)
                    resumed = next(p for p in placements if p.start == 6)
                    self.assertEqual(accent.runs[0].rawFont().familyName(), 'Curve Test Fallback')
                    self.assertEqual(resumed.runs[0].rawFont().familyName(), 'Curve Test Primary')
                    self.assertGreater(abs(first.transform.m12()), 0.01)
                    self.assertGreater(abs(resumed.transform.m12()), 0.01)

    def test_native_ligatures_remain_single_clusters_without_duplicate_glyphs(self):
        doc, layout = self.document('office ffi fi fl')
        self.assert_native_glyphs_preserved(layout)
        placements = layout.lines[0].placements
        spans = [(p.start, p.end) for p in placements]
        for expected in ((1, 4), (7, 10), (11, 13), (14, 16)):
            self.assertIn(expected, spans)
            placement = next(p for p in placements if (p.start, p.end) == expected)
            self.assertEqual(len(placement.runs), 1)
        self.assertEqual(sum(len(p.runs) for p in placements), 10)

    def test_combining_marks_stay_with_their_base_across_font_fallback(self):
        doc, layout = self.document('A\u0301\u0301 B')
        self.assert_native_glyphs_preserved(layout)
        placements = layout.lines[0].placements
        first = next(p for p in placements if p.start == 0)
        self.assertEqual(first.end, 3)
        self.assertGreaterEqual(len(first.runs), 2)
        self.assertFalse(any(p.start in (1, 2) for p in placements))


if __name__ == '__main__':
    unittest.main()
