"""Real Qt typography on synthetic source bubbles; no OCR/model calls."""

import copy
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QRawFont, QTextCursor, QTextDocument
from PySide6.QtWidgets import QApplication

from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType, TextBlockItem
from modules.rendering.auto_bubble_fit import fit_new_text_item, load_original_for_fitting
from modules.rendering.bubble_boundary import estimate_bubble_boundary
from modules.utils.textblock import TextBlock


class AutomaticBubbleFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if os.name == 'nt':
            font_directory = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
            for name in ('arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf', 'segoeui.ttf'):
                path = font_directory / name
                if path.is_file():
                    QFontDatabase.addApplicationFont(str(path))

    def setUp(self):
        self.settings = SimpleNamespace(auto_bubble_fit=True, bubble_padding=8,
                                        min_font_size=10, max_font_size=28)
        image = Image.new('RGB', (400, 260), (140, 95, 70))
        draw = ImageDraw.Draw(image)
        draw.ellipse((30, 30, 369, 229), fill=(215, 245, 250), outline=(0, 0, 0), width=3)
        draw.rectangle((150, 100, 167, 111), fill=(0, 0, 0))
        draw.rectangle((181, 100, 198, 111), fill=(0, 0, 0))
        draw.rectangle((176, 130, 206, 141), fill=(0, 0, 0))
        self.original = np.array(image)
        self.original.setflags(write=False)
        self.block = TextBlock(text_bbox=np.array([130, 90, 270, 150]),
                               bubble_bbox=np.array([30, 30, 370, 230]),
                               text='Original OCR text', translation='Bonjour mon ami !',
                               text_segm_points=np.array([[140, 100], [220, 140]]))
        self.properties = TextItemProperties(text='Bonjour mon\nami !', font_family='Arial',
                                             font_size=18, text_color=QColor('black'),
                                             position=(130, 90), width=140, height=60,
                                             region_id=self.block.region_id)

    def fit(self, **kwargs):
        return fit_new_text_item(self.properties, self.block, self.original, self.settings, **kwargs)

    def restore(self, properties):
        document = QTextDocument()
        font = QFont()
        self.assertTrue(font.fromString(properties.document_font))
        document.setDefaultFont(font)
        document.setHtml(properties.text)
        document.setTextWidth(properties.width)
        return document

    def assert_fitted(self, properties):
        self.assertEqual(properties.layout_preferences.get('mode'), 'bubble')
        self.assertEqual(properties.layout_preferences['boundary_shape'], 'auto')
        self.assertEqual(properties.region_id, self.properties.region_id)
        self.assertGreaterEqual(properties.font_size, self.settings.min_font_size)
        self.assertLessEqual(properties.font_size, self.settings.max_font_size)

    def assert_raster_inside(self, properties, outline=0):
        boundary = estimate_bubble_boundary(self.original, self.block.xyxy,
                                             self.block.bubble_xyxy, padding=self.settings.bubble_padding)
        self.assertTrue(boundary.available, boundary.reason)
        mask = np.zeros(self.original.shape[:2], dtype=bool)
        x, y = boundary.origin
        height, width = boundary.mask.shape
        mask[y:y+height, x:x+width] = boundary.mask
        document = self.restore(properties)
        image = QImage(self.original.shape[1], self.original.shape[0], QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.translate(*properties.position)
        for dx in (-outline, 0, outline):
            for dy in (-outline, 0, outline):
                painter.save()
                painter.translate(dx, dy)
                document.drawContents(painter)
                painter.restore()
        painter.end()
        pixels = np.array(image.constBits()).reshape(image.height(), image.bytesPerLine())[:, :image.width()*4]
        ink = pixels.reshape(image.height(), image.width(), 4)[:, :, 3] > 0
        self.assertTrue(ink.any())
        self.assertFalse(np.any(ink & ~mask))

    def test_real_fonts_logical_text_manual_breaks_and_source_immutability(self):
        # Explicitly loaded fonts avoid passing on Windows sandbox tofu metrics.
        raw = QRawFont.fromFont(QFont('Arial', 18))
        self.assertTrue(raw.isValid())
        self.assertTrue(all(raw.glyphIndexesForString('Bonjour ação Tiếng Việt')))
        before = copy.deepcopy(self.properties)
        pixels = self.original.copy()
        bounds, segmentation = self.block.xyxy.copy(), self.block.segm_pts.copy()
        result = self.fit(source_text='Bonjour mon ami !\nMerci beaucoup.')
        self.assert_fitted(result)
        self.assertEqual(self.restore(result).toPlainText(), 'Bonjour mon ami !\nMerci beaucoup.')
        self.assertEqual(self.properties, before)
        np.testing.assert_array_equal(self.original, pixels)
        np.testing.assert_array_equal(self.block.xyxy, bounds)
        np.testing.assert_array_equal(self.block.segm_pts, segmentation)
        self.assertEqual(self.block.text, 'Original OCR text')
        self.assertEqual(self.block.translation, 'Bonjour mon ami !')
        self.assert_raster_inside(result)

    def test_untranslated_class_does_not_disable_valid_bubble_fitting(self):
        self.block.text_class = 'text_free'
        self.assert_fitted(self.fit())

    def test_disabled_never_estimates_or_changes_properties(self):
        self.settings.auto_bubble_fit = False
        with patch('modules.rendering.auto_bubble_fit.estimate_bubble_boundary', side_effect=AssertionError):
            result = self.fit()
        self.assertEqual(result, self.properties)
        self.assertIsNot(result, self.properties)
        result.layout_preferences['mode'] = 'bubble'
        self.assertEqual(self.properties.layout_preferences, {})

    def test_missing_image_or_uncertain_boundary_keeps_box(self):
        result = fit_new_text_item(self.properties, self.block, None, self.settings)
        self.assertEqual(result, self.properties)
        self.original = np.full_like(self.original, 230)
        self.assertEqual(self.fit(), self.properties)

    def test_rotation_vertical_scale_and_page_seams_keep_box(self):
        for name, value in (('rotation', 5), ('vertical', True), ('scale', 1.2)):
            with self.subTest(name=name):
                original = getattr(self.properties, name)
                setattr(self.properties, name, value)
                self.assertEqual(self.fit(), self.properties)
                setattr(self.properties, name, original)
        self.block.angle = -8
        self.assertEqual(self.fit(), self.properties)
        self.block.angle = 0
        for bounds in ((130, -10, 270, 150), (130, 90, 270, 280)):
            self.block.xyxy = np.array(bounds)
            self.assertEqual(self.fit(), self.properties)

    def test_existing_layout_and_protection_are_retained(self):
        self.properties.layout_preferences = {'mode': 'box', 'bound_width': 140}
        self.assertEqual(self.fit(), self.properties)
        self.properties.layout_preferences = {}
        self.block.region_metadata['protections']['keep_layout'] = True
        self.assertEqual(self.fit(), self.properties)

    def test_no_readable_fit_keeps_original_box_instead_of_shrinking(self):
        self.properties.text = ' '.join(['A long sentence that cannot fit.'] * 80)
        self.settings.min_font_size = self.settings.max_font_size = 24
        self.assertEqual(self.fit(), self.properties)

    def test_small_inline_span_cannot_bypass_minimum_readability(self):
        self.properties.text = '<p>Bonjour <span style="font-size:6pt">ami</span> !</p>'
        self.properties.document_font = QFont('Arial', 18).toString()
        # Even maximum base size 28 leaves this relative span below 10 pt.
        self.assertEqual(self.fit(), self.properties)

    def test_rich_spans_defaults_and_partial_utf16_outline_are_preserved(self):
        self.properties.text = '<p>Olá <b>mundo</b> e <i>bonjour</i> <u>amigos</u> 😀.</p>'
        self.properties.document_font = QFont('Arial', 18).toString()
        self.properties.selection_outlines = [OutlineInfo(4, 9, QColor('blue'), 1.5, OutlineType.Selection)]
        # Plain source cannot overwrite a supplied rich document.
        result = self.fit(source_text='Must not replace the rich content')
        self.assert_fitted(result)
        document = self.restore(result)
        self.assertEqual(document.toPlainText(), 'Olá mundo e bonjour amigos 😀.')
        for word, check in (('mundo', lambda f: f.fontWeight() >= QFont.Weight.Bold),
                            ('bonjour', lambda f: f.fontItalic()), ('amigos', lambda f: f.fontUnderline())):
            cursor = document.find(word)
            self.assertTrue(check(cursor.charFormat()), word)
        self.assertEqual(result.selection_outlines, self.properties.selection_outlines)
        self.assertIsNot(result.selection_outlines[0], self.properties.selection_outlines[0])
        self.assert_raster_inside(result, outline=1.5)

    def test_full_outline_is_reindexed_in_utf16_and_missing_metadata_is_created(self):
        logical = 'Olá 😀 mundo!\nTiếng Việt.'
        self.properties.outline_color = QColor('white')
        self.properties.outline_width = 2
        self.properties.selection_outlines = [OutlineInfo(0, 1, QColor('white'), 2, OutlineType.Full_Document)]
        result = self.fit(source_text=logical)
        self.assert_fitted(result)
        self.assertEqual(result.selection_outlines[0].end, len(logical.encode('utf-16-le')) // 2)
        self.assertEqual(self.properties.selection_outlines[0].end, 1)
        self.assert_raster_inside(result, outline=2)
        self.properties.selection_outlines = []
        result = self.fit(source_text=logical)
        self.assertEqual(len(result.selection_outlines), 1)
        self.assertEqual(result.selection_outlines[0].type, OutlineType.Full_Document)

    def test_partial_outline_with_changed_text_falls_back_without_corrupting_offsets(self):
        self.properties.selection_outlines = [OutlineInfo(2, 9, QColor('white'), 1, OutlineType.Selection)]
        self.assertEqual(self.fit(source_text='Bonjour mon ami !'), self.properties)

    def test_persisted_font_and_html_recreate_actual_canvas_metrics(self):
        self.properties.bold = True
        self.properties.italic = True
        self.properties.underline = True
        result = self.fit(source_text='Bonjour mon ami !')
        self.assert_fitted(result)
        document = self.restore(result)
        item = TextBlockItem(text=result.text, font_family=result.font_family, font_size=result.font_size,
                             render_color=result.text_color, alignment=result.alignment,
                             line_spacing=result.line_spacing, bold=result.bold,
                             italic=result.italic, underline=result.underline)
        saved = QFont()
        saved.fromString(result.document_font)
        item.document().setDefaultFont(saved)
        item.set_text(result.text, result.width)
        self.assertEqual(item.toPlainText(), document.toPlainText())
        self.assertAlmostEqual(item.document().size().width(), document.size().width(), places=3)
        self.assertAlmostEqual(item.document().size().height(), document.size().height(), places=3)
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        self.assertTrue(cursor.charFormat().fontItalic())
        self.assertTrue(cursor.charFormat().fontUnderline())

    def test_can_build_in_background_thread_without_widget_creation(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(self.fit, source_text='Bonjour, mon ami !').result(timeout=10)
        self.assert_fitted(result)

    def test_original_loader_reads_rgb_source_and_returns_independent_readonly_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'original.png')
            Image.fromarray(self.original).save(path)
            original = load_original_for_fitting(path, expected_shape=self.original.shape)
            self.assertIsNotNone(original)
            np.testing.assert_array_equal(original, self.original)
            self.assertFalse(original.flags.writeable)
            self.assertIsNone(load_original_for_fitting(path, expected_shape=(99, 99, 3)))
            self.assertIsNone(load_original_for_fitting(str(Path(directory) / 'missing.png')))


if __name__ == '__main__':
    unittest.main()
