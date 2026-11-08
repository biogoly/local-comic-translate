"""Real Qt rich-text, fallback glyph, geometry and export contracts for styles."""
import copy
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.controllers.lettering_style_format import (
    apply_style_properties, capture_style, DEFAULT_STYLE_FIELDS,
    font_warnings, styled_properties, style_from_properties,
)
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import TextBlockItem, OutlineInfo, OutlineType
from app.ui.canvas.save_renderer import ImageSaveRenderer


class LetteringStyleFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.item = TextBlockItem(font_family='Arial', font_size=18, outline_color=None)
        self.item.set_plain_text('Hello brave world\nSecond line')
        self.item.setTextWidth(210)
        self.item.setPos(15, 20)
        self.item.setRotation(13)
        self.item.setScale(1.1)
        self.item.setTransformOriginPoint(40, 30)
        self.item.region_id = 'region-test'
        self.item.layout_preferences = {'min_font_size': 10}
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(6)
        cursor.setPosition(11, QtGui.QTextCursor.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Bold)
        fmt.setFontItalic(True)
        fmt.setFontUnderline(True)
        fmt.setForeground(QtGui.QColor('#cc3300'))
        cursor.mergeCharFormat(fmt)
        self.item.setTextCursor(cursor)
        self.item._format_selection = (6, 11)
        self.item.selection_outlines.append(OutlineInfo(6, 11, QtGui.QColor('#00aa55'), 2, OutlineType.Selection))
        self.before = TextItemProperties.from_text_item(self.item).to_dict()
        self.style = dict(font_family='Segoe UI', font_size=22.5, text_color='#003366',
            outline=True, outline_color='#eeeeff', outline_width=1.5,
            line_spacing=1.6, alignment=1, bold=False, italic=False, underline=False)

    def tearDown(self):
        self.item.deleteLater()
        self.app.processEvents()

    def char_format(self, position):
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(position)
        cursor.setPosition(position+1, QtGui.QTextCursor.KeepAnchor)
        return cursor.charFormat()

    def test_capture_is_base_style_and_plain_data_not_selected_word_style(self):
        style = capture_style(self.item)
        self.assertFalse(style['bold'])
        self.assertFalse(style['italic'])
        self.assertFalse(style['underline'])
        self.assertEqual(style['text_color'], '#000000')
        self.assertEqual(style['font_family'], 'Arial')
        self.assertEqual(style, style_from_properties(self.before))
        self.assertTrue(all(isinstance(value, (str, int, float, bool)) for value in style.values()))

    def test_default_fields_preserve_word_emphasis_and_all_unselected_properties(self):
        after = styled_properties(self.before, self.style)
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)
        apply_style_properties(self.item, after)
        emphasized = self.char_format(7)
        self.assertTrue(emphasized.fontItalic())
        self.assertTrue(emphasized.fontUnderline())
        self.assertGreaterEqual(emphasized.fontWeight(), int(QtGui.QFont.Bold))
        self.assertEqual(emphasized.foreground().color().name(), '#003366')
        self.assertFalse(self.char_format(1).fontItalic())
        self.assertEqual(self.item.toPlainText(), 'Hello brave world\nSecond line')
        self.assertEqual(self.item.font_family, 'Segoe UI')
        self.assertEqual(self.item.font_size, 22.5)
        self.assertEqual(self.item.textCursor().selectedText(), 'brave')
        self.assertEqual(self.item._format_selection, (6, 11))

    def test_family_only_preserves_word_color_weight_size_and_layout(self):
        after = styled_properties(self.before, self.style, {'font_family'})
        apply_style_properties(self.item, after)
        emphasized = self.char_format(7)
        self.assertEqual(emphasized.foreground().color().name(), '#cc3300')
        self.assertTrue(emphasized.fontItalic())
        self.assertGreaterEqual(emphasized.fontWeight(), int(QtGui.QFont.Bold))
        self.assertEqual(self.item.font_size, self.before['font_size'])
        self.assertEqual(self.item.line_spacing, self.before['line_spacing'])
        self.assertEqual(self.item.selection_outlines, self.before['selection_outlines'])

    def test_explicit_emphasis_fields_flatten_only_those_requested(self):
        after = styled_properties(self.before, self.style, {'italic'})
        apply_style_properties(self.item, after)
        fmt = self.char_format(7)
        self.assertFalse(fmt.fontItalic())
        self.assertTrue(fmt.fontUnderline())
        self.assertGreaterEqual(fmt.fontWeight(), int(QtGui.QFont.Bold))
        after = styled_properties(self.before, {'bold': True, 'underline': False}, {'bold', 'underline'})
        apply_style_properties(self.item, after)
        self.assertGreaterEqual(self.char_format(1).fontWeight(), int(QtGui.QFont.Bold))
        self.assertFalse(self.char_format(7).fontUnderline())
        self.assertTrue(self.char_format(7).fontItalic())

    def test_geometry_identity_direction_and_cursor_are_preserved(self):
        after = styled_properties(self.before, self.style)
        for key in ('position','width','height','rotation','scale','transform_origin','direction','vertical','region_id','layout_preferences'):
            self.assertEqual(after[key], self.before[key], key)
        flags = QtCore.Qt.TextEditorInteraction
        self.item.setTextInteractionFlags(flags)
        apply_style_properties(self.item, after)
        self.assertEqual(self.item.pos(), QtCore.QPointF(*self.before['position']))
        self.assertEqual(self.item.textWidth(), self.before['width'])
        self.assertEqual(self.item.rotation(), 13)
        self.assertEqual(self.item.scale(), 1.1)
        self.assertEqual(self.item.transformOriginPoint(), QtCore.QPointF(40,30))
        self.assertEqual(self.item.textInteractionFlags(), flags)

    def test_all_paragraphs_receive_spacing_and_alignment(self):
        after = styled_properties(self.before, self.style, {'line_spacing','alignment'})
        apply_style_properties(self.item, after)
        block = self.item.document().firstBlock()
        while block.isValid():
            self.assertEqual(block.blockFormat().lineHeight(), 160)
            self.assertTrue(block.blockFormat().alignment() & QtCore.Qt.AlignLeft)
            block = block.next()

    def test_outline_changes_preserve_individual_word_outlines(self):
        after = styled_properties(self.before, self.style, {'outline','outline_color','outline_width'})
        apply_style_properties(self.item, after)
        self.assertTrue(self.item.outline)
        self.assertEqual(len(self.item.selection_outlines), 2)
        self.assertEqual(self.item.selection_outlines[0], self.before['selection_outlines'][0])
        self.assertEqual(self.item.selection_outlines[1].type, OutlineType.Full_Document)
        disabled = styled_properties(after, {'outline':False}, {'outline'})
        apply_style_properties(self.item, disabled)
        self.assertFalse(self.item.outline)
        self.assertEqual(self.item.selection_outlines, self.before['selection_outlines'])

    def test_snapshot_restore_is_exact_after_style_and_subsequent_edit(self):
        after = styled_properties(self.before, self.style)
        apply_style_properties(self.item, after)
        self.item.setPlainText('A later spelling edit')
        apply_style_properties(self.item, self.before)
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)
        apply_style_properties(self.item, after)
        self.assertEqual(self.item.toHtml(), after['text'])
        self.assertEqual(self.item.font_family, after['font_family'])
        self.assertEqual(self.item.line_spacing, after['line_spacing'])

    def test_empty_fields_and_absent_properties_are_exact_noops(self):
        self.assertEqual(styled_properties(self.before, self.style, set()), self.before)
        self.assertEqual(styled_properties(self.before, {'bold':True}, {'font_family'}), self.before)

    def test_nonselected_invalid_style_is_rejected_explicitly(self):
        with self.assertRaises(ValueError):
            styled_properties(self.before, {'font_size':float('nan')}, {'font_size'})

    def test_missing_font_warns_without_substituting_stored_family(self):
        after = styled_properties(self.before, {'font_family':'Definitely Missing Typeface 789'}, {'font_family'})
        warnings = font_warnings(after)
        self.assertTrue(any(w['code']=='missing_font' and w['font']=='Definitely Missing Typeface 789' for w in warnings))
        self.assertEqual(after['font_family'], 'Definitely Missing Typeface 789')

    def test_missing_glyph_warning_uses_actual_shaping(self):
        self.item.set_plain_text('Hello\U0010ffff')
        warnings = font_warnings(TextItemProperties.from_text_item(self.item).to_dict())
        self.assertTrue(any(w['code']=='missing_glyphs' and '\U0010ffff' in w['characters'] for w in warnings))

    def test_existing_language_samples_survive_style_and_available_fallback(self):
        text = 'English Français Español Deutsch Italiano Türkçe Русский 日本語 简体中文 한국어 Tiếng Việt Português'
        self.item.set_plain_text(text)
        before = TextItemProperties.from_text_item(self.item).to_dict()
        after = styled_properties(before, {'font_family':'Arial','font_size':16}, {'font_family','font_size'})
        apply_style_properties(self.item, after)
        self.assertEqual(self.item.toPlainText(), text)
        warnings = font_warnings(after)
        # The sandbox registers Japanese/Korean fallback fonts, not a full
        # Simplified Chinese font. Unsupported characters must be diagnosed;
        # supported Latin, Cyrillic, Japanese and Korean must not be false alarms.
        self.assertTrue(all(w['code'] == 'missing_glyphs' for w in warnings))
        self.assertTrue(all(set(w['characters']) <= {'简'} for w in warnings))

    def test_vertical_style_keeps_bound_geometry_and_custom_layout(self):
        self.item.set_plain_text('日本語の文字')
        self.item.set_vertical(True)
        self.item.layout.set_max_size(150, 240)
        self.item.layout_preferences = {'bound_height':240}
        before = TextItemProperties.from_text_item(self.item).to_dict()
        after = styled_properties(before, {'line_spacing':1.7,'font_size':20}, {'line_spacing','font_size'})
        apply_style_properties(self.item, after)
        self.assertTrue(self.item.vertical)
        self.assertEqual(self.item.layout.line_spacing,1.7)
        self.assertEqual(self.item.layout.max_height,240)
        self.assertEqual(self.item.toPlainText(),'日本語の文字')
        apply_style_properties(self.item, before)
        self.assertEqual(self.item.toHtml(), before['text'])

    def test_saved_style_export_matches_live_html_and_visible_ink(self):
        after = styled_properties(self.before, self.style)
        apply_style_properties(self.item, after)
        saved = TextItemProperties.from_text_item(self.item).to_dict()
        renderer = ImageSaveRenderer(np.full((240,300,3),255,np.uint8))
        renderer.add_state_to_image({'text_items_state':[saved]})
        restored = next(item for item in renderer.scene.items() if isinstance(item, TextBlockItem))
        self.assertEqual(restored.toHtml(), self.item.toHtml())
        self.assertEqual(restored.font_family, self.item.font_family)
        self.assertEqual(restored.line_spacing, self.item.line_spacing)
        self.assertEqual(restored.selection_outlines, self.item.selection_outlines)
        self.assertTrue(np.any(renderer.render_to_image()!=255))

    def test_deleting_and_retyping_keeps_applied_base_font(self):
        after = styled_properties(self.before, self.style, DEFAULT_STYLE_FIELDS | {'bold'})
        apply_style_properties(self.item, after)
        self.item.setPlainText('')
        cursor = self.item.textCursor()
        cursor.insertText('Typed again')
        self.assertEqual(self.item.font_family,'Segoe UI')
        self.assertEqual(self.item.document().defaultFont().family(),'Segoe UI')
        self.assertEqual(cursor.charFormat().font().family(),'Segoe UI')


if __name__ == '__main__':
    unittest.main()
