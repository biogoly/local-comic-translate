"""Native-text preview behavior and explicit PSD export disclosure."""
import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.dayu_widgets import dayu_theme
from app.ui.dialogs.lettering_transform import LetteringTransformDialog
from modules.utils.lettering_transform import normalize_lettering_transform


class LetteringTransformUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.item = TextBlockItem(font_family='Arial', font_size=22, outline_color=None)
        self.item.set_plain_text('One emphasized word')
        self.item.setTextWidth(250)
        self.item.setPos(40, 60)
        self.item.setRotation(12)
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(4)
        cursor.setPosition(14, QtGui.QTextCursor.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Bold)
        cursor.mergeCharFormat(fmt)
        self.before = TextItemProperties.from_text_item(self.item).to_dict()
        self.dialog = LetteringTransformDialog(self.before)

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.item.deleteLater()
        self.app.processEvents()

    def test_preview_transforms_actual_text_and_preserves_the_document(self):
        self.dialog.controls['scale_x'].setValue(150)
        self.dialog.controls['skew_x'].setValue(20)
        self.dialog.controls['perspective_x'].setValue(30)
        current, transformed = self.dialog._preview_items
        self.assertTrue(current.transform().isIdentity())
        self.assertFalse(transformed.transform().isAffine())
        self.assertEqual(current.toHtml(), transformed.toHtml())
        self.assertNotEqual(current.sceneBoundingRect(), transformed.sceneBoundingRect())
        self.assertEqual(self.dialog.candidate['position'], self.before['position'])
        self.assertEqual(self.dialog.candidate['rotation'], 12)
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)
        self.assertIn('labeled image layer', self.dialog.export_note.text())

    def test_affine_export_is_disclosed_as_editable(self):
        self.dialog.controls['scale_y'].setValue(80)
        self.dialog.controls['skew_x'].setValue(-15)
        self.assertTrue(self.dialog._preview_items[1].transform().isAffine())
        self.assertIn('editable text layer', self.dialog.export_note.text())

    def test_cancel_discards_transformation(self):
        self.dialog.controls['perspective_y'].setValue(40)
        self.dialog.cancel_button.click()
        self.assertIsNone(self.dialog.result_properties)
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)

    def test_apply_returns_independent_snapshot(self):
        self.dialog.controls['scale_x'].setValue(170)
        self.dialog.apply_button.click()
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.Accepted)
        self.assertEqual(self.dialog.result_properties['lettering_transform']['scale_x'], 1.7)
        self.dialog.candidate['lettering_transform']['scale_x'] = 2
        self.assertEqual(self.dialog.result_properties['lettering_transform']['scale_x'], 1.7)
        self.assertTrue(self.item.transform().isIdentity())

    def test_reset_clears_distortion_but_preserves_existing_rotation(self):
        self.dialog.controls['scale_x'].setValue(170)
        self.dialog.controls['perspective_y'].setValue(30)
        self.dialog.reset_button.click()
        self.assertEqual(self.dialog.candidate['lettering_transform'], {})
        self.assertEqual(self.dialog.candidate['rotation'], 12)
        self.assertTrue(self.dialog._preview_items[1].transform().isIdentity())

    def test_opening_preserves_precision_until_a_control_changes(self):
        props = copy.deepcopy(self.before)
        props['lettering_transform'] = normalize_lettering_transform({'scale_x': 1.234567, 'skew_x': 13.2468})
        self.dialog.close()
        self.dialog.deleteLater()
        self.dialog = LetteringTransformDialog(props)
        self.assertEqual(self.dialog.candidate, props)
        self.dialog.controls['scale_y'].setValue(130)
        self.assertEqual(self.dialog.candidate['lettering_transform']['scale_x'], 1.234567)
        self.assertEqual(self.dialog.candidate['lettering_transform']['skew_x'], 13.2468)

    def test_preview_failure_disables_apply(self):
        with patch('app.ui.dialogs.lettering_transform.apply_style_properties', side_effect=ValueError('fixture')):
            self.dialog.rebuild()
        self.assertIsNone(self.dialog.candidate)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIn('could not be previewed', self.dialog.status_label.text())

    def test_six_themes_and_matching_preview_scale(self):
        old = dayu_theme.theme_name
        try:
            self.dialog.show()
            self.app.processEvents()
            for theme in ('light', 'dark', 'midnight', 'parchment', 'lavender', 'mint'):
                dayu_theme.set_theme(theme)
                self.dialog._apply_theme()
                self.dialog.controls['scale_x'].setValue(180)
                self.assertIn(dayu_theme.primary_text_color, self.dialog.styleSheet())
                self.assertIn(dayu_theme.background_in_color, self.dialog.styleSheet())
                self.assertAlmostEqual(self.dialog.current_view.transform().m11(),
                                       self.dialog.transformed_view.transform().m11(), places=2)
        finally:
            dayu_theme.set_theme(old)
