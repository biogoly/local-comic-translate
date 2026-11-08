"""Curved lettering controls preserve original text and reversible snapshots."""
import copy
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtGui, QtWidgets

from app.ui.canvas.text_item import TextBlockItem
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.dialogs.lettering_transform import LetteringTransformDialog
from modules.utils.lettering_transform import normalize_lettering_transform


class CurvedLetteringUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.item = TextBlockItem(font_family='Arial', font_size=24, outline_color=None)
        self.item.set_plain_text('One bold word\nA second line')
        self.item.setTextWidth(320)
        self.item.setPos(40, 60)
        self.item.setRotation(15)
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(4)
        cursor.setPosition(8, QtGui.QTextCursor.KeepAnchor)
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

    def test_old_snapshot_canonical_form_and_curve_validation(self):
        old = normalize_lettering_transform({'scale_x': 1.3, 'perspective_y': -.2})
        self.assertNotIn('arc_angle', old)
        self.assertEqual(normalize_lettering_transform(dict(old, arc_angle=0)), old)
        self.assertEqual(normalize_lettering_transform({'arc_angle': 0}), {})
        for angle in (121, -121, float('nan'), float('inf'), True, '30'):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                normalize_lettering_transform({'arc_angle': angle})
        self.assertEqual(normalize_lettering_transform({'arc_angle': -120})['arc_angle'], -120)

    def test_curve_preview_preserves_rich_text_and_exposes_export_limit(self):
        self.dialog.controls['arc_angle'].setValue(80)
        self.assertTrue(self.dialog.apply_button.isEnabled())
        current, curved = self.dialog._preview_items
        self.assertEqual(current.toHtml(), curved.toHtml())
        self.assertEqual(curved.toPlainText(), self.item.toPlainText())
        self.assertNotEqual(current.boundingRect(), curved.boundingRect())
        self.assertIs(curved.layout, curved.document().documentLayout())
        self.assertEqual(curved.lettering_transform['arc_angle'], 80)
        self.assertIn('curved lettering', self.dialog.export_note.text())
        self.assertIn('labeled image layer', self.dialog.export_note.text())
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)

    def test_curve_combines_with_perspective_then_reset_retains_position_and_rotation(self):
        self.dialog.controls['arc_angle'].setValue(-65)
        self.dialog.controls['perspective_x'].setValue(20)
        curved = self.dialog._preview_items[1]
        self.assertFalse(curved.transform().isAffine())
        self.assertEqual(curved.lettering_transform['arc_angle'], -65)
        self.dialog.reset_button.click()
        self.assertEqual(self.dialog.candidate['lettering_transform'], {})
        self.assertEqual(self.dialog.candidate['position'], self.before['position'])
        self.assertEqual(self.dialog.candidate['rotation'], self.before['rotation'])
        self.assertEqual(self.dialog._preview_items[1].toHtml(), self.item.toHtml())

    def test_cancel_and_apply_keep_live_page_unmodified_until_controller_commits(self):
        self.dialog.controls['arc_angle'].setValue(60)
        self.dialog.cancel_button.click()
        self.assertIsNone(self.dialog.result_properties)
        self.dialog.apply_selection()
        result = self.dialog.result_properties
        self.assertEqual(result['lettering_transform']['arc_angle'], 60)
        self.dialog.candidate['lettering_transform']['arc_angle'] = 80
        self.assertEqual(result['lettering_transform']['arc_angle'], 60)
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)

    def test_vertical_text_disables_curve_without_changing_other_transforms(self):
        self.dialog.close()
        self.dialog.deleteLater()
        props = copy.deepcopy(self.before)
        props['vertical'] = True
        self.dialog = LetteringTransformDialog(props)
        self.assertFalse(self.dialog.controls['arc_angle'].isEnabled())
        self.assertIn('horizontal lettering', self.dialog.curve_note.text())
        self.dialog.controls['scale_x'].setValue(125)
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.assertNotIn('arc_angle', self.dialog.candidate['lettering_transform'])

    def test_opening_existing_curve_preserves_parameter_precision(self):
        self.dialog.close()
        self.dialog.deleteLater()
        props = copy.deepcopy(self.before)
        props['lettering_transform'] = normalize_lettering_transform({'arc_angle': 32.45678})
        self.dialog = LetteringTransformDialog(props)
        self.assertEqual(self.dialog.candidate, props)
        self.dialog.controls['scale_x'].setValue(110)
        self.assertEqual(self.dialog.candidate['lettering_transform']['arc_angle'], 32.45678)
