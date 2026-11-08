"""Native Qt interactions for the bounded, explicit candidate preview."""
import os
from pathlib import Path
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog


class ArtisticEditCandidatesUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.font_id = -1
        if os.name == 'nt':
            font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'arial.ttf'
            cls.font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
            if cls.font_id < 0:
                raise RuntimeError('Could not load the installed Arial test font')
        cls.font = QtGui.QFont('Arial', 10)
        raw_font = QtGui.QRawFont.fromFont(cls.font)
        if not raw_font.isValid() or 0 in raw_font.glyphIndexesForString('Candidate'):
            raise RuntimeError('Candidate preview tests require actual readable font glyphs')

    @classmethod
    def tearDownClass(cls):
        if cls.font_id >= 0:
            QtGui.QFontDatabase.removeApplicationFont(cls.font_id)

    def setUp(self):
        self.before = np.full((600, 800, 3), (20, 40, 60), np.uint8)
        self.candidates = [
            np.full_like(self.before, (200, 50, 80)),
            np.full_like(self.before, (50, 180, 90)),
        ]
        self.dialog = ArtisticEditPreviewDialog(
            self.before, self.candidates[0], 'Candidate one summary', apply_available=True)
        self.dialog.setFont(self.font)
        self.dialog.configure_candidates(['Candidate 1'], 0)
        self.dialog.show()
        self.app.processEvents()
        self.addCleanup(self.dialog.deleteLater)
        self.addCleanup(self.dialog.close)

    @staticmethod
    def pixel(view):
        return view._item.pixmap().toImage().pixelColor(0, 0).getRgb()[:3]

    def two_candidates(self, selected=0):
        self.dialog.configure_candidates(['Candidate 1', 'Candidate 2'], selected)

        def select(index):
            self.dialog.set_candidate(
                self.before, self.candidates[index], f'Candidate {index + 1} summary',
                apply_available=True)
            self.dialog.set_comparison_candidate(
                self.candidates[1 - index], f'Candidate {2 - index}')

        self.dialog.candidate_selected.connect(select)
        select(selected)
        self.app.processEvents()

    def test_single_candidate_has_explicit_alternative_action(self):
        accepted = QtTest.QSignalSpy(self.dialog.accepted)
        self.assertTrue(self.dialog.candidate_row.isHidden())
        self.assertTrue(self.dialog.alternative_button.isVisible())
        self.assertTrue(self.dialog.alternative_button.isEnabled())
        QtTest.QTest.mouseClick(self.dialog.alternative_button, QtCore.Qt.LeftButton)
        self.assertEqual(self.dialog.result(), ArtisticEditPreviewDialog.GENERATE_ALTERNATIVE)
        self.assertEqual(accepted.count(), 0)
        self.assertFalse(self.dialog.isVisible())

    def test_keyboard_selection_refreshes_preview_without_accepting(self):
        selected = QtTest.QSignalSpy(self.dialog.candidate_selected)
        accepted = QtTest.QSignalSpy(self.dialog.accepted)
        self.two_candidates()
        self.assertEqual(selected.count(), 0)
        self.assertTrue(self.dialog.candidate_row.isVisible())
        self.assertFalse(self.dialog.alternative_button.isEnabled())
        self.assertIn('Two candidates', self.dialog.alternative_button.toolTip())
        self.dialog.candidate_combo.setFocus()
        QtTest.QTest.keyClick(self.dialog.candidate_combo, QtCore.Qt.Key_Down)
        self.assertEqual(selected.count(), 1)
        self.assertEqual(selected.at(0), [1])
        self.assertEqual(self.dialog.selected_candidate_index, 1)
        self.assertEqual(self.dialog.summary_label.text(), 'Candidate 2 summary')
        self.assertEqual(self.dialog.after_title.text(), 'Candidate 2')
        self.assertEqual(self.pixel(self.dialog.after_view), (50, 180, 90))
        self.assertEqual(self.pixel(self.dialog.before_view), (20, 40, 60))
        self.assertEqual(accepted.count(), 0)
        self.assertTrue(self.dialog.isVisible())

    def test_ab_comparison_follows_other_candidate_and_can_return_to_before(self):
        self.two_candidates()
        self.dialog.comparison_combo.setFocus()
        QtTest.QTest.keyClick(self.dialog.comparison_combo, QtCore.Qt.Key_Down)
        self.assertEqual(self.dialog.before_title.text(), 'Candidate 2')
        self.assertEqual(self.pixel(self.dialog.before_view), (50, 180, 90))
        self.assertEqual(self.pixel(self.dialog.after_view), (200, 50, 80))
        QtTest.QTest.keyClick(self.dialog.candidate_combo, QtCore.Qt.Key_Down)
        self.assertEqual(self.dialog.before_title.text(), 'Candidate 1')
        self.assertEqual(self.pixel(self.dialog.before_view), (200, 50, 80))
        self.assertEqual(self.pixel(self.dialog.after_view), (50, 180, 90))
        self.assertEqual(self.dialog.comparison_combo.currentData(), 'candidate')
        QtTest.QTest.keyClick(self.dialog.comparison_combo, QtCore.Qt.Key_Home)
        self.assertEqual(self.dialog.before_title.text(), 'Before')
        self.assertEqual(self.pixel(self.dialog.before_view), (20, 40, 60))

    def test_stale_gate_resets_when_controller_supplies_current_candidate(self):
        self.dialog.block_apply('Page changed')
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertFalse(self.dialog.alternative_button.isEnabled())
        self.assertIn('stale', self.dialog.alternative_button.toolTip())
        self.dialog.set_candidate(
            self.before, self.candidates[1], 'New preview', apply_available=True)
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.assertTrue(self.dialog.alternative_button.isEnabled())
        self.assertTrue(self.dialog.stale_label.isHidden())
        self.assertEqual(self.dialog.stale_label.text(), '')
        self.dialog.set_candidate(
            self.before, self.candidates[0], 'Old preview',
            apply_available=False, stale_reason='Region changed')
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertFalse(self.dialog.alternative_button.isEnabled())
        self.assertEqual(self.dialog.stale_label.text(), 'Region changed')

    def test_candidate_bounds_reject_overflow_without_reconfiguring_selection(self):
        self.two_candidates(1)
        for labels, index in (([], 0), (['A', 'B', 'C'], 0), (['A', 'B'], 2)):
            with self.subTest(labels=labels, index=index), self.assertRaises(ValueError):
                self.dialog.configure_candidates(labels, index)
        self.assertEqual(self.dialog.selected_candidate_index, 1)
        self.assertEqual(self.dialog.candidate_combo.count(), 2)
        self.dialog.configure_candidates(['Candidate 1'], 0)
        self.assertTrue(self.dialog.candidate_row.isHidden())
        self.assertEqual(self.dialog.comparison_combo.count(), 1)
        self.assertEqual(self.pixel(self.dialog.before_view), (20, 40, 60))

    def test_wheel_zoom_is_synchronized_and_preserved_when_comparing(self):
        self.two_candidates()
        fit_button = next(button for button in self.dialog.findChildren(QtWidgets.QPushButton)
                          if button.text() == 'Fit')
        QtTest.QTest.mouseClick(fit_button, QtCore.Qt.LeftButton)
        original_scale = self.dialog.after_view.transform().m11()
        viewport = self.dialog.after_view.viewport()
        position = QtCore.QPointF(viewport.rect().center())
        event = QtGui.QWheelEvent(
            position, QtCore.QPointF(viewport.mapToGlobal(position.toPoint())),
            QtCore.QPoint(), QtCore.QPoint(0, 120), QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
        QtWidgets.QApplication.sendEvent(viewport, event)
        zoom = self.dialog.after_view.transform().m11()
        self.assertGreater(zoom, original_scale)
        self.assertAlmostEqual(self.dialog.before_view.transform().m11(), zoom)
        QtTest.QTest.keyClick(self.dialog.candidate_combo, QtCore.Qt.Key_Down)
        QtTest.QTest.keyClick(self.dialog.comparison_combo, QtCore.Qt.Key_Down)
        self.assertAlmostEqual(self.dialog.after_view.transform().m11(), zoom)
        self.assertAlmostEqual(self.dialog.before_view.transform().m11(), zoom)
        QtTest.QTest.mouseClick(fit_button, QtCore.Qt.LeftButton)
        self.assertAlmostEqual(self.dialog.after_view.transform().m11(), original_scale)

    def test_regenerate_actions_keep_selected_candidate_and_existing_decision_codes(self):
        self.two_candidates(1)
        for text, result in (
            ('Regenerate Same Seed', ArtisticEditPreviewDialog.REGENERATE_SAME_SEED),
            ('Regenerate New Seed', ArtisticEditPreviewDialog.REGENERATE_NEW_SEED),
        ):
            with self.subTest(text=text):
                self.dialog.show()
                button = next(button for button in self.dialog.findChildren(QtWidgets.QPushButton)
                              if button.text() == text)
                QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
                self.assertEqual(self.dialog.result(), result)
                self.assertEqual(self.dialog.selected_candidate_index, 1)


if __name__ == '__main__':
    unittest.main()
