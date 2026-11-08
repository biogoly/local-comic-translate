"""Fixed-source gradient previews with real Qt sampling and no model calls."""
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from app.ui.dayu_widgets import dayu_theme
from app.ui.dialogs.gradient_repair import GradientRepairDialog
from modules.repair.gradient import GradientFitError, GradientSampleLine, fit_gradient


class GradientRepairUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        y, x = np.mgrid[:80, :100]
        self.original = np.stack((30 + x, 80 + y, 180 - x), axis=2).astype(np.uint8)
        self.current = self.original.copy()
        self.mask = np.zeros((80, 100), np.uint8)
        self.mask[25:55, 35:65] = 255
        self.mask[25, 35:65] = 128
        self.current[26:54, 36:64] = (220, 5, 20)
        self.points = [(8, 8), (92, 8), (92, 72), (8, 72), (50, 8), (50, 72)]
        self.dialog = GradientRepairDialog(self.current, self.original, self.mask)
        self.dialog.show()
        self.app.processEvents()

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def add_samples(self):
        for point in self.points:
            self.dialog.add_sample(*point)

    def repaired(self):
        image = self.dialog.current.copy()
        x, y, width, height = self.dialog.patch['bbox']
        image[y:y + height, x:x + width] = self.dialog.patch['image']
        return image

    def test_inputs_are_owned_immutable_snapshots(self):
        before = self.dialog.current.copy()
        self.current[:] = 0
        self.original[:] = 255
        self.mask[:] = 0
        np.testing.assert_array_equal(self.dialog.current, before)
        self.assertTrue(np.any(self.dialog.mask))
        for pixels in (self.dialog.current, self.dialog.original, self.dialog.mask):
            self.assertFalse(pixels.flags.writeable)

    def test_preview_stays_inside_mask_and_is_repeatable(self):
        self.add_samples()
        self.dialog.generate_preview()
        self.assertTrue(self.dialog.apply_button.isEnabled())
        first = self.repaired()
        np.testing.assert_array_equal(first[self.mask == 0], self.current[self.mask == 0])
        np.testing.assert_array_equal(first[30:50, 40:60], self.original[30:50, 40:60])
        self.dialog.generate_preview()
        np.testing.assert_array_equal(self.repaired(), first)
        np.testing.assert_array_equal(self.dialog.current, self.current)
        self.assertEqual(self.dialog.metadata['source'], 'current')
        self.assertEqual(self.dialog.metadata['sample_count'], 6)
        self.dialog.accept()
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)

    def test_protected_pixels_survive_feathered_gradient_preview(self):
        protection = np.zeros_like(self.mask)
        protection[25:45, 35:50] = 255
        self.dialog.close()
        self.dialog.deleteLater()
        self.dialog = GradientRepairDialog(self.current, self.original, self.mask, protection=protection)
        protection[:] = 0  # caller changes cannot mutate preview snapshots
        self.add_samples()
        self.dialog.generate_preview()
        after = self.repaired()
        np.testing.assert_array_equal(after[25:45, 35:50], self.current[25:45, 35:50])
        np.testing.assert_array_equal(after[30:50, 51:60], self.original[30:50, 51:60])
        self.assertFalse(self.dialog.protection.flags.writeable)

    def test_protected_overhang_does_not_require_samples_beyond_writable_ink(self):
        # A broad stroke over the neighboring protected word extends beyond the
        # clean sample hull. The unprotected part still lies fully within it.
        mask = self.mask.copy()
        mask[32:40, 65:91] = 255
        protected = np.zeros_like(mask)
        protected[30:42, 65:92] = 255
        current = self.current.copy()
        current[32:40, 65:91] = (20, 10, 90)
        samples = [(20, 18), (72, 18), (72, 60), (20, 60)]
        dialog = GradientRepairDialog(current, self.original, mask, protection=protected)
        try:
            for point in samples:
                dialog.add_sample(*point)
            QtTest.QTest.mouseClick(dialog.generate_button, QtCore.Qt.LeftButton)
            self.assertTrue(dialog.apply_button.isEnabled(), dialog.status_label.text())
            x, y, w, h = dialog.patch['bbox']
            after = current.copy()
            after[y:y+h, x:x+w] = dialog.patch['image']
            np.testing.assert_array_equal(after[30:50, 40:60], self.original[30:50, 40:60])
            locked = (protected != 0) | (mask == 0)
            np.testing.assert_array_equal(after[locked], current[locked])
            for invalid in ((80, 35), GradientSampleLine((80, 29), (80, 43))):
                with self.subTest(invalid=invalid):
                    dialog._add_observation(invalid)
                    self.assertEqual(dialog.sample_points, samples)
                    self.assertIn('outside the painted mask', dialog.status_label.text())
                    with self.assertRaises(GradientFitError) as error:
                        fit_gradient(current, mask, samples + [invalid], protection=protected)
                    self.assertEqual(error.exception.code, 'sample_inside_mask')
            # One uncovered writable pixel must still prevent extrapolation.
            protected[35, 90] = 0
            with self.assertRaises(GradientFitError) as error:
                fit_gradient(current, mask, samples, protection=protected)
            self.assertEqual(error.exception.code, 'insufficient_coverage')
        finally:
            dialog.close()

    def test_fully_protected_gradient_has_no_candidate_to_apply(self):
        dialog = GradientRepairDialog(self.current, self.original, self.mask, protection=self.mask)
        try:
            for point in self.points:
                dialog.add_sample(*point)
            QtTest.QTest.mouseClick(dialog.generate_button, QtCore.Qt.LeftButton)
            self.assertIn('would not change any pixels', dialog.status_label.text())
            self.assertIsNone(dialog.patch)
            self.assertFalse(dialog.apply_button.isEnabled())
            dialog.accept()
            self.assertNotEqual(dialog.result(), QtWidgets.QDialog.Accepted)
        finally:
            dialog.close()

    def test_source_change_clears_samples_and_does_not_sample_preview(self):
        self.add_samples()
        self.dialog.generate_preview()
        self.dialog.source_combo.setCurrentIndex(1)
        self.assertEqual(self.dialog.sample_points, [])
        self.assertIsNone(self.dialog.patch)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIs(self.dialog.source, self.dialog.original)
        self.add_samples()
        self.dialog.generate_preview()
        self.assertEqual(self.dialog.metadata['source'], 'original')
        np.testing.assert_array_equal(self.repaired()[30:50, 40:60], self.original[30:50, 40:60])

    def test_new_or_removed_sample_invalidates_apply(self):
        self.add_samples()
        self.dialog.generate_preview()
        self.dialog.add_sample(8, 40)
        self.assertIsNone(self.dialog.patch)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.dialog.generate_preview()
        self.dialog.remove_last_sample()
        self.assertIsNone(self.dialog.patch)
        self.dialog.clear_samples()
        self.assertFalse(self.dialog.generate_button.isEnabled())
        self.assertEqual(self.dialog._markers, [])

    def test_invalid_sample_locations_are_not_added(self):
        for point in ((-1, 0), (100, 0), (40, 40)):
            self.dialog.add_sample(*point)
            self.assertEqual(self.dialog.sample_points, [])
            self.assertTrue(self.dialog.status_label.text())
        self.dialog.add_sample(8, 8)
        self.dialog.add_sample(8, 8)
        self.assertEqual(self.dialog.sample_points, [(8, 8)])

    def test_zoomed_click_uses_image_pixels_and_pan_is_not_a_sample(self):
        view = self.dialog.sample_view
        self.dialog.zoom(1.25)
        pos = view.mapFromScene(QtCore.QPointF(8.5, 8.5))
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pos)
        self.assertEqual(self.dialog.sample_points, [(8, 8)])
        start = view.mapFromScene(QtCore.QPointF(75.5, 20.5))
        QtTest.QTest.mousePress(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
        QtTest.QTest.mouseMove(view.viewport(), start + QtCore.QPoint(15, 15))
        QtTest.QTest.mouseRelease(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start + QtCore.QPoint(15, 15))
        self.assertEqual(len(self.dialog.sample_points), 1)

    def test_zoomed_line_counts_once_and_preserves_points_and_real_endpoints(self):
        d = self.dialog
        for xy in [(15, 14), (85, 14), (85, 64)]:
            d.add_sample(*xy)
        d.zoom(1.25)
        view = d.sample_view
        start = view.mapFromScene(QtCore.QPointF(15.5, 64.5))
        end = view.mapFromScene(QtCore.QPointF(25.5, 64.5))
        QtTest.QTest.mousePress(view.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, start)
        QtTest.QTest.mouseMove(view.viewport(), end)
        self.assertEqual(len(d.sample_points), 3)
        self.assertEqual(view._line_preview.line(), QtCore.QLineF(15.5, 64.5, 25.5, 64.5))
        QtTest.QTest.mouseRelease(view.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, end)
        self.assertIsNone(view._line_preview)
        self.assertEqual(d.sample_points[-1], GradientSampleLine((15, 64), (25, 64)))
        self.assertEqual(d.sample_count.text(), 'Samples: 4')
        d.generate_preview()
        self.assertTrue(d.apply_button.isEnabled(), d.status_label.text())
        self.assertEqual(json.loads(d.metadata['sample_points_json']), [[15, 14], [85, 14], [85, 64]])
        self.assertEqual(json.loads(d.metadata['sample_lines_json']), [[[15, 64], [25, 64]]])
        self.assertEqual(d.metadata['sample_count'], 4)
        np.testing.assert_array_equal(self.repaired()[30:50, 40:60], self.original[30:50, 40:60])
        np.testing.assert_array_equal(d.current, self.current)
        np.testing.assert_array_equal(d.mask, self.mask)
        d.add_sample(20, 64)  # The line midpoint already supplies this observation.
        self.assertEqual(len(d.sample_points), 4)
        d.add_sample(49, 12)
        self.assertIsNone(d.patch)
        d.remove_last_sample()
        self.assertIsInstance(d.sample_points[-1], GradientSampleLine)
        d.remove_last_sample()
        self.assertEqual(len(d.sample_points), 3)
        self.assertFalse(d.generate_button.isEnabled())

    def test_line_crossing_mask_or_cancelled_gesture_does_not_add_observation(self):
        d = self.dialog
        view = d.sample_view
        vp = view.viewport()
        start = view.mapFromScene(QtCore.QPointF(15.5, 36.5))
        end = view.mapFromScene(QtCore.QPointF(84.5, 36.5))
        def press():
            view.setFocus()
            QtTest.QTest.mousePress(vp, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, start)
            self.assertIsNotNone(view._line_preview)

        press()
        QtTest.QTest.mouseRelease(vp, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, end)
        self.assertIn('outside the painted mask', d.status_label.text())
        press()
        QtTest.QTest.mouseRelease(vp, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, QtCore.QPoint(-5, start.y()))
        self.assertIsNone(view._line_preview)
        press()
        d.source_combo.setFocus()
        self.app.processEvents()
        self.assertIsNone(view._line_preview)
        QtTest.QTest.mouseRelease(vp, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, end)
        press()
        d.reject()
        self.assertIsNone(view._line_preview)
        self.assertEqual(d.sample_points, [])
        self.assertFalse(d.apply_button.isEnabled())

    def test_fit_failure_and_cancel_leave_no_accepted_patch(self):
        for x in (10, 30, 70, 90):
            self.dialog.add_sample(x, 8)
        self.dialog.generate_preview()
        self.assertIsNone(self.dialog.patch)
        self.assertIn('both directions', self.dialog.status_label.text())
        self.dialog.accept()
        self.assertNotEqual(self.dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)
        self.dialog.clear_samples()
        self.add_samples()
        self.dialog.generate_preview()
        self.dialog.reject()
        self.assertIsNone(self.dialog.patch)
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.DialogCode.Rejected)

    def test_missing_original_is_disabled(self):
        dialog = GradientRepairDialog(self.current, None, self.mask)
        try:
            self.assertFalse(dialog.source_combo.model().item(1).isEnabled())
            self.assertEqual(dialog.source_combo.currentData(), 'current')
        finally:
            dialog.close()

    def test_eight_samples_around_round_mask_generate_from_button(self):
        y, x = np.mgrid[:80, :100]
        mask = (((x - 50) ** 2 + (y - 40) ** 2) <= 100).astype(np.uint8) * 255
        current = self.original.copy()
        current[mask > 0] = 0
        dialog = GradientRepairDialog(current, self.original, mask)
        try:
            for point in ((62, 40), (59, 49), (50, 52), (41, 49),
                          (38, 40), (41, 31), (50, 28), (59, 31)):
                dialog.add_sample(*point)
            QtTest.QTest.mouseClick(dialog.generate_button, QtCore.Qt.MouseButton.LeftButton)
            self.assertTrue(dialog.apply_button.isEnabled(), dialog.status_label.text())
            self.assertEqual(dialog.metadata['sample_count'], 8)
            self.assertIn('Preview ready', dialog.status_label.text())
            x0, y0, width, height = dialog.patch['bbox']
            np.testing.assert_array_equal(dialog.patch['image'], self.original[y0:y0+height, x0:x0+width])
        finally:
            dialog.close()

    def test_unexpected_fit_error_is_visible_and_retryable_from_button(self):
        self.add_samples()
        with patch('app.ui.dialogs.gradient_repair.fit_gradient', side_effect=np.linalg.LinAlgError('SVD failed')):
            with self.assertLogs('app.ui.dialogs.gradient_repair', level='ERROR'):
                QtTest.QTest.mouseClick(self.dialog.generate_button, QtCore.Qt.MouseButton.LeftButton)
        self.assertIn('Unable to generate preview: SVD failed', self.dialog.status_label.text())
        self.assertEqual(self.dialog.status_label.property('severity'), 'error')
        self.assertIsNone(self.dialog.patch)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertTrue(self.dialog.generate_button.isEnabled())
        self.assertIsNone(self.app.overrideCursor())
        QtTest.QTest.mouseClick(self.dialog.generate_button, QtCore.Qt.MouseButton.LeftButton)
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.assertEqual(self.dialog.status_label.property('severity'), 'info')

    def test_display_failure_cannot_publish_partial_patch(self):
        self.add_samples()
        with patch.object(self.dialog.preview_view, 'set_image', side_effect=[None, RuntimeError('Preview display failed')]):
            with self.assertLogs('app.ui.dialogs.gradient_repair', level='ERROR'):
                QtTest.QTest.mouseClick(self.dialog.generate_button, QtCore.Qt.MouseButton.LeftButton)
        self.assertIn('Preview display failed', self.dialog.status_label.text())
        self.assertIsNone(self.dialog.patch)
        self.assertEqual(self.dialog.metadata, {})
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertTrue(self.dialog.generate_button.isEnabled())
        self.assertIsNone(self.app.overrideCursor())

    def test_all_themes_keep_readable_controls_and_matching_view_scales(self):
        old_theme = dayu_theme.theme_name
        try:
            for name in ('light', 'dark', 'midnight', 'parchment', 'lavender', 'mint'):
                dayu_theme.set_theme(name)
                self.dialog._apply_theme()
                self.assertIn(dayu_theme.primary_text_color, self.dialog.styleSheet())
                self.dialog.fit_page()
                self.dialog.zoom(1.25)
                self.assertAlmostEqual(self.dialog.sample_view.transform().m11(),
                                       self.dialog.preview_view.transform().m11())
                self.assertGreater(self.dialog.apply_button.sizeHint().height(), 24)
        finally:
            dayu_theme.set_theme(old_theme)


if __name__ == '__main__':
    unittest.main()
