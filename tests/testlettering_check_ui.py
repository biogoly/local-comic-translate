"""Advisory lettering checks never approve, modify or gate an image preview."""
import os
from types import SimpleNamespace
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets

from app.ui.dialogs.artistic_edit_preview import ArtisticEditPreviewDialog


class LetteringCheckUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def dialog(self, available=True):
        before = np.full((32, 32, 3), (80, 140, 180), np.uint8)
        after = np.full((32, 32, 3), (180, 140, 80), np.uint8)
        dialog = ArtisticEditPreviewDialog(before, after, 'Generated preview',
                                          apply_available=available)
        self.addCleanup(dialog.deleteLater)
        return dialog

    @staticmethod
    def result(status='match', expected='MA CHÉRIE!', recognized='MA CHÉRIE!', differences=()):
        return SimpleNamespace(status=status, expected=expected,
                               recognized=recognized, differences=differences)

    def test_freeform_preview_has_no_check_or_false_result(self):
        dialog = self.dialog()
        self.assertFalse(dialog.lettering_check_button.isEnabled())
        self.assertFalse(dialog.lettering_details_toggle.isEnabled())
        self.assertIn('Use selected translation', dialog.lettering_status_label.text())
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertEqual(dialog.result(), 0)

    def test_available_check_emits_signal_but_does_not_accept_preview(self):
        dialog = self.dialog()
        dialog.configure_lettering_check('MA CHÉRIE!', 'Default OCR (local)')
        spy = QtTest.QSignalSpy(dialog.lettering_check_requested)
        accepted = QtTest.QSignalSpy(dialog.accepted)
        dialog.show()
        self.app.processEvents()
        QtTest.QTest.mouseClick(dialog.lettering_check_button, QtCore.Qt.LeftButton)
        self.assertEqual(spy.count(), 1)
        self.assertEqual(accepted.count(), 0)
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertEqual(dialog.lettering_route_label.text(), 'Default OCR (local)')
        self.assertFalse(dialog.lettering_check_button.autoDefault())
        dialog.close()

    def test_busy_state_disables_only_check_and_apply_remains_explicit(self):
        dialog = self.dialog()
        dialog.configure_lettering_check('Bonjour', 'Default OCR (local)')
        dialog.set_lettering_check_busy(True)
        self.assertFalse(dialog.lettering_check_button.isEnabled())
        self.assertIn('Checking', dialog.lettering_status_label.text())
        for button in dialog.findChildren(QtWidgets.QPushButton):
            if button is not dialog.lettering_check_button:
                self.assertTrue(button.isEnabled(), button.text())
        accepted = QtTest.QSignalSpy(dialog.accepted)
        dialog.apply_button.click()
        self.assertEqual(accepted.count(), 1)

    def test_results_are_advisory_and_leave_images_and_apply_state_unchanged(self):
        for available in (False, True):
            with self.subTest(apply_available=available):
                dialog = self.dialog(available)
                dialog.configure_lettering_check('MA CHÉRIE!', 'Default OCR (local)')
                before = dialog.before_view._item.pixmap().toImage()
                after = dialog.after_view._item.pixmap().toImage()
                accepted = QtTest.QSignalSpy(dialog.accepted)
                for status in ('match', 'mismatch', 'uncertain'):
                    dialog.set_lettering_check_busy(True)
                    dialog.show_lettering_check_result(self.result(status))
                    self.assertTrue(dialog.lettering_check_button.isEnabled())
                    self.assertEqual(dialog.apply_button.isEnabled(), available)
                    self.assertEqual(accepted.count(), 0)
                    self.assertEqual(dialog.before_view._item.pixmap().toImage(), before)
                    self.assertEqual(dialog.after_view._item.pixmap().toImage(), after)
                    self.assertTrue(dialog.lettering_details_toggle.isChecked())
                self.assertIn('could not read enough text', dialog.lettering_status_label.text())

    def test_difference_report_retains_accents_and_plain_markup(self):
        dialog = self.dialog()
        dialog.configure_lettering_check('<b>CHÉRIE</b>', '<b>Hosted OCR</b>')
        differences = (
            SimpleNamespace(kind='replace', expected='CHÉRIE', recognized='CHERIE'),
            SimpleNamespace(kind='missing', expected='<b>', recognized=''),
            SimpleNamespace(kind='extra', expected='', recognized='<script>extra</script>'),
        )
        dialog.show_lettering_check_result(self.result(
            'mismatch', '<b>CHÉRIE</b>', '<script>CHERIE</script>', differences))
        text = dialog.lettering_details.toPlainText()
        self.assertIn('<b>CHÉRIE</b>', text)
        self.assertIn('<script>CHERIE</script>', text)
        self.assertIn('Requested: CHÉRIE | OCR: CHERIE', text)
        self.assertIn('Not read by OCR: <b>', text)
        self.assertIn('Extra OCR text: <script>extra</script>', text)
        self.assertEqual(dialog.lettering_route_label.textFormat(), QtCore.Qt.PlainText)
        self.assertEqual(dialog.lettering_status_label.textFormat(), QtCore.Qt.PlainText)
        self.assertEqual(dialog.lettering_status_label.text(), 'Possible lettering differences')
        self.assertIn('Review the image', dialog.lettering_advisory_label.text())

    def test_unavailable_route_and_error_do_not_enable_stale_apply(self):
        dialog = self.dialog(False)
        dialog.configure_lettering_check('Hello', 'Hosted OCR', 'Configure an OCR key first.')
        self.assertFalse(dialog.lettering_check_button.isEnabled())
        self.assertEqual(dialog.lettering_status_label.text(), 'Configure an OCR key first.')
        dialog.show_lettering_check_error('<b>connection failed</b>')
        self.assertFalse(dialog.lettering_check_button.isEnabled())
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertIn('<b>connection failed</b>', dialog.lettering_status_label.text())
        dialog.configure_lettering_check('Hello', 'Default OCR (local)')
        dialog.set_lettering_check_busy(True)
        dialog.show_lettering_check_error('OCR failed')
        self.assertTrue(dialog.lettering_check_button.isEnabled())
        self.assertFalse(dialog.apply_button.isEnabled())

    def test_details_can_collapse_and_reconfigure_clears_old_result(self):
        dialog = self.dialog()
        dialog.configure_lettering_check('Hello', 'Default OCR (local)')
        dialog.show_lettering_check_result(self.result('match', 'Hello', 'Hello'))
        self.assertFalse(dialog.lettering_details.isHidden())
        dialog.lettering_details_toggle.setChecked(False)
        self.assertTrue(dialog.lettering_details.isHidden())
        dialog.configure_lettering_check(None, '')
        self.assertFalse(dialog.lettering_check_button.isEnabled())
        self.assertNotIn('Hello', dialog.lettering_details.toPlainText())
        self.assertNotIn('matches', dialog.lettering_status_label.text())
        self.assertTrue(dialog.apply_button.isEnabled())


if __name__ == '__main__':
    unittest.main()
