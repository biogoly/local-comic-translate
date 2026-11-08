"""Real Qt fitting/application with synthetic original pixels; no model calls."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

import testtext_fit as existing
from testlayout_boundary import fixture
from app.ui.dialogs.text_fit import TextFitDialog
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.save_renderer import ImageSaveRenderer


class BubbleFitWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        existing.FitTextWorkflowTests.setUp(self)
        self.block.xyxy = np.array([65, 65, 165, 125])
        self.block.bubble_xyxy = np.array([20, 20, 220, 180])
        self.item.setPos(65, 65)
        self.item.setTextWidth(100)
        self.source = fixture()
        self.main.image_data = {'page.png': self.source.copy()}
        self.ctrl = self.main.text_ctrl.fit_controller
        self.loader = patch.object(self.ctrl, '_original_image', return_value=(self.source, ''))
        self.read_original = self.loader.start()

    def tearDown(self):
        self.loader.stop()
        existing.FitTextWorkflowTests.tearDown(self)

    _dialog_action = existing.FitTextWorkflowTests._dialog_action

    def apply_bubble(self):
        def interact(dialog):
            dialog.mode.setCurrentIndex(1)
            self.assertIsNotNone(dialog.boundary, dialog.boundary_note)
            self.assertTrue(dialog.apply_button.isEnabled(), dialog.status.text())
            QtTest.QTest.mouseClick(dialog.apply_button, QtCore.Qt.MouseButton.LeftButton)
        self._dialog_action(interact)

    def test_apply_moves_only_lettering_and_is_one_undo_step(self):
        before = TextItemProperties.from_text_item(self.item).to_dict()
        source = self.source.copy()
        bounds = self.block.xyxy.copy()
        bubble = self.block.bubble_xyxy.copy()
        count = self.stack.count()
        self.apply_bubble()
        after = TextItemProperties.from_text_item(self.item).to_dict()
        self.assertEqual(self.stack.count(), count + 1)
        self.assertEqual(after['layout_preferences']['mode'], 'bubble')
        self.assertNotEqual(after['position'], before['position'])
        self.assertEqual(self.item.toPlainText(), 'Bonjour mon ami!')
        np.testing.assert_array_equal(self.source, source)
        np.testing.assert_array_equal(self.block.xyxy, bounds)
        np.testing.assert_array_equal(self.block.bubble_xyxy, bubble)
        self.assertFalse(self.main.applied_patches)
        self.stack.undo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), before)
        self.stack.redo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), after)

    def test_cancel_preserves_layout_and_fallback_remains_usable(self):
        self.read_original.return_value = (None, 'missing_image')
        before = TextItemProperties.from_text_item(self.item).to_dict()
        def interact(dialog):
            dialog.mode.setCurrentIndex(1)
            self.assertIsNone(dialog.boundary)
            self.assertIn('Using Box', dialog.boundary_status.text())
            self.assertEqual(dialog.preferences()['mode'], 'box')
            self.assertTrue(dialog.apply_button.isEnabled())
            dialog.reject()
        self._dialog_action(interact)
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), before)

    def test_manual_ellipse_works_without_original_and_persists_override(self):
        self.read_original.return_value = (None, 'missing_image')
        def interact(dialog):
            dialog.boundary_shape.setCurrentIndex(2)
            dialog.mode.setCurrentIndex(1)
            self.assertEqual(dialog.boundary.source, 'ellipse')
            dialog.padding.setValue(12)
            dialog.rebuild()
            self.assertTrue(dialog.apply_button.isEnabled())
            dialog.accept()
        self._dialog_action(interact)
        saved = TextItemProperties.from_dict(TextItemProperties.from_text_item(self.item).to_dict())
        self.assertEqual(saved.layout_preferences['boundary_shape'], 'ellipse')
        self.assertEqual(saved.layout_preferences['padding'], 12)
        self.assertEqual(saved.layout_preferences['boundary_box'], [20, 20, 220, 180])
        self.assertEqual(self.read_original.call_count, 0)
        restored = self.main.image_viewer.add_text_item(saved)
        self.assertEqual(restored.toHtml(), self.item.toHtml())
        self.assertEqual(restored.pos(), self.item.pos())
        renderer = ImageSaveRenderer(self.source)
        renderer.add_state_to_image({'text_items_state': [saved.to_dict()]})
        rendered = [obj for obj in renderer.scene.items() if hasattr(obj, 'toPlainText')]
        self.assertEqual(rendered[0].pos(), self.item.pos())
        self.assertEqual(rendered[0].toHtml(), self.item.toHtml())

    def test_rotated_and_vertical_items_use_explicit_box_fallback(self):
        for vertical, rotation, scale in ((False, 15, 1), (True, 0, 1), (False, 0, 2)):
            self.item.set_vertical(vertical)
            self.item.setRotation(rotation)
            self.item.setScale(scale)
            dialog = TextFitDialog(self.item, 100, 80, 4, 24)
            try:
                dialog.mode.setCurrentIndex(1)
                self.assertIsNone(dialog.boundary)
                self.assertIn('Using Box', dialog.boundary_status.text())
                self.assertEqual(dialog.preferences()['mode'], 'box')
            finally:
                dialog.timer.stop()
                dialog.deleteLater()

    def test_move_after_fit_rechecks_real_boundary_without_reloading_pixels(self):
        self.apply_bubble()
        self.ctrl.refresh()
        reads = self.read_original.call_count
        self.item.setPos(0, 0)
        self.ctrl.refresh()
        self.assertIn('overflows', self.main.text_fit_status.text())
        self.assertEqual(self.read_original.call_count, reads)

    def test_manual_override_outside_page_cannot_move_text_off_canvas(self):
        before = self.item.pos()
        def interact(dialog):
            dialog.boundary_shape.setCurrentIndex(1)
            dialog.mode.setCurrentIndex(1)
            dialog.boundary_left.setValue(500)
            dialog.rebuild()
            self.assertIsNone(dialog.boundary)
            self.assertIn('outside the page', dialog.boundary_status.text())
            self.assertEqual(dialog.candidate_page_position(), (before.x(), before.y()))
            dialog.reject()
        self._dialog_action(interact)
        self.assertEqual(self.item.pos(), before)

    def test_processing_disables_fit_and_source_change_invalidates_preview(self):
        self.main.task_runner_ctrl = SimpleNamespace(is_processing_queue=True)
        self.ctrl.refresh()
        self.assertFalse(self.main.fit_text_button.isEnabled())
        self.assertIn('processing', self.main.text_fit_status.text())
        self.main.task_runner_ctrl.is_processing_queue = False
        before = self.item.toHtml()
        def interact(dialog):
            dialog.mode.setCurrentIndex(1)
            self.block.bubble_xyxy[0] += 3
            dialog.accept()
        with patch.object(QtWidgets.QMessageBox, 'information') as notice:
            self._dialog_action(interact)
        self.assertEqual(notice.call_count, 1)
        self.assertEqual(self.item.toHtml(), before)

    def test_lazy_original_materialization_does_not_invalidate_first_preview(self):
        ready = [False]
        def materialize():
            ready[0] = True
            return self.source, ''
        self.read_original.side_effect = lambda _path: materialize()
        with patch.object(self.ctrl, '_file_stamp', side_effect=lambda path: (123, 456) if ready[0] else None):
            with patch.object(QtWidgets.QMessageBox, 'information') as notice:
                self.apply_bubble()
        self.assertEqual(notice.call_count, 0)
        self.assertEqual(self.item.layout_preferences['mode'], 'bubble')

    def test_original_loader_bypasses_processed_image_cache(self):
        self.loader.stop()
        self.main.image_data['page.png'][:] = 0
        with patch('app.path_materialization.ensure_path_materialized'), patch('imkit.read_image', return_value=self.source) as read:
            actual, reason = self.ctrl._original_image('page.png')
        self.loader.start()
        self.assertFalse(reason)
        self.assertFalse(actual.flags.writeable)
        self.assertEqual(read.call_args.args, ('page.png',))
        np.testing.assert_array_equal(actual, self.source)

    def test_webtoon_origin_subtracts_horizontal_centering_as_well_as_y(self):
        self.main.image_viewer.webtoon_manager.layout_manager.image_positions = [0, 400]
        self.main.image_files.append('second.png')
        self.item._page_path = 'second.png'
        self.item.setPos(95, 465)
        self.block.xyxy += [30, 400, 30, 400]
        self.block.bubble_xyxy += [30, 400, 30, 400]
        context = self.ctrl._boundary_context(self.item, ('second.png', QtCore.QPointF(30, 400), self.stack))
        self.assertEqual(context['position'], (65, 65))
        self.assertEqual(context['text_bounds'], [65, 65, 165, 125])
        self.assertEqual(context['bubble_bounds'], [20, 20, 220, 180])

    def test_webtoon_loading_without_page_extents_disables_fit(self):
        self.main.image_viewer.webtoon_manager.layout_manager.image_positions = [0, 400]
        self.ctrl.refresh()
        self.assertFalse(self.main.fit_text_button.isEnabled())


if __name__ == '__main__':
    unittest.main()
