import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from testmanual_text_box import _ManualTextMain
from app.controllers.text_fit import FitTextCommand
from app.ui.dialogs.text_fit import TextFitDialog
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.save_renderer import ImageSaveRenderer
from modules.utils.textblock import TextBlock


class FitTextWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.main.fit_text_button = QtWidgets.QPushButton()
        self.main.text_fit_status = QtWidgets.QLabel()
        self.block = TextBlock(text_bbox=np.array([10, 10, 130, 90]), is_manual=True,
                               translation='Bonjour mon ami!')
        self.main.blk_list.append(self.block)
        self.item = self.main.text_ctrl.create_manual_text_item(self.block, self.block.translation)
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(8)
        cursor.setPosition(11, QtGui.QTextCursor.MoveMode.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Weight.Bold)
        fmt.setFontItalic(True)
        fmt.setFontUnderline(True)
        cursor.mergeCharFormat(fmt)
        self.main.text_ctrl._commit_pending_text_command()
        self.stack = self.main.undo_stacks['page.png']

    def tearDown(self):
        self.main.deleteLater()
        self.app.processEvents()

    def _dialog_action(self, callback):
        errors = []
        def interact():
            dialog = self.app.activeModalWidget()
            try:
                self.assertIsInstance(dialog, TextFitDialog)
                callback(dialog)
            except Exception as exc:
                errors.append(exc)
                if dialog is not None:
                    dialog.reject()
        QtCore.QTimer.singleShot(0, interact)
        self.main.text_ctrl.fit_controller.open_preview()
        if errors:
            raise errors[0]

    def _apply(self):
        def interact(dialog):
            self.assertTrue(dialog.apply_button.isEnabled(), dialog.status.text())
            QtTest.QTest.mouseClick(dialog.apply_button, QtCore.Qt.MouseButton.LeftButton)
        self._dialog_action(interact)

    def test_preview_cancel_does_not_change_document_geometry_or_undo(self):
        before = TextItemProperties.from_text_item(self.item).to_dict()
        count = self.stack.count()
        def cancel(dialog):
            dialog.area_width.setValue(90)
            dialog.rebuild()
            self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), before)
            QtTest.QTest.mouseClick(dialog.cancel_button, QtCore.Qt.MouseButton.LeftButton)
        self._dialog_action(cancel)
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), before)
        self.assertEqual(self.stack.count(), count)

    def test_apply_is_one_undo_step_preserving_words_formatting_and_ocr_bounds(self):
        before = TextItemProperties.from_text_item(self.item).to_dict()
        bounds = self.block.xyxy.copy()
        count = self.stack.count()
        self._apply()
        after = TextItemProperties.from_text_item(self.item).to_dict()
        self.assertEqual(self.stack.count(), count + 1)
        self.assertEqual(self.item.toPlainText(), 'Bonjour mon ami!')
        self.assertEqual(self.block.region_id, self.item.region_id)
        np.testing.assert_array_equal(self.block.xyxy, bounds)
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(9)
        self.assertTrue(cursor.charFormat().fontItalic())
        self.assertTrue(cursor.charFormat().fontUnderline())
        self.assertGreaterEqual(cursor.charFormat().fontWeight(), int(QtGui.QFont.Weight.Bold))
        self.stack.undo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), before)
        self.stack.redo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), after)

    def test_impossible_fit_disabled_and_invalid_limits_do_not_accept(self):
        dialog = TextFitDialog(self.item, 3, 3, 12, 72)
        try:
            self.assertFalse(dialog.apply_button.isEnabled())
            self.assertIn('Does not fit', dialog.status.text())
            dialog.minimum.setValue(100)
            dialog.maximum.setValue(10)
            dialog.rebuild()
            self.assertFalse(dialog.apply_button.isEnabled())
        finally:
            dialog.deleteLater()

    def test_stale_preview_does_not_overwrite_new_text(self):
        count = self.stack.count()
        def interact(dialog):
            self.item.setPlainText('Newer text')
            QtTest.QTest.mouseClick(dialog.apply_button, QtCore.Qt.MouseButton.LeftButton)
        with patch.object(QtWidgets.QMessageBox, 'information') as notice:
            self._dialog_action(interact)
        notice.assert_called_once()
        self.assertEqual(self.item.toPlainText(), 'Newer text')
        self.assertEqual(self.stack.count(), count)

    def test_save_restore_and_flat_export_keep_fitted_geometry_and_font(self):
        self._apply()
        properties = TextItemProperties.from_text_item(self.item).to_dict()
        expected_html = self.item.toHtml()
        expected_size = self.item.boundingRect().size()
        self.main.image_viewer._scene.removeItem(self.item)
        self.main.image_viewer.text_items.remove(self.item)
        restored = self.main.image_viewer.add_text_item(properties)
        self.assertEqual(restored.toHtml(), expected_html)
        self.assertEqual(restored.boundingRect().size(), expected_size)
        renderer = ImageSaveRenderer(np.full((100, 140, 3), 255, dtype=np.uint8))
        renderer.add_state_to_image({'text_items_state': [properties]})
        text = next(i for i in renderer.scene.items() if hasattr(i, 'region_id'))
        self.assertEqual(text.toHtml(), expected_html)
        self.assertEqual(text.boundingRect().size(), expected_size)
        self.assertTrue(np.any(renderer.render_to_image() < 255))
        self.stack.undo()
        self.assertEqual(restored.toPlainText(), 'Bonjour mon ami!')
        self.assertEqual(restored.layout_preferences, {})
        self.stack.redo()
        self.assertEqual(restored.toHtml(), expected_html)

    def test_unloaded_page_undo_changes_saved_state_without_other_page(self):
        before = TextItemProperties.from_text_item(self.item).to_dict()
        self._apply()
        self.main.image_viewer._scene.removeItem(self.item)
        self.main.image_viewer.text_items.remove(self.item)
        self.main.image_files.append('other.png')
        self.main.curr_img_idx = 1
        self.main.blk_list = []
        other = self.main.image_viewer.add_text_item(TextItemProperties(text='Unchanged', width=90,
                                                                       text_color=QtGui.QColor('black')))
        self.stack.undo()
        saved = self.main.image_states['page.png']['viewer_state']['text_items_state'][0]
        self.assertEqual(saved['text'], before['text'])
        self.assertEqual(other.toPlainText(), 'Unchanged')

    def test_fit_keeps_resize_and_inline_edit_available(self):
        self._apply()
        width = self.item.boundingRect().width()
        self.item.resize_handle = 'right'
        self.item.resize_start = self.item.mapToScene(QtCore.QPointF(width, 10))
        self.item.resize_item(self.item.resize_start + QtCore.QPointF(3, 0))
        self.assertGreater(self.item.boundingRect().width(), width)
        self.item.enter_editing_mode()
        cursor = self.item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(7, QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.item.setTextCursor(cursor)
        self.item.set_bold(True)
        self.assertEqual(self.item.toPlainText(), 'Bonjour mon ami!')

    def test_warning_uses_region_bounds_not_zoom_or_document_grown_height(self):
        self.block.xyxy = np.array([10, 10, 25, 13])
        ctrl = self.main.text_ctrl.fit_controller
        ctrl.refresh()
        self.assertIn('overflows', self.main.text_fit_status.text())
        self.main.image_viewer.scale(3, 3)
        ctrl.refresh()
        self.assertIn('overflows', self.main.text_fit_status.text())

    def test_item_scale_is_included_in_available_space(self):
        ctrl = self.main.text_ctrl.fit_controller
        self.item.setScale(2)
        self.assertEqual(ctrl._limits(self.item)[:2], (60, 40))

    def _webtoon(self):
        viewer = self.main.image_viewer
        viewer.webtoon_manager.layout_manager.image_positions = [0, 400]
        self.main.webtoon_mode = True
        self.main.image_files.append('second.png')
        viewer.scene_to_page_coordinates = lambda point: (
            int(point.y() // 400) if 0 <= point.y() < 800 and 0 <= point.x() < 500 else None,
            QtCore.QPointF(point.x(), point.y() % 400))
        viewer.page_to_scene_coordinates = lambda index, point: QtCore.QPointF(point.x(), point.y() + index * 400)

    def test_webtoon_seam_is_guarded_without_changing_text(self):
        self._webtoon()
        self.item.setPos(10, 395)
        original = self.item.toHtml()
        ctrl = self.main.text_ctrl.fit_controller
        self.assertIsNone(ctrl._page(self.item))
        ctrl.refresh()
        self.assertFalse(self.main.fit_text_button.isEnabled())
        self.assertIn('contained within one page', self.main.text_fit_status.text())
        self.assertEqual(self.item.toHtml(), original)

    def test_webtoon_undo_selects_owner_among_duplicate_logical_ids(self):
        self._webtoon()
        viewer = self.main.image_viewer
        first = self.item
        first_html = first.toHtml()
        first.selected = False
        first.setSelected(False)
        props = TextItemProperties.from_text_item(first)
        props.position = (10, 410)
        second = viewer.add_text_item(props)
        second._page_path = 'second.png'
        second.selected = True
        second.setSelected(True)
        self.main.curr_img_idx = 1
        self.main.curr_tblock_item = second
        second_stack = QtGui.QUndoStack(self.main)
        self.main.undo_stacks['second.png'] = second_stack
        self.main.image_states['second.png'] = {'blk_list': [self.block.deep_copy()]}
        self._apply()
        self.assertEqual(first.toHtml(), first_html)
        self.assertTrue(second.layout_preferences)
        saved = self.main.image_states['second.png']['viewer_state']['text_items_state'][0]
        self.assertEqual(saved['position'], (10, 10))
        second_stack.undo()
        self.assertEqual(first.toHtml(), first_html)
        self.assertEqual(second.toHtml(), first_html)
        self.assertEqual(second.pos(), QtCore.QPointF(10, 410))

    def test_current_vertical_preview_retains_original_dimensions(self):
        self.item.set_vertical(True)
        self.item.layout.set_max_size(150, 180)
        before = self.item.boundingRect().size()
        dialog = TextFitDialog(self.item, 80, 90, 4, 40)
        try:
            dialog.choice.setCurrentIndex(0)
            self.assertEqual(dialog.preview_item.boundingRect().size(), before)
            self.assertEqual(dialog.preview_item.toPlainText(), self.item.toPlainText())
            self.assertEqual(dialog.document.documentLayout().max_height, 180)
        finally:
            dialog.deleteLater()


if __name__ == '__main__':
    unittest.main()
