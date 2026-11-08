"""Stable-region transform application, cancellation, and page-local undo."""
import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from testmanual_text_box import _ManualTextMain
from app.controllers.lettering_transform import LetteringTransformController
from app.ui.dialogs.lettering_transform import LetteringTransformDialog
from app.ui.canvas.text.text_item_properties import TextItemProperties
from modules.utils.region_metadata import new_id
from modules.utils.textblock import TextBlock


class LetteringTransformWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.main = _ManualTextMain()
        self.main.lettering_transform_button = QtWidgets.QPushButton()
        self.main._translation_project_epoch = 1
        self.main.image_states['page.png']['page_id'] = new_id()
        self.ctrl = LetteringTransformController(self.main)
        self.notices = []
        self.ctrl._notice = self.notices.append
        self.block = TextBlock(text_bbox=np.array([10, 10, 130, 70]), is_manual=True, translation='One bold word')
        self.main.blk_list.append(self.block)
        self.item = self.main.text_ctrl.create_manual_text_item(self.block, self.block.translation)
        self.item.region_id = self.block.region_id
        self.item._page_path = 'page.png'
        self.item.selected = True
        cursor = QtGui.QTextCursor(self.item.document())
        cursor.setPosition(4)
        cursor.setPosition(8, QtGui.QTextCursor.KeepAnchor)
        fmt = QtGui.QTextCharFormat()
        fmt.setFontWeight(QtGui.QFont.Bold)
        cursor.mergeCharFormat(fmt)
        self.main.text_ctrl._commit_pending_text_command()
        self.stack = self.main.undo_stacks['page.png']
        self.before = TextItemProperties.from_text_item(self.item).to_dict()

    def tearDown(self):
        self.ctrl.shutdown()
        self.main.deleteLater()
        self.app.processEvents()

    def run_editor(self, accept=True, during=None, reset=False, arc_angle=None):
        def create(properties, parent=None):
            dialog = LetteringTransformDialog(properties, parent)
            def execute():
                if reset:
                    dialog.reset_button.click()
                else:
                    dialog.controls['scale_x'].setValue(140)
                    dialog.controls['perspective_y'].setValue(25)
                    if arc_angle is not None:
                        dialog.controls['arc_angle'].setValue(arc_angle)
                if during:
                    during()
                if accept:
                    dialog.apply_selection()
                else:
                    dialog.reject()
                return dialog.result()
            dialog.exec = execute
            return dialog
        with patch('app.controllers.lettering_transform.LetteringTransformDialog', side_effect=create):
            self.ctrl.open_editor()

    def test_apply_is_one_action_and_never_changes_ocr_or_text(self):
        count = self.stack.count()
        bbox = self.block.xyxy.copy()
        self.run_editor()
        self.assertEqual(self.stack.count(), count + 1)
        self.assertEqual(self.stack.undoText(), 'Transform lettering')
        self.assertFalse(self.item.transform().isAffine())
        self.assertEqual(self.item.toHtml(), self.before['text'])
        np.testing.assert_array_equal(self.block.xyxy, bbox)
        after = TextItemProperties.from_text_item(self.item).to_dict()
        self.stack.undo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)
        self.stack.redo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), after)

    def test_cancel_does_not_mutate_page_or_undo_stack(self):
        count = self.stack.count()
        self.run_editor(accept=False)
        self.assertEqual(self.stack.count(), count)
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)

    def test_project_change_rejects_stale_preview(self):
        count = self.stack.count()
        self.run_editor(during=lambda: setattr(self.main, '_translation_project_epoch', 2))
        self.assertEqual(self.stack.count(), count)
        self.assertTrue(self.item.transform().isIdentity())
        self.assertTrue(self.notices)

    def test_text_edit_rejects_stale_preview(self):
        self.run_editor(during=lambda: self.item.setPlainText('Edited while preview open'))
        self.assertEqual(self.item.toPlainText(), 'Edited while preview open')
        self.assertTrue(self.item.transform().isIdentity())
        self.assertTrue(self.notices)

    def test_offpage_undo_updates_stored_state_and_reloaded_item(self):
        self.run_editor()
        self.main.image_files.append('second.png')
        self.main.image_states['second.png'] = {'page_id': new_id()}
        self.main.curr_img_idx = 1
        self.main.image_viewer._scene.removeItem(self.item)
        self.main.image_viewer.text_items.clear()
        self.stack.undo()
        saved = self.main.image_states['page.png']['viewer_state']['text_items_state'][0]
        self.assertEqual(saved, self.before)
        self.assertNotIn('viewer_state', self.main.image_states['second.png'])
        self.main.curr_img_idx = 0
        restored = self.main.image_viewer.add_text_item(TextItemProperties.from_dict(saved))
        restored._page_path = 'page.png'
        self.stack.redo()
        self.assertFalse(restored.transform().isAffine())

    def test_reset_distortion_is_undoable(self):
        self.run_editor()
        transform = QtGui.QTransform(self.item.transform())
        self.run_editor(reset=True)
        self.assertTrue(self.item.transform().isIdentity())
        self.stack.undo()
        self.assertEqual(self.item.transform(), transform)

    def test_no_selection_or_busy_processing_prevents_dialog(self):
        self.item.selected = False
        with patch('app.controllers.lettering_transform.LetteringTransformDialog') as dialog:
            self.ctrl.open_editor()
            self.main._batch_active = True
            self.ctrl.open_editor()
            dialog.assert_not_called()
        self.assertEqual(len(self.notices), 2)

    def test_curved_apply_and_reset_are_exactly_undoable_without_cleanup_changes(self):
        count = self.stack.count()
        bbox = self.block.xyxy.copy()
        self.run_editor(arc_angle=70)
        self.assertEqual(self.stack.count(), count + 1)
        self.assertEqual(self.item.lettering_transform['arc_angle'], 70)
        self.assertIs(self.item.layout, self.item.document().documentLayout())
        self.assertEqual(self.item.toHtml(), self.before['text'])
        curved = TextItemProperties.from_text_item(self.item).to_dict()
        self.stack.undo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), self.before)
        self.stack.redo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), curved)
        self.run_editor(reset=True)
        self.assertEqual(self.item.lettering_transform, {})
        self.stack.undo()
        self.assertEqual(TextItemProperties.from_text_item(self.item).to_dict(), curved)
        np.testing.assert_array_equal(self.block.xyxy, bbox)

    def test_curved_redo_after_page_reload_restores_layout_and_rich_text(self):
        self.run_editor(arc_angle=-60)
        curved = TextItemProperties.from_text_item(self.item).to_dict()
        self.main.image_viewer._scene.removeItem(self.item)
        self.main.image_viewer.text_items.clear()
        self.stack.undo()
        saved = self.main.image_states['page.png']['viewer_state']['text_items_state'][0]
        self.assertEqual(saved, self.before)
        restored = self.main.image_viewer.add_text_item(TextItemProperties.from_dict(saved))
        restored._page_path = 'page.png'
        self.stack.redo()
        self.assertEqual(TextItemProperties.from_text_item(restored).to_dict(), curved)
        self.assertIs(restored.layout, restored.document().documentLayout())
